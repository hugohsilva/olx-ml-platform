from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from docker.errors import DockerException, ImageNotFound

from application.errors import ImageBuildError
from application.ports import ImageBuildRequest
from infrastructure.config import EgressPolicyError, PlatformConfig

_PROXY_PORT = 3128
_POLICY_PROBE_SOURCE = """\
import json
import os
import socket

proxy_host = os.environ["MLPLATFORM_PROXY_HOST"]
proxy_port = int(os.environ["MLPLATFORM_PROXY_PORT"])
allowed = json.loads(os.environ["MLPLATFORM_ALLOWED_REPOSITORIES"])

def connect_through_proxy(host, port, expected):
    with socket.create_connection((proxy_host, proxy_port), timeout=5) as proxy:
        proxy.sendall(
            ("CONNECT " + host + ":" + str(port)
             + " HTTP/1.1\\r\\nHost: " + host + "\\r\\n\\r\\n").encode("ascii")
        )
        response = proxy.recv(1024).split(b" ", 2)
        if len(response) < 2 or int(response[1]) != expected:
            raise SystemExit("proxy returned an unexpected policy result")

for repository in allowed:
    host, separator, port = repository.rpartition(":")
    if not separator:
        host, port = repository, "443"
    connect_through_proxy(host, int(port), 200)

connect_through_proxy("not-allowlisted.invalid", 443, 403)
try:
    socket.create_connection(("1.1.1.1", 443), timeout=2).close()
except OSError:
    pass
else:
    raise SystemExit("direct build-network egress is not blocked")
"""

_PROXY_SOURCE = """\
import json
import os
import select
import socket
import socketserver
from urllib.parse import urlsplit

allowed = {}
for entry in json.loads(os.environ["MLPLATFORM_ALLOWED_REPOSITORIES"]):
    host, separator, port = entry.rpartition(":")
    if not separator or "]" in host:
        host, port = entry, ""
    allowed[host.rstrip(".").lower()] = int(port) if port else None

def permitted(host, port):
    host = host.rstrip(".").lower()
    allowed_port = allowed.get(host, -1)
    if allowed_port is None:
        return port in (80, 443)
    return allowed_port == port

def tunnel(client, upstream):
    client.setblocking(False)
    upstream.setblocking(False)
    sockets = (client, upstream)
    while True:
        readable, _, exceptional = select.select(sockets, (), sockets, 60)
        if exceptional or not readable:
            return
        for source in readable:
            try:
                data = source.recv(65536)
            except OSError:
                return
            if not data:
                return
            target = upstream if source is client else client
            try:
                target.sendall(data)
            except OSError:
                return

class Handler(socketserver.StreamRequestHandler):
    def reply(self, status, message):
        body = message.encode("utf-8")
        self.wfile.write(
            ("HTTP/1.1 " + status + "\\r\\nConnection: close\\r\\nContent-Length: "
             + str(len(body)) + "\\r\\n\\r\\n").encode("ascii") + body
        )

    def handle(self):
        self.connection.settimeout(15)
        line = self.rfile.readline(8192).decode("ascii", "replace").strip()
        parts = line.split()
        if len(parts) != 3:
            self.reply("400 Bad Request", "invalid proxy request")
            return
        method, target, _ = parts
        headers = []
        while True:
            header = self.rfile.readline(8192)
            if not header or header in (b"\\r\\n", b"\\n"):
                break
            headers.append(header)
        if method.upper() == "CONNECT":
            parsed = urlsplit("//" + target)
            host, port = parsed.hostname, parsed.port
            if not host or port is None or not permitted(host, port):
                self.reply("403 Forbidden", "destination is not allowlisted")
                return
            request_tail = b""
        else:
            parsed = urlsplit(target)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            if parsed.scheme != "http" or not host or not permitted(host, port):
                self.reply("403 Forbidden", "destination is not allowlisted")
                return
            request_tail = (
                (method + " " + (parsed.path or "/")
                 + (("?" + parsed.query) if parsed.query else "") + " HTTP/1.1\\r\\n").encode("ascii")
                + b"".join(
                    header for header in headers
                    if not header.lower().startswith((b"proxy-", b"connection:"))
                )
                + b"Connection: close\\r\\n\\r\\n"
            )
        try:
            upstream = socket.create_connection((host, port), timeout=10)
        except OSError:
            self.reply("502 Bad Gateway", "allowlisted destination is unavailable")
            return
        with upstream:
            if method.upper() == "CONNECT":
                self.wfile.write(b"HTTP/1.1 200 Connection Established\\r\\n\\r\\n")
                tunnel(self.connection, upstream)
            else:
                upstream.sendall(request_tail)
                tunnel(self.connection, upstream)

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

with Server(("0.0.0.0", 3128), Handler) as server:
    server.serve_forever()
"""


@dataclass(slots=True)
class DockerBuildEgressPolicy:
    config: PlatformConfig
    request: ImageBuildRequest
    image: str
    docker_client: Any
    internal_network_name: str
    outbound_network_name: str
    proxy_url: str = ""
    policy_id: str = ""
    _proxy_container: Any | None = None
    _internal_network: Any | None = None
    _outbound_network: Any | None = None

    @classmethod
    def for_run(
        cls,
        config: PlatformConfig,
        request: ImageBuildRequest,
        image: str,
        docker_client: Any,
    ) -> DockerBuildEgressPolicy:
        suffix = request.run_id.hex[:20]
        return cls(
            config=config,
            request=request,
            image=image,
            docker_client=docker_client,
            internal_network_name=f"mlplatform-build-{suffix}",
            outbound_network_name=f"mlplatform-egress-{suffix}",
        )

    def start(self) -> DockerBuildEgressPolicy:
        try:
            try:
                self.docker_client.images.get(self.image)
            except ImageNotFound as exc:
                raise EgressPolicyError(
                    "The trusted Python base image must be available locally before build "
                    "egress can be isolated."
                ) from exc
            self._internal_network = self.docker_client.networks.create(
                self.internal_network_name,
                driver="bridge",
                internal=True,
                labels={"mlplatform.run_id": str(self.request.run_id)},
            )
            self._outbound_network = self.docker_client.networks.create(
                self.outbound_network_name,
                driver="bridge",
                labels={"mlplatform.run_id": str(self.request.run_id)},
            )
            self._proxy_container = self.docker_client.containers.create(
                image=self.image,
                name=f"mlplatform-egress-proxy-{self.request.run_id.hex[:20]}",
                entrypoint=["python"],
                command=["-c", _PROXY_SOURCE],
                user="65532:65532",
                environment={
                    "MLPLATFORM_ALLOWED_REPOSITORIES": json.dumps(
                        list(self.request.allowed_repositories)
                    )
                },
                labels={"mlplatform.run_id": str(self.request.run_id)},
                network=self.outbound_network_name,
                nano_cpus=max(1, round(self.request.cpus * 1_000_000_000)),
                mem_limit=self.request.memory_limit,
                read_only=True,
                tmpfs={"/tmp": "rw,noexec,nosuid,size=16m"},
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                privileged=False,
            )
            self._proxy_container.start()
            self._internal_network.connect(self._proxy_container)
            self._proxy_container.reload()
            proxy_address = self._proxy_container.attrs["NetworkSettings"]["Networks"][
                self.internal_network_name
            ]["IPAddress"]
            if not proxy_address:
                raise EgressPolicyError(
                    "The egress proxy has no address on the isolated build network."
                )
            self.proxy_url = f"http://{proxy_address}:{_PROXY_PORT}"
            self.policy_id = f"docker-internal-proxy:{self.internal_network_name}"
            self._verify_effective_profile()
            self._verify_network_policy()
            return self
        except (DockerException, KeyError, TypeError, EgressPolicyError) as exc:
            self.close()
            raise ImageBuildError(
                "A verifiable Docker build egress policy could not be applied.",
                run_id=self.request.run_id,
                logs=(str(exc),),
            ) from exc

    def _verify_effective_profile(self) -> None:
        if self._internal_network is None or self._proxy_container is None:
            raise EgressPolicyError("The isolated build network or egress proxy is unavailable.")
        self._internal_network.reload()
        self._proxy_container.reload()
        if self._internal_network.attrs.get("Internal") is not True:
            raise EgressPolicyError("The build network is not isolated from external networks.")
        host_config = self._proxy_container.attrs["HostConfig"]
        if (
            host_config.get("Privileged") is not False
            or host_config.get("CapDrop") != ["ALL"]
            or "no-new-privileges:true" not in (host_config.get("SecurityOpt") or [])
            or host_config.get("ReadonlyRootfs") is not True
            or host_config.get("NanoCpus") != max(1, round(self.request.cpus * 1_000_000_000))
        ):
            raise EgressPolicyError("The egress proxy security profile was not applied.")
        if host_config.get("Binds") or any(
            mount.get("Type") == "bind"
            for mount in self._proxy_container.attrs.get("Mounts", [])
            if isinstance(mount, dict)
        ):
            raise EgressPolicyError("The egress proxy unexpectedly mounted a host directory.")

    def _verify_network_policy(self) -> None:
        if not self.proxy_url or self._internal_network is None:
            raise EgressPolicyError("The isolated build egress policy is not ready.")
        proxy_host = self.proxy_url.removeprefix("http://").rsplit(":", 1)[0]
        probe = None
        try:
            probe = self.docker_client.containers.create(
                image=self.image,
                name=f"mlplatform-egress-check-{self.request.run_id.hex[:20]}",
                entrypoint=["python"],
                command=["-c", _POLICY_PROBE_SOURCE],
                user="65532:65532",
                environment={
                    "MLPLATFORM_ALLOWED_REPOSITORIES": json.dumps(
                        list(self.request.allowed_repositories)
                    ),
                    "MLPLATFORM_PROXY_HOST": proxy_host,
                    "MLPLATFORM_PROXY_PORT": str(_PROXY_PORT),
                },
                labels={"mlplatform.run_id": str(self.request.run_id)},
                network=self.internal_network_name,
                nano_cpus=max(1, round(self.request.cpus * 1_000_000_000)),
                mem_limit=self.request.memory_limit,
                read_only=True,
                tmpfs={"/tmp": "rw,noexec,nosuid,size=16m"},
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                privileged=False,
            )
            probe.start()
            result = probe.wait(timeout=min(15, self.request.timeout_seconds))
            output = probe.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            if result.get("StatusCode") != 0:
                raise EgressPolicyError(
                    "Docker did not enforce the allowlisted build egress policy."
                    + (f" Probe output: {output.strip()}" if output.strip() else "")
                )
        except DockerException as exc:
            raise EgressPolicyError(f"Docker egress policy verification failed: {exc}") from exc
        finally:
            if probe is not None:
                try:
                    probe.remove(force=True)
                except DockerException as exc:
                    raise EgressPolicyError(
                        "The temporary egress probe could not be removed."
                    ) from exc

    def close(self) -> None:
        errors: list[Exception] = []
        if self._proxy_container is not None:
            try:
                self._proxy_container.remove(force=True)
            except DockerException as exc:
                errors.append(exc)
            self._proxy_container = None
        for network in (self._outbound_network, self._internal_network):
            if network is None:
                continue
            try:
                network.remove()
            except DockerException as exc:
                errors.append(exc)
        self._outbound_network = None
        self._internal_network = None
        if errors:
            raise ImageBuildError(
                "Temporary Docker egress policy resources could not be removed.",
                run_id=self.request.run_id,
                logs=tuple(str(error) for error in errors),
            ) from errors[0]
