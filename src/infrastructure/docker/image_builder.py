from __future__ import annotations

import io
import re
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import docker
from docker.errors import DockerException

from application.errors import ImageBuildError, ImageBuildTimeout
from application.ports import ImageBuildRequest, ImageBuildResult
from infrastructure.config import PlatformConfig
from infrastructure.docker.egress_policy import DockerBuildEgressPolicy

_BUILD_TIMEOUT_BUFFER_SECONDS = 10
_MAX_BUILD_LOG_LINE_LENGTH = 1000
_PACKAGE_VERSION_PATTERN = re.compile(r"([A-Za-z0-9_.-]+)==([^\s=]+)\Z")
_URL_CREDENTIAL_PATTERN = re.compile(r"(https?://)[^/@\s]+@")
_SECRET_PATTERN = re.compile(r"(?i)\b(password|token|secret|api[_-]?key)\b(\s*[=:]\s*)\S+")
_ROOTLESS_BUILDKIT_IMAGE = "moby/buildkit:rootless"

_INSTALLER = """\
import ctypes
import os
import sys

if os.geteuid() == 0:
    raise SystemExit("dependency installation must not run as root")

libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(38, 1, 0, 0, 0) != 0:
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number))

with open("/proc/self/status", encoding="ascii") as status_file:
    status = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in status_file
        if ":" in line
    }

if status.get("NoNewPrivs") != "1":
    raise SystemExit("no-new-privileges is not effective for dependency installation")
if any(status.get(name) != "0000000000000000" for name in ("CapEff", "CapPrm", "CapInh")):
    raise SystemExit("dependency installation has effective, permitted, or inheritable capabilities")

os.execv(
    sys.executable,
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--user",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "-r",
        "/tmp/requirements.txt",
    ],
)
"""

_DOCKERFILE_SECURITY_CHECK = """\
RUN python -c 'import os; s=dict(line.split(":", 1) for line in open("/proc/self/status") if ":" in line); assert os.geteuid() != 0; assert s.get("CapEff", "").strip() == "0000000000000000"; print("build security profile verified")'
"""


class DockerImageBuilder:
    def __init__(
        self,
        config: PlatformConfig,
        *,
        docker_client: Any | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        egress_policy_factory: Callable[
            [PlatformConfig, ImageBuildRequest, str, Any], DockerBuildEgressPolicy
        ]
        | None = None,
    ) -> None:
        self._config = config
        self._docker_client = docker_client
        self._command_runner = command_runner
        self._egress_policy_factory = egress_policy_factory or (
            lambda platform_config, request, image, client: DockerBuildEgressPolicy.for_run(
                platform_config,
                request,
                image,
                client,
            )
        )

    def build(self, request: ImageBuildRequest) -> ImageBuildResult:
        deadline = time.monotonic() + request.timeout_seconds
        if (
            request.base_image_registry != self._config.base_image_registry
            or request.allowed_repositories != self._config.build_allowed_repositories
        ):
            raise ImageBuildError(
                "Image build repositories do not match the verified local configuration.",
                run_id=request.run_id,
            )

        base_image = _base_image(request.base_image_registry, request.python_version)
        policy = self._egress_policy_factory(
            self._config,
            request,
            base_image,
            self._docker(),
        ).start()
        builder_name = f"mlplatform-{request.run_id.hex[:20]}"
        builder_created = False
        primary_error: ImageBuildError | ImageBuildTimeout | None = None

        try:
            if time.monotonic() >= deadline:
                raise ImageBuildTimeout(
                    f"Image build exceeded its configured timeout of {request.timeout_seconds} seconds.",
                    run_id=request.run_id,
                )
            builder_command = self._create_builder_command(builder_name, request, policy)
            self._run(
                builder_command,
                request=request,
                deadline=deadline,
            )
            builder_created = True
            self._verify_builder_profile(builder_name, request, policy)
            with tempfile.TemporaryDirectory(prefix="mlplatform-build-") as context_name:
                context_directory = Path(context_name)
                self._prepare_context(request, context_directory)
                image_id, build_logs = self._run_build(
                    request,
                    builder_name,
                    context_directory,
                    deadline,
                    policy,
                )
            versions = self._read_dependency_versions(image_id, request)
            return ImageBuildResult(
                image_id=image_id,
                dependency_versions=versions,
                logs=(f"verified build egress policy: {policy.policy_id}", *build_logs),
            )
        except (ImageBuildError, ImageBuildTimeout) as exc:
            primary_error = exc
            raise
        finally:
            cleanup_errors: list[ImageBuildError] = []
            if builder_created:
                try:
                    self._remove_builder(builder_name)
                except ImageBuildError as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            try:
                policy.close()
            except ImageBuildError as cleanup_error:
                cleanup_errors.append(cleanup_error)
            if primary_error is not None:
                for cleanup_error in cleanup_errors:
                    primary_error.logs = (
                        *primary_error.logs,
                        str(cleanup_error),
                        *cleanup_error.logs,
                    )
                    primary_error.add_note(
                        f"Temporary build resource cleanup also failed: {cleanup_error}"
                    )
            elif cleanup_errors:
                raise cleanup_errors[0]

    def _create_builder_command(
        self,
        builder_name: str,
        request: ImageBuildRequest,
        policy: DockerBuildEgressPolicy,
    ) -> list[str]:
        _parse_memory_limit(request.memory_limit)
        memory_limit = request.memory_limit
        cpu_period = 100_000
        cpu_quota = max(1, round(request.cpus * cpu_period))
        return [
            "docker",
            "buildx",
            "create",
            "--name",
            builder_name,
            "--driver",
            "docker-container",
            "--driver-opt",
            f"image={_ROOTLESS_BUILDKIT_IMAGE}",
            "--driver-opt",
            f"memory={memory_limit}",
            "--driver-opt",
            f"cpu-period={cpu_period}",
            "--driver-opt",
            f"cpu-quota={cpu_quota}",
            "--driver-opt",
            f"network={policy.internal_network_name}",
            "--driver-opt",
            f"env.HTTP_PROXY={policy.proxy_url}",
            "--driver-opt",
            f"env.HTTPS_PROXY={policy.proxy_url}",
            "--driver-opt",
            f"env.http_proxy={policy.proxy_url}",
            "--driver-opt",
            f"env.https_proxy={policy.proxy_url}",
            "--bootstrap",
        ]

    def _verify_builder_profile(
        self,
        builder_name: str,
        request: ImageBuildRequest,
        policy: DockerBuildEgressPolicy,
    ) -> None:
        try:
            container = self._docker().containers.get(f"buildx_buildkit_{builder_name}0")
            container.reload()
            attributes = container.attrs
            host_config = attributes["HostConfig"]
            config = attributes["Config"]
            mounts = attributes.get("Mounts", [])
        except (DockerException, KeyError, TypeError) as exc:
            raise ImageBuildError(
                "BuildKit resource and mount settings could not be verified.",
                run_id=request.run_id,
            ) from exc

        if not isinstance(host_config, dict) or not isinstance(config, dict):
            raise ImageBuildError(
                "BuildKit resource and user settings could not be verified.",
                run_id=request.run_id,
            )
        expected_memory = _parse_memory_limit(request.memory_limit)
        expected_cpu_period = 100_000
        expected_cpu_quota = max(1, round(request.cpus * expected_cpu_period))
        binds = host_config.get("Binds") or []
        if not _is_rootless_buildkit_wrapper(config):
            raise ImageBuildError(
                "BuildKit is not running the required rootless worker profile.",
                run_id=request.run_id,
            )
        if (
            host_config.get("Memory") != expected_memory
            or host_config.get("CpuPeriod") != expected_cpu_period
            or host_config.get("CpuQuota") != expected_cpu_quota
            or host_config.get("NetworkMode") != policy.internal_network_name
        ):
            raise ImageBuildError(
                "BuildKit did not apply the configured network and resource limits.",
                run_id=request.run_id,
            )
        proxy_environment = config.get("Env") or []
        expected_proxy = {
            f"HTTP_PROXY={policy.proxy_url}",
            f"HTTPS_PROXY={policy.proxy_url}",
            f"http_proxy={policy.proxy_url}",
            f"https_proxy={policy.proxy_url}",
        }
        if not expected_proxy.issubset(set(proxy_environment)):
            raise ImageBuildError(
                "BuildKit is not configured to use the verified egress proxy.",
                run_id=request.run_id,
            )
        configured_user = config.get("User")
        if not isinstance(configured_user, str) or configured_user.split(":", 1)[0] in {
            "",
            "0",
            "root",
        }:
            raise ImageBuildError(
                "BuildKit is not running as a non-root user.",
                run_id=request.run_id,
            )
        if binds or any(mount.get("Type") == "bind" for mount in mounts if isinstance(mount, dict)):
            raise ImageBuildError(
                "BuildKit unexpectedly mounted a host directory.",
                run_id=request.run_id,
            )

    def _prepare_context(
        self,
        request: ImageBuildRequest,
        context_directory: Path,
    ) -> None:
        try:
            project_root = request.project_directory.resolve(strict=True)
            requirements_path = project_root / request.requirements_file
            if (
                requirements_path.is_symlink()
                or not requirements_path.is_file()
                or requirements_path.parent != project_root
            ):
                raise ImageBuildError(
                    "requirements.txt must be a regular file at the project root.",
                    run_id=request.run_id,
                )
            (context_directory / "requirements.txt").write_bytes(requirements_path.read_bytes())
            base_image = _base_image(request.base_image_registry, request.python_version)
            dockerfile = generate_image_build_dockerfile(base_image=base_image)
            (context_directory / "install_dependencies.py").write_text(
                _INSTALLER,
                encoding="utf-8",
            )
            (context_directory / "Dockerfile").write_text(dockerfile, encoding="utf-8")
            (context_directory / ".dockerignore").write_text(
                "*\n!Dockerfile\n!requirements.txt\n!install_dependencies.py\n",
                encoding="utf-8",
            )
        except ImageBuildError:
            raise
        except OSError as exc:
            raise ImageBuildError(
                "The temporary training image build context could not be prepared.",
                run_id=request.run_id,
            ) from exc
        except ValueError as exc:
            raise ImageBuildError(
                "The configured Python base image version is invalid.",
                run_id=request.run_id,
            ) from exc

    def _run_build(
        self,
        request: ImageBuildRequest,
        builder_name: str,
        context_directory: Path,
        deadline: float,
        policy: DockerBuildEgressPolicy,
    ) -> tuple[str, tuple[str, ...]]:
        image_tag = f"mlplatform-training:{request.run_id.hex}"
        iid_file = context_directory / "image-id"
        command = [
            "docker",
            "buildx",
            "build",
            "--builder",
            builder_name,
            "--load",
            "--network",
            "default",
            "--build-arg",
            f"HTTP_PROXY={policy.proxy_url}",
            "--build-arg",
            f"HTTPS_PROXY={policy.proxy_url}",
            "--build-arg",
            f"http_proxy={policy.proxy_url}",
            "--build-arg",
            f"https_proxy={policy.proxy_url}",
            "--tag",
            image_tag,
            "--iidfile",
            str(iid_file),
            "--file",
            str(context_directory / "Dockerfile"),
            str(context_directory),
        ]
        if any(argument == "--allow" or argument.startswith("--allow=") for argument in command):
            raise ImageBuildError(
                "Image builds must not request insecure BuildKit entitlements.",
                run_id=request.run_id,
            )
        result = self._run(command, request=request, deadline=deadline)
        try:
            image_id = iid_file.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise ImageBuildError(
                "BuildKit did not provide an immutable image identity.",
                run_id=request.run_id,
                logs=_process_logs(result),
            ) from exc
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ImageBuildError(
                "BuildKit returned an invalid immutable image identity.",
                run_id=request.run_id,
                logs=_process_logs(result),
            )
        return image_id, _process_logs(result)

    def _read_dependency_versions(
        self,
        image_id: str,
        request: ImageBuildRequest,
    ) -> tuple[tuple[str, str], ...]:
        container = None
        try:
            container = self._docker().containers.create(
                image=image_id,
                command=["python", "-c", "pass"],
                network_mode="none",
                user="65532:65532",
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
            )
            archive, _ = container.get_archive("/opt/mlplatform/dependencies.freeze")
            content = _read_single_archive_file(archive)
        except (DockerException, OSError, tarfile.TarError, ValueError) as exc:
            raise ImageBuildError(
                "Installed dependency versions could not be collected from the image.",
                run_id=request.run_id,
            ) from exc
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except DockerException as exc:
                    raise ImageBuildError(
                        "Temporary dependency-inspection container could not be removed.",
                        run_id=request.run_id,
                    ) from exc

        try:
            return _parse_dependency_versions(content.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise ImageBuildError(
                "Installed dependency versions are malformed.",
                run_id=request.run_id,
            ) from exc

    def _remove_builder(self, builder_name: str) -> None:
        try:
            result = self._command_runner(
                ["docker", "buildx", "rm", "--force", builder_name],
                capture_output=True,
                check=False,
                text=True,
                timeout=_BUILD_TIMEOUT_BUFFER_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ImageBuildError("Temporary BuildKit builder could not be removed.") from exc
        if result.returncode != 0:
            raise ImageBuildError(
                "Temporary BuildKit builder could not be removed.",
                logs=_process_logs(result),
            )

    def _run(
        self,
        command: Sequence[str],
        *,
        request: ImageBuildRequest,
        deadline: float,
    ) -> subprocess.CompletedProcess[str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ImageBuildTimeout(
                f"Image build exceeded its configured timeout of {request.timeout_seconds} seconds.",
                run_id=request.run_id,
            )
        try:
            result = self._command_runner(
                list(command),
                capture_output=True,
                check=False,
                text=True,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired as exc:
            logs = _exception_logs(exc)
            raise ImageBuildTimeout(
                f"Image build exceeded its configured timeout of {request.timeout_seconds} seconds.",
                run_id=request.run_id,
                logs=logs,
            ) from exc
        except OSError as exc:
            raise ImageBuildError(
                "Docker Buildx is unavailable or could not be executed.",
                run_id=request.run_id,
                logs=(str(exc),),
            ) from exc
        if result.returncode != 0:
            raise ImageBuildError(
                f"Docker Buildx command failed with status {result.returncode}.",
                run_id=request.run_id,
                logs=_process_logs(result),
            )
        return result

    def _docker(self) -> Any:
        if self._docker_client is None:
            try:
                self._docker_client = docker.from_env()
            except DockerException as exc:
                raise ImageBuildError("The Docker daemon is unavailable.") from exc
        return self._docker_client


def generate_image_build_dockerfile(*, base_image: str) -> str:
    if not base_image or any(character.isspace() for character in base_image):
        raise ValueError("base_image must be a non-empty image reference.")
    return (
        f"FROM {base_image}\n"
        "USER root\n"
        "RUN mkdir -p /opt/mlplatform /home/mlplatform "
        "&& chown -R 65532:65532 /opt/mlplatform /home/mlplatform\n"
        "COPY --chown=65532:65532 requirements.txt /tmp/requirements.txt\n"
        "COPY --chown=65532:65532 install_dependencies.py /usr/local/bin/install_dependencies.py\n"
        "USER 65532:65532\n"
        "ENV HOME=/home/mlplatform PYTHONUSERBASE=/home/mlplatform/.local "
        "PATH=/home/mlplatform/.local/bin:$PATH\n"
        f"{_DOCKERFILE_SECURITY_CHECK}"
        "RUN python /usr/local/bin/install_dependencies.py\n"
        "RUN python -m pip freeze --all > /opt/mlplatform/dependencies.freeze\n"
        "WORKDIR /workspace\n"
    )


def _is_rootless_buildkit_wrapper(config: dict[str, Any]) -> bool:
    """Validate the Buildx wrapper that Docker Desktop necessarily creates.

    Docker Desktop starts the rootless BuildKit image through ``rootlesskit`` in a
    privileged *wrapper* container. That privilege only creates the user
    namespace; the BuildKit daemon and build workers use the configured non-root
    user. The Dockerfile-owned installer independently asserts non-root identity,
    ``NoNewPrivs=1``, and zero effective/permitted/inheritable capabilities.
    """

    configured_user = config.get("User")
    entrypoint = config.get("Entrypoint")
    image = config.get("Image")
    return (
        image == _ROOTLESS_BUILDKIT_IMAGE
        and isinstance(configured_user, str)
        and configured_user.split(":", 1)[0] not in {"", "0", "root"}
        and isinstance(entrypoint, list)
        and entrypoint[:1] == ["rootlesskit"]
    )


def _base_image(registry: str, python_version: str) -> str:
    if re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", python_version) is None:
        raise ValueError("python_version must be a Python major.minor[.patch] version.")
    return f"{registry}/library/python:{python_version}-slim"


def _parse_memory_limit(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([mg])", value.lower()) if isinstance(value, str) else None
    if match is None:
        raise ValueError("memory_limit must use M or G units.")
    amount = int(match.group(1))
    multiplier = 1024**2 if match.group(2) == "m" else 1024**3
    if amount <= 0 or amount * multiplier > 1024**3:
        raise ValueError("memory_limit must be greater than 0 and at most 1g.")
    return amount * multiplier


def _read_single_archive_file(chunks: Sequence[bytes]) -> bytes:
    archive = tarfile.open(fileobj=io.BytesIO(b"".join(chunks)), mode="r:*")
    with archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        if len(files) != 1:
            raise ValueError("Expected one regular file in the Docker archive.")
        file_object = archive.extractfile(files[0])
        if file_object is None:
            raise ValueError("Docker archive member could not be read.")
        return file_object.read()


def _parse_dependency_versions(content: str) -> tuple[tuple[str, str], ...]:
    versions: dict[str, str] = {}
    for line in content.splitlines():
        if not line:
            continue
        match = _PACKAGE_VERSION_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError("pip freeze output contains a non-pinned dependency.")
        name, version = match.groups()
        normalized_name = re.sub(r"[-_.]+", "-", name).lower()
        if normalized_name in versions:
            raise ValueError("pip freeze output contains a duplicate dependency.")
        versions[normalized_name] = version
    if not versions:
        raise ValueError("pip freeze output is empty.")
    return tuple(sorted(versions.items()))


def _process_logs(result: subprocess.CompletedProcess[str]) -> tuple[str, ...]:
    content = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return tuple(_redact_log_line(line) for line in content.splitlines() if line)[:1000]


def _exception_logs(error: subprocess.TimeoutExpired) -> tuple[str, ...]:
    parts = []
    for value in (error.stdout, error.stderr):
        if isinstance(value, bytes):
            parts.append(value.decode("utf-8", errors="replace"))
        elif isinstance(value, str):
            parts.append(value)
    return tuple(_redact_log_line(line) for part in parts for line in part.splitlines() if line)[
        :1000
    ]


def _redact_log_line(line: str) -> str:
    redacted = _URL_CREDENTIAL_PATTERN.sub(r"\1<redacted>@", line)
    redacted = _SECRET_PATTERN.sub(r"\1\2<redacted>", redacted)
    return redacted[:_MAX_BUILD_LOG_LINE_LENGTH]
