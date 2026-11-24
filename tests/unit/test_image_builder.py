from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path
from time import monotonic
from uuid import uuid4

import pytest

from application.errors import ImageBuildError, ImageBuildTimeout
from infrastructure.config import PlatformConfig
from infrastructure.docker.image_builder import (
    DockerImageBuilder,
    _parse_dependency_versions,
    _redact_log_line,
    generate_image_build_dockerfile,
)


def test_generated_image_build_runs_installer_non_root_and_records_versions() -> None:
    dockerfile = generate_image_build_dockerfile(base_image="docker.io/library/python:3.11-slim")

    assert "FROM docker.io/library/python:3.11-slim" in dockerfile
    assert "USER 65532:65532" in dockerfile
    assert "RUN python /usr/local/bin/install_dependencies.py" in dockerfile
    assert "python -m pip freeze --all" in dockerfile
    assert "CapEff" in dockerfile
    assert "/proc/self/status" in dockerfile


def test_installed_dependencies_are_normalized_and_sorted() -> None:
    assert _parse_dependency_versions("z-pkg==2.0\na_pkg==1.0\n") == (
        ("a-pkg", "1.0"),
        ("z-pkg", "2.0"),
    )


@pytest.mark.parametrize("output", ["", "editable @ file:///tmp/pkg", "pkg==1\nPkg==2"])
def test_installed_dependency_output_rejects_unpinned_or_duplicate_lines(output: str) -> None:
    with pytest.raises(ValueError):
        _parse_dependency_versions(output)


def test_build_logs_redact_url_credentials_and_secret_values() -> None:
    assert (
        _redact_log_line("https://alice:password@packages.example/token password=hunter2")
        == "https://<redacted>@packages.example/token password=<redacted>"
    )


def test_build_refuses_to_run_without_verifiable_egress_policy(tmp_path: Path) -> None:
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path)})
    invoked_commands: list[list[str]] = []

    class RefusedPolicy:
        def start(self) -> RefusedPolicy:
            raise ImageBuildError("egress policy could not be verified")

    builder = DockerImageBuilder(
        config,
        docker_client=object(),
        command_runner=lambda command, **_: invoked_commands.append(command),
        egress_policy_factory=lambda *_: RefusedPolicy(),
    )
    request = _build_request(tmp_path, config)

    with pytest.raises(ImageBuildError, match="egress policy could not be verified"):
        builder.build(request)

    assert invoked_commands == []


def test_image_build_context_contains_only_generated_files_and_requirements(tmp_path: Path) -> None:
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path)})
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "requirements.txt").write_text("scikit-learn==1.8.0\n", encoding="utf-8")
    (project / "train.py").write_text("untrusted training code", encoding="utf-8")
    (project / "data").mkdir()
    request = _build_request(tmp_path, config)
    builder = DockerImageBuilder(
        config,
        egress_policy_factory=lambda *_: _policy(),
    )
    context = tmp_path / "context"
    context.mkdir()

    builder._prepare_context(request, context)

    assert {path.name for path in context.iterdir()} == {
        ".dockerignore",
        "Dockerfile",
        "install_dependencies.py",
        "requirements.txt",
    }
    assert (context / "requirements.txt").read_text(encoding="utf-8") == "scikit-learn==1.8.0\n"


def test_builder_profile_fails_closed_when_resource_limits_are_missing(tmp_path: Path) -> None:
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path)})
    request = _build_request(tmp_path, config)
    container = FakeBuilderContainer()
    client = FakeDockerClient(container)
    builder = DockerImageBuilder(config, docker_client=client)

    with pytest.raises(ImageBuildError, match="network and resource limits"):
        builder._verify_builder_profile("test-builder", request, _policy())


def test_builder_profile_fails_closed_when_rootless_worker_profile_is_missing(
    tmp_path: Path,
) -> None:
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path)})
    request = _build_request(tmp_path, config)
    container = FakeBuilderContainer()
    container.attrs["Config"]["User"] = "root"
    builder = DockerImageBuilder(config, docker_client=FakeDockerClient(container))

    with pytest.raises(ImageBuildError, match="rootless worker profile"):
        builder._verify_builder_profile("test-builder", request, _policy())


def test_builder_command_sets_rootless_image_and_effective_resource_limits(tmp_path: Path) -> None:
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path)})
    request = _build_request(tmp_path, config)
    builder = DockerImageBuilder(config)

    command = builder._create_builder_command("test-builder", request, _policy())

    assert "image=moby/buildkit:rootless" in command
    assert "--allow" not in command
    assert "memory=1g" in command
    assert "cpu-period=100000" in command
    assert "cpu-quota=100000" in command
    assert "network=mlplatform-build-network" in command
    assert "env.HTTPS_PROXY=http://172.20.0.2:3128" in command


def test_builder_profile_accepts_verified_limits_and_non_root_user(tmp_path: Path) -> None:
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path)})
    request = _build_request(tmp_path, config)
    container = FakeBuilderContainer()
    container.attrs["HostConfig"].update(
        {
            "Memory": 1024**3,
            "CpuPeriod": 100_000,
            "CpuQuota": 100_000,
            "NetworkMode": "mlplatform-build-network",
        }
    )
    container.attrs["Config"]["Env"] = [
        "HTTP_PROXY=http://172.20.0.2:3128",
        "HTTPS_PROXY=http://172.20.0.2:3128",
        "http_proxy=http://172.20.0.2:3128",
        "https_proxy=http://172.20.0.2:3128",
    ]
    builder = DockerImageBuilder(config, docker_client=FakeDockerClient(container))

    builder._verify_builder_profile("test-builder", request, _policy())


def test_image_build_captures_immutable_image_id_and_build_logs(tmp_path: Path) -> None:
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path)})
    request = _build_request(tmp_path, config)
    context = tmp_path / "context"
    context.mkdir()
    image_id = "sha256:" + "b" * 64

    def command_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        iid_file = Path(command[command.index("--iidfile") + 1])
        iid_file.write_text(image_id, encoding="ascii")
        return subprocess.CompletedProcess(command, 0, stdout="build complete\n", stderr="")

    builder = DockerImageBuilder(config, command_runner=command_runner)
    actual_image_id, logs = builder._run_build(
        request,
        "test-builder",
        context,
        monotonic() + 30,
        _policy(),
    )

    assert actual_image_id == image_id
    assert logs == ("build complete",)


def test_image_builder_reads_actual_installed_versions_from_image(tmp_path: Path) -> None:
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path)})
    request = _build_request(tmp_path, config)
    archive_buffer = io.BytesIO()
    version_file = b"pip==25.0\nscikit-learn==1.8.0\n"
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        member = tarfile.TarInfo("dependencies.freeze")
        member.size = len(version_file)
        archive.addfile(member, io.BytesIO(version_file))
    container = FakeArchiveContainer(archive_buffer.getvalue())
    builder = DockerImageBuilder(
        config,
        docker_client=FakeArchiveDockerClient(container),
    )

    versions = builder._read_dependency_versions("sha256:" + "b" * 64, request)

    assert versions == (("pip", "25.0"), ("scikit-learn", "1.8.0"))
    assert container.options["network_mode"] == "none"
    assert container.options["user"] == "65532:65532"
    assert container.removed


def test_image_builder_classifies_build_timeout_and_preserves_partial_logs(
    tmp_path: Path,
) -> None:
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path)})
    request = _build_request(tmp_path, config)

    def timeout_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            command,
            kwargs["timeout"],
            output="partial build output",
            stderr="",
        )

    builder = DockerImageBuilder(config, command_runner=timeout_runner)

    with pytest.raises(ImageBuildTimeout) as error:
        builder._run(
            ["docker", "buildx", "build"],
            request=request,
            deadline=monotonic() + 30,
        )

    assert error.value.run_id == request.run_id
    assert error.value.logs == ("partial build output",)


def test_image_builder_classifies_build_oom_as_image_build_error(tmp_path: Path) -> None:
    config = PlatformConfig.from_env({"MLPLATFORM_HOME": str(tmp_path)})
    request = _build_request(tmp_path, config)

    def oom_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 137, stdout="out of memory", stderr="")

    builder = DockerImageBuilder(config, command_runner=oom_runner)

    with pytest.raises(ImageBuildError, match="status 137") as error:
        builder._run(
            ["docker", "buildx", "build"],
            request=request,
            deadline=monotonic() + 30,
        )

    assert error.value.run_id == request.run_id
    assert error.value.logs == ("out of memory",)


class FakeBuilderContainer:
    def __init__(self) -> None:
        self.attrs = {
            "HostConfig": {
                "Privileged": False,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges:true"],
                "Memory": 0,
                "CpuPeriod": 0,
                "CpuQuota": 0,
                "Binds": [],
                "NetworkMode": "",
            },
            "Config": {
                "User": "1000:1000",
                "Image": "moby/buildkit:rootless",
                "Entrypoint": ["rootlesskit", "buildkitd"],
                "Env": [],
            },
            "Mounts": [],
        }

    def reload(self) -> None:
        pass


class FakeContainers:
    def __init__(self, container: FakeBuilderContainer) -> None:
        self.container = container

    def get(self, _: str) -> FakeBuilderContainer:
        return self.container


class FakeDockerClient:
    def __init__(self, container: FakeBuilderContainer) -> None:
        self.containers = FakeContainers(container)


class FakeArchiveContainer:
    def __init__(self, archive: bytes) -> None:
        self.archive = archive
        self.options: dict[str, object] = {}
        self.removed = False

    def get_archive(self, _: str) -> tuple[list[bytes], dict[str, object]]:
        return [self.archive], {}

    def remove(self, *, force: bool) -> None:
        self.removed = force


class FakeArchiveContainers:
    def __init__(self, container: FakeArchiveContainer) -> None:
        self.container = container

    def create(self, **options: object) -> FakeArchiveContainer:
        self.container.options = options
        return self.container


class FakeArchiveDockerClient:
    def __init__(self, container: FakeArchiveContainer) -> None:
        self.containers = FakeArchiveContainers(container)


class FakePolicy:
    internal_network_name = "mlplatform-build-network"
    proxy_url = "http://172.20.0.2:3128"
    policy_id = "docker-internal-proxy:mlplatform-build-network"

    def start(self) -> FakePolicy:
        return self

    def close(self) -> None:
        pass


def _policy() -> FakePolicy:
    return FakePolicy()


def _build_request(tmp_path: Path, config: PlatformConfig):
    from application.ports import ImageBuildRequest

    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "requirements.txt").write_text("scikit-learn==1.8.0\n", encoding="utf-8")
    return ImageBuildRequest(
        run_id=uuid4(),
        project_directory=project,
        requirements_file="requirements.txt",
        python_version="3.11.9",
        base_image_registry=config.base_image_registry,
        allowed_repositories=config.build_allowed_repositories,
        cpus=1.0,
        memory_limit="1g",
        timeout_seconds=600,
    )
