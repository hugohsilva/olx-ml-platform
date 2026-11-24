from __future__ import annotations

from typer.testing import CliRunner

from cli.commands.train import register_commands
from cli.main import create_app


def test_train_command_exposes_the_expected_resource_options() -> None:
    result = CliRunner().invoke(
        create_app([register_commands]),
        ["train", "--help"],
        terminal_width=140,
    )

    assert result.exit_code == 0
    assert "--timeout-seconds" in result.output
    assert "--cpus" in result.output
    assert "--memory-limit" in result.output


def test_train_command_rejects_timeout_above_the_configured_maximum() -> None:
    result = CliRunner().invoke(
        create_app([register_commands]),
        ["train", "training-project", "--timeout-seconds", "1801"],
    )

    assert result.exit_code != 0
    assert "1800" in result.output


def test_train_command_accepts_build_timeout_option_and_enforces_maximum() -> None:
    result = CliRunner().invoke(
        create_app([register_commands]),
        ["train", "training-project", "--build-timeout-seconds", "1801"],
    )

    assert result.exit_code != 0
    assert "1800" in result.output
    assert "No such option" not in result.output


def test_train_command_rejects_invalid_cpu_and_memory_before_opening_storage() -> None:
    runner = CliRunner()
    app = create_app([register_commands])

    cpu_result = runner.invoke(app, ["train", "training-project", "--cpus", "0"])
    memory_result = runner.invoke(
        app,
        ["train", "training-project", "--memory-limit", "2g"],
    )

    assert cpu_result.exit_code != 0
    assert "--cpus" in cpu_result.output
    assert memory_result.exit_code != 0
    assert "1g" in memory_result.output
