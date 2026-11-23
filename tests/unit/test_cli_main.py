from __future__ import annotations

import typer
from typer.testing import CliRunner

from cli.main import create_app, register_command


def test_typer_entrypoint_provides_help_without_owning_command_implementations() -> None:
    result = CliRunner().invoke(create_app(), ["--help"])

    assert result.exit_code == 0
    assert "Manage local ML training executions and model serving." in result.output


def test_commands_can_be_registered_from_separate_modules() -> None:
    def command_group(application: typer.Typer) -> None:
        def ping() -> None:
            typer.echo("pong")

        register_command(application, ping, name="ping")

    result = CliRunner().invoke(create_app([command_group]), ["ping"])

    assert result.exit_code == 0
    assert result.output.strip() == "pong"
