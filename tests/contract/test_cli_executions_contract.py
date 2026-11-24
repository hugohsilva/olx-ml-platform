from __future__ import annotations

from typer.testing import CliRunner

from cli.commands.executions import register_commands
from cli.main import create_app


def test_execution_commands_are_exposed_by_the_cli_contract() -> None:
    app = create_app([register_commands])

    result = CliRunner().invoke(app, ["executions", "--help"], terminal_width=140)

    assert result.exit_code == 0
    assert "list" in result.output
    assert "show" in result.output
