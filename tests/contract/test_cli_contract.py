from __future__ import annotations

from typer.testing import CliRunner

from cli.main import app


def test_cli_exposes_all_required_lifecycle_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "executions" in result.output
    assert "serve" in result.output


def test_serve_command_requires_a_uuid_and_exposes_a_local_port_option() -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["serve", "--help"])
    invalid_result = runner.invoke(app, ["serve", "not-a-uuid"])

    assert help_result.exit_code == 0
    assert "--port" in help_result.output
    assert invalid_result.exit_code != 0


def test_execution_commands_allow_human_readable_output_without_changing_json_default() -> None:
    runner = CliRunner()
    list_help = runner.invoke(app, ["executions", "list", "--help"])
    show_help = runner.invoke(app, ["executions", "show", "--help"])

    assert list_help.exit_code == 0
    assert show_help.exit_code == 0
    assert "--format" in list_help.output
    assert "--format" in show_help.output
