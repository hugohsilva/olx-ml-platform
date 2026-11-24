from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import typer

CommandRegistrar = Callable[[typer.Typer], None]


def create_app(
    command_registrars: Iterable[CommandRegistrar] = (),
) -> typer.Typer:
    application = typer.Typer(
        name="mlplatform",
        help="Manage local ML training executions and model serving.",
        no_args_is_help=True,
    )
    application.callback()(_root_callback)
    for register_commands in command_registrars:
        register_commands(application)
    return application


def _root_callback() -> None:
    return None


def register_command(
    application: typer.Typer,
    command: Callable[..., Any],
    *,
    name: str,
    help: str | None = None,
) -> None:
    application.command(name=name, help=help)(command)


def _default_command_registrars() -> tuple[CommandRegistrar, ...]:
    from cli.commands.executions import register_commands as register_execution_commands
    from cli.commands.serve import register_commands as register_serve_commands
    from cli.commands.train import register_commands as register_training_commands

    return (register_training_commands, register_execution_commands, register_serve_commands)


app = create_app(_default_command_registrars())


def main() -> None:
    app()
