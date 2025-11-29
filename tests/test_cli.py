"""CLI smoke tests."""

from typer.testing import CliRunner

from llm_lab.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Command line utilities for LLM Lab" in result.stdout
