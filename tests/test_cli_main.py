from __future__ import annotations

from typer.testing import CliRunner

from mixprep.cli.main import app

runner = CliRunner()


class TestCliWiring:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # no_args_is_help=True exits with code 2 but still prints help
        assert "models" in result.output.lower()

    def test_models_help_shows_subcommands(self):
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0
        assert "download" in result.output.lower()

    def test_models_download_unknown_exits_1(self):
        result = runner.invoke(app, ["models", "download", "--model", "nonexistent-xyz"])
        assert result.exit_code == 1

    def test_models_status_runs(self):
        # Just checks it doesn't crash — actual output tested in test_commands_models
        result = runner.invoke(app, ["models", "status"])
        assert result.exit_code == 0

    def test_models_clear_no_args_exits_1(self):
        result = runner.invoke(app, ["models", "clear"])
        assert result.exit_code == 1

    def test_models_clear_unknown_exits_1(self):
        result = runner.invoke(app, ["models", "clear", "nonexistent-xyz"])
        assert result.exit_code == 1
