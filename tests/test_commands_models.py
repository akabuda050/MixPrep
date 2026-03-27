from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mixprep.models.registry import ModelEntry, ModelRegistry


def _make_registry(*names: str) -> ModelRegistry:
    registry = MagicMock(spec=ModelRegistry)
    entries = [
        ModelEntry(
            name=name,
            url=f"https://example.com/{name}.pb",
            framework="tensorflow",
            description=f"desc for {name}",
        )
        for name in names
    ]
    registry.all.return_value = entries
    registry.names.return_value = list(names)
    registry.get.side_effect = lambda n: next((e for e in entries if e.name == n), None)
    return registry


class TestRunStatus:
    def test_shows_present_and_missing(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("essentia-discogs-effnet")
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                with patch(
                    "mixprep.cli.commands.models.model_status",
                    return_value=[
                        {
                            "name": "present-model",
                            "filename": "present-model.pb",
                            "present": True,
                            "size_mb": 1.0,
                            "description": "d",
                        },
                        {
                            "name": "missing-model",
                            "filename": "missing-model.pb",
                            "present": False,
                            "size_mb": None,
                            "description": "d",
                        },
                    ],
                ):
                    from io import StringIO

                    from rich.console import Console

                    buf = StringIO()
                    with patch.object(cmd_module, "console", Console(file=buf, highlight=False)):
                        from mixprep.cli.commands.models import run_status

                        run_status()
                    output = buf.getvalue()
        assert "✗" in output  # missing file
        assert "✓" in output  # present file


class TestRunDownload:
    def test_downloads_single_model(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("essentia-discogs-effnet")
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                with patch(
                    "mixprep.cli.commands.models.download_model", return_value=tmp_path / "model.pb"
                ) as mock_dl:
                    from mixprep.cli.commands.models import run_download

                    run_download(model="essentia-discogs-effnet", force=False)

        mock_dl.assert_called_once()
        assert mock_dl.call_args[0][0].name == "essentia-discogs-effnet"

    def test_unknown_model_exits(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("essentia-discogs-effnet")
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                from mixprep.cli.commands.models import run_download

                with pytest.raises(SystemExit) as exc:
                    run_download(model="nonexistent", force=False)
        assert exc.value.code == 1

    def test_downloads_all_when_no_model(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("model-a", "model-b")
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                with patch(
                    "mixprep.cli.commands.models.download_all",
                    return_value=(
                        {"model-a": tmp_path / "a.pb", "model-b": tmp_path / "b.pb"},
                        {},
                    ),
                ) as mock_all:
                    from mixprep.cli.commands.models import run_download

                    run_download(model=None, force=False)

        mock_all.assert_called_once_with(registry, force=False)

    def test_partial_failure_exits_1_with_hint(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("model-a", "model-b")
        printed = []
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                with patch(
                    "mixprep.cli.commands.models.download_all",
                    return_value=(
                        {"model-b": tmp_path / "b.pb"},
                        {"model-a": RuntimeError("timeout")},
                    ),
                ):
                    with patch.object(cmd_module.console, "print", side_effect=printed.append):
                        from mixprep.cli.commands.models import run_download

                        with pytest.raises(SystemExit) as exc:
                            run_download(model=None, force=False)

        assert exc.value.code == 1
        summary = " ".join(str(m) for m in printed)
        assert "1" in summary  # 1 failed
        assert "mixprep models download" in summary  # retry hint

    def test_force_flag_passed_through(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("essentia-discogs-effnet")
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                with patch(
                    "mixprep.cli.commands.models.download_model", return_value=tmp_path / "model.pb"
                ) as mock_dl:
                    from mixprep.cli.commands.models import run_download

                    run_download(model="essentia-discogs-effnet", force=True)

        assert mock_dl.call_args[1]["force"] is True


class TestRunClear:
    def test_clears_single_existing_model(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("essentia-discogs-effnet")
        model_file = tmp_path / "essentia-discogs-effnet.pb"
        model_file.write_bytes(b"weights")

        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                from mixprep.cli.commands.models import run_clear

                run_clear(model="essentia-discogs-effnet", all=False)

        assert not model_file.exists()

    def test_clear_missing_model_no_error(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("essentia-discogs-effnet")
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                from mixprep.cli.commands.models import run_clear

                run_clear(model="essentia-discogs-effnet", all=False)
        # no file, no exception

    def test_clear_unknown_model_exits(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("essentia-discogs-effnet")
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                from mixprep.cli.commands.models import run_clear

                with pytest.raises(SystemExit) as exc:
                    run_clear(model="nonexistent", all=False)
        assert exc.value.code == 1

    def test_clear_no_model_no_all_exits(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("essentia-discogs-effnet")
        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                from mixprep.cli.commands.models import run_clear

                with pytest.raises(SystemExit) as exc:
                    run_clear(model=None, all=False)
        assert exc.value.code == 1

    def test_clear_all_removes_present_files(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("model-a", "model-b")
        (tmp_path / "model-a.pb").write_bytes(b"weights")

        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                from mixprep.cli.commands.models import run_clear

                run_clear(model=None, all=True)

        assert not (tmp_path / "model-a.pb").exists()

    def test_clear_all_counts_removed(self, tmp_path: Path):
        from mixprep.cli.commands import models as cmd_module

        registry = _make_registry("model-a", "model-b")
        (tmp_path / "model-a.pb").write_bytes(b"weights")
        (tmp_path / "model-b.pb").write_bytes(b"weights")

        output_lines = []

        with patch.dict(os.environ, {"MIXPREP_MODELS_DIR": str(tmp_path)}):
            with patch.object(cmd_module, "registry", registry):
                with patch.object(cmd_module.console, "print", side_effect=output_lines.append):
                    from mixprep.cli.commands.models import run_clear

                    run_clear(model=None, all=True)

        summary = next(line for line in output_lines if "Done." in line)
        assert "2" in summary
