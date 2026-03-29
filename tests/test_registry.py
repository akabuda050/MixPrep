from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from mixprep.models.registry import ModelEntry, ModelRegistry

MINIMAL_MANIFEST = textwrap.dedent("""\
    models:
      test-model:
        url: "https://example.com/model.pb"
        framework: tensorflow
        description: "A test model"
      another-model:
        url: "https://example.com/another.json"
        framework: json
        description: "Another model"
""")


@pytest.fixture
def manifest_file(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(MINIMAL_MANIFEST)
    return p


class TestModelEntry:
    def test_fields(self):
        entry = ModelEntry(
            name="foo",
            url="https://example.com/foo.pb",
            framework="tensorflow",
            description="desc",
        )
        assert entry.name == "foo"
        assert entry.framework == "tensorflow"


class TestModelRegistry:
    def test_loads_all_models(self, manifest_file: Path):
        registry = ModelRegistry(manifest_file)
        assert len(registry.all()) == 2

    def test_get_known_model(self, manifest_file: Path):
        registry = ModelRegistry(manifest_file)
        entry = registry.get("test-model")
        assert entry is not None
        assert entry.url == "https://example.com/model.pb"
        assert entry.framework == "tensorflow"

    def test_get_unknown_model_returns_none(self, manifest_file: Path):
        registry = ModelRegistry(manifest_file)
        assert registry.get("does-not-exist") is None

    def test_names(self, manifest_file: Path):
        registry = ModelRegistry(manifest_file)
        assert set(registry.names()) == {"test-model", "another-model"}

    def test_missing_manifest_is_empty(self, tmp_path: Path):
        registry = ModelRegistry(tmp_path / "nonexistent.yaml")
        assert registry.all() == []
        assert registry.names() == []

    def test_empty_manifest_is_empty(self, tmp_path: Path):
        p = tmp_path / "manifest.yaml"
        p.write_text("models: {}\n")
        registry = ModelRegistry(p)
        assert registry.all() == []


class TestProductionManifest:
    """Sanity checks on the real manifest.yaml shipped with the package."""

    def test_loads_without_error(self):
        registry = ModelRegistry()
        assert len(registry.all()) > 0

    def test_all_entries_have_url(self):
        registry = ModelRegistry()
        for entry in registry.all():
            assert entry.url.startswith("https://"), f"{entry.name} has bad URL"

    def test_all_entries_have_description(self):
        registry = ModelRegistry()
        for entry in registry.all():
            assert entry.description, f"{entry.name} has empty description"

    def test_known_models_present(self):
        registry = ModelRegistry()
        expected = [
            "essentia-discogs-effnet",
            "essentia-maest",
            "essentia-genre-jamendo",
            "essentia-arousal-valence-deam",
        ]
        for name in expected:
            assert registry.get(name) is not None, f"Missing model: {name}"
