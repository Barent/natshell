"""Tests for model download helpers in model_manager.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from natshell.model_manager import (
    BUNDLED_TIERS,
    download_bundled_model,
    format_download_menu,
)


class TestBundledTiers:
    def test_has_three_tiers(self):
        assert len(BUNDLED_TIERS) == 3

    def test_tier_keys(self):
        assert set(BUNDLED_TIERS.keys()) == {"light", "standard", "enhanced"}

    @pytest.mark.parametrize("tier", ["light", "standard", "enhanced"])
    def test_tier_has_required_fields(self, tier: str):
        t = BUNDLED_TIERS[tier]
        assert "name" in t
        assert "description" in t
        assert "hf_repo" in t
        assert "hf_file" in t
        assert t["hf_repo"]  # non-empty
        assert t["hf_file"].endswith(".gguf")


class TestFormatDownloadMenu:
    def test_shows_all_tiers(self, tmp_path: Path):
        text = format_download_menu(tmp_path)
        assert "light" in text
        assert "standard" in text
        assert "enhanced" in text

    def test_marks_downloaded(self, tmp_path: Path):
        # Create a fake downloaded model
        tier = BUNDLED_TIERS["light"]
        (tmp_path / tier["hf_file"]).touch()
        text = format_download_menu(tmp_path)
        assert "downloaded" in text

    def test_no_marks_when_empty(self, tmp_path: Path):
        text = format_download_menu(tmp_path)
        assert "downloaded" not in text

    def test_shows_usage_hint(self, tmp_path: Path):
        text = format_download_menu(tmp_path)
        assert "/model download" in text


class TestDownloadBundledModel:
    async def test_invalid_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown tier"):
            await download_bundled_model("nonexistent")

    async def test_download_calls_hf_hub(self, tmp_path: Path):
        expected_path = str(tmp_path / "Qwen3-4B-Q4_K_M.gguf")

        with patch(
            "natshell.model_manager.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=expected_path,
        ):
            result = await download_bundled_model("light", models_dir=tmp_path)
            assert result == Path(expected_path)

    async def test_download_failure_raises_runtime_error(self, tmp_path: Path):
        with patch(
            "natshell.model_manager.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=Exception("network error"),
        ):
            with pytest.raises(RuntimeError, match="Download failed"):
                await download_bundled_model("light", models_dir=tmp_path)

    async def test_progress_callback(self, tmp_path: Path):
        messages = []
        expected_path = str(tmp_path / "Qwen3-4B-Q4_K_M.gguf")

        with patch(
            "natshell.model_manager.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=expected_path,
        ):
            await download_bundled_model(
                "light",
                models_dir=tmp_path,
                progress_callback=messages.append,
            )
        assert len(messages) == 1
        assert "Light" in messages[0]

    async def test_creates_models_dir(self, tmp_path: Path):
        models_dir = tmp_path / "sub" / "models"
        expected_path = str(models_dir / "Qwen3-4B-Q4_K_M.gguf")

        with patch(
            "natshell.model_manager.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=expected_path,
        ):
            await download_bundled_model("light", models_dir=models_dir)
        assert models_dir.is_dir()


class TestSetupWizardTierSharing:
    """Verify that setup_wizard.py uses the shared BUNDLED_TIERS."""

    def test_wizard_tiers_reference_bundled(self):
        from natshell.setup_wizard import MODEL_TIERS

        # Numbered tiers 1-3 should be the same objects as BUNDLED_TIERS
        assert MODEL_TIERS["1"] is BUNDLED_TIERS["light"]
        assert MODEL_TIERS["2"] is BUNDLED_TIERS["standard"]
        assert MODEL_TIERS["3"] is BUNDLED_TIERS["enhanced"]
