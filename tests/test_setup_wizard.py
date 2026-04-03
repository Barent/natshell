"""Tests for the first-run setup wizard."""

from __future__ import annotations

import io
import os
import tomllib
from pathlib import Path
from unittest.mock import patch

from natshell.setup_wizard import (
    MODEL_TIERS,
    _detect_gpu_info,
    _write_initial_config,
    run_setup_wizard,
    should_run_wizard,
)

# ── TestModelTiers ───────────────────────────────────────────────────────


class TestModelTiers:
    def test_all_tiers_have_required_keys(self):
        for tier_id, tier in MODEL_TIERS.items():
            assert "name" in tier, f"Tier {tier_id} missing 'name'"
            assert "description" in tier
            assert "hf_repo" in tier
            assert "hf_file" in tier

    def test_five_tiers(self):
        assert len(MODEL_TIERS) == 5

    def test_tier_1_is_light(self):
        assert MODEL_TIERS["1"]["name"] == "Light"
        assert "4B" in MODEL_TIERS["1"]["hf_file"]

    def test_tier_2_is_standard(self):
        assert MODEL_TIERS["2"]["name"] == "Standard"
        assert "8B" in MODEL_TIERS["2"]["hf_file"]

    def test_tier_3_is_enhanced(self):
        assert MODEL_TIERS["3"]["name"] == "Enhanced"
        assert "Mistral" in MODEL_TIERS["3"]["hf_file"]

    def test_tier_4_is_remote(self):
        assert MODEL_TIERS["4"]["name"] == "Remote only"
        assert MODEL_TIERS["4"]["hf_repo"] == ""

    def test_tier_5_is_skip(self):
        assert MODEL_TIERS["5"]["name"] == "Skip"
        assert MODEL_TIERS["5"]["hf_repo"] == ""


# ── TestWriteInitialConfig ───────────────────────────────────────────────


class TestWriteInitialConfig:
    def test_creates_file(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        _write_initial_config(
            config_path, "Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"
        )
        assert config_path.exists()

    def test_valid_toml(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        _write_initial_config(
            config_path, "Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"
        )
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        assert data["model"]["hf_repo"] == "Qwen/Qwen3-8B-GGUF"
        assert data["model"]["hf_file"] == "Qwen3-8B-Q4_K_M.gguf"

    def test_sections_present(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        _write_initial_config(config_path, "test/repo", "test.gguf")
        content = config_path.read_text()
        assert "[model]" in content

    def test_permissions(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        _write_initial_config(config_path, "test/repo", "test.gguf")
        mode = os.stat(config_path).st_mode & 0o777
        assert mode == 0o600

    def test_creates_parent_dirs(self, tmp_path: Path):
        config_path = tmp_path / "sub" / "dir" / "config.toml"
        _write_initial_config(config_path, "test/repo", "test.gguf")
        assert config_path.exists()

    def test_empty_repo_writes_minimal(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        _write_initial_config(config_path, "", "")
        content = config_path.read_text()
        assert "[model]" not in content

    def test_n_gpu_layers_default(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        _write_initial_config(config_path, "test/repo", "test.gguf")
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        assert data["model"]["n_gpu_layers"] == -1


# ── TestRunSetupWizard ───────────────────────────────────────────────────


class TestRunSetupWizard:
    def test_default_choice_is_standard(self, tmp_path: Path):
        """Pressing Enter with no input selects tier 2 (Standard/8B)."""
        output = io.StringIO()
        with (
            patch(
                "natshell.platform.config_dir",
                return_value=tmp_path / ".config" / "natshell",
            ),
            patch(
                "natshell.setup_wizard._check_llama_available",
                return_value=True,
            ),
        ):
            result = run_setup_wizard(
                output=output, input_fn=lambda _: ""
            )
        assert result is not None
        config_path = (
            tmp_path / ".config" / "natshell" / "config.toml"
        )
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        assert "8B" in data["model"]["hf_file"]

    def test_choice_1_light(self, tmp_path: Path):
        output = io.StringIO()
        with (
            patch(
                "natshell.platform.config_dir",
                return_value=tmp_path / ".config" / "natshell",
            ),
            patch(
                "natshell.setup_wizard._check_llama_available",
                return_value=True,
            ),
        ):
            result = run_setup_wizard(
                output=output, input_fn=lambda _: "1"
            )
        assert result is not None
        config_path = (
            tmp_path / ".config" / "natshell" / "config.toml"
        )
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        assert "4B" in data["model"]["hf_file"]

    def test_choice_2_standard(self, tmp_path: Path):
        output = io.StringIO()
        with (
            patch(
                "natshell.platform.config_dir",
                return_value=tmp_path / ".config" / "natshell",
            ),
            patch(
                "natshell.setup_wizard._check_llama_available",
                return_value=True,
            ),
        ):
            result = run_setup_wizard(
                output=output, input_fn=lambda _: "2"
            )
        assert result is not None
        config_path = (
            tmp_path / ".config" / "natshell" / "config.toml"
        )
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        assert "8B" in data["model"]["hf_file"]

    def test_choice_3_enhanced(self, tmp_path: Path):
        output = io.StringIO()
        with (
            patch(
                "natshell.platform.config_dir",
                return_value=tmp_path / ".config" / "natshell",
            ),
            patch(
                "natshell.setup_wizard._check_llama_available",
                return_value=True,
            ),
        ):
            result = run_setup_wizard(
                output=output, input_fn=lambda _: "3"
            )
        assert result is not None
        config_path = (
            tmp_path / ".config" / "natshell" / "config.toml"
        )
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        assert "Mistral" in data["model"]["hf_file"]

    def test_choice_4_remote(self, tmp_path: Path):
        output = io.StringIO()
        with (
            patch(
                "natshell.platform.config_dir",
                return_value=tmp_path / ".config" / "natshell",
            ),
            patch(
                "natshell.setup_wizard._check_llama_available",
                return_value=True,
            ),
        ):
            result = run_setup_wizard(
                output=output, input_fn=lambda _: "4"
            )
        # Choice 4 now writes an Ollama config and returns the path
        assert result is not None
        config_path = tmp_path / ".config" / "natshell" / "config.toml"
        assert config_path.exists()
        content = config_path.read_text()
        assert "ollama" in content.lower()
        assert "localhost:11434" in content

    def test_choice_5_skip(self, tmp_path: Path):
        output = io.StringIO()
        with (
            patch(
                "natshell.platform.config_dir",
                return_value=tmp_path / ".config" / "natshell",
            ),
            patch(
                "natshell.setup_wizard._check_llama_available",
                return_value=True,
            ),
        ):
            result = run_setup_wizard(
                output=output, input_fn=lambda _: "5"
            )
        assert result is None

    def test_invalid_choice_defaults_to_standard(self, tmp_path: Path):
        output = io.StringIO()
        with (
            patch(
                "natshell.platform.config_dir",
                return_value=tmp_path / ".config" / "natshell",
            ),
            patch(
                "natshell.setup_wizard._check_llama_available",
                return_value=True,
            ),
        ):
            result = run_setup_wizard(
                output=output, input_fn=lambda _: "9"
            )
        assert result is not None
        config_path = (
            tmp_path / ".config" / "natshell" / "config.toml"
        )
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        assert "8B" in data["model"]["hf_file"]

    def test_eof_returns_none(self, tmp_path: Path):
        output = io.StringIO()

        def raise_eof(_):
            raise EOFError

        with (
            patch(
                "natshell.platform.config_dir",
                return_value=tmp_path / ".config" / "natshell",
            ),
            patch(
                "natshell.setup_wizard._check_llama_available",
                return_value=True,
            ),
        ):
            result = run_setup_wizard(
                output=output, input_fn=raise_eof
            )
        assert result is None

    def test_keyboard_interrupt_returns_none(self, tmp_path: Path):
        output = io.StringIO()

        def raise_ki(_):
            raise KeyboardInterrupt

        with (
            patch(
                "natshell.platform.config_dir",
                return_value=tmp_path / ".config" / "natshell",
            ),
            patch(
                "natshell.setup_wizard._check_llama_available",
                return_value=True,
            ),
        ):
            result = run_setup_wizard(
                output=output, input_fn=raise_ki
            )
        assert result is None

    def test_gpu_detection_output(self, tmp_path: Path):
        output = io.StringIO()
        mock_gpu_info = ("NVIDIA RTX 4090 (24576 MB VRAM)", True)
        with (
            patch(
                "natshell.platform.config_dir",
                return_value=tmp_path / ".config" / "natshell",
            ),
            patch(
                "natshell.setup_wizard._check_llama_available",
                return_value=True,
            ),
            patch(
                "natshell.setup_wizard._detect_gpu_info",
                return_value=mock_gpu_info,
            ),
        ):
            run_setup_wizard(output=output, input_fn=lambda _: "5")
        text = output.getvalue()
        assert "NVIDIA RTX 4090" in text

    def test_gpu_warning_no_backend(self, tmp_path: Path):
        output = io.StringIO()
        mock_gpu_info = ("AMD Radeon RX 7900", False)
        with (
            patch(
                "natshell.platform.config_dir",
                return_value=tmp_path / ".config" / "natshell",
            ),
            patch(
                "natshell.setup_wizard._check_llama_available",
                return_value=True,
            ),
            patch(
                "natshell.setup_wizard._detect_gpu_info",
                return_value=mock_gpu_info,
            ),
        ):
            run_setup_wizard(output=output, input_fn=lambda _: "5")
        text = output.getvalue()
        assert "without GPU support" in text


# ── TestDetectGpuInfo ────────────────────────────────────────────────────


class TestDetectGpuInfo:
    def _patch_gpu(self, gpus, backend):
        """Patch gpu functions, clearing lru_cache first."""
        from natshell import gpu as gpu_mod

        gpu_mod.detect_gpus.cache_clear()
        return (
            patch.object(
                gpu_mod, "detect_gpus", return_value=gpus
            ),
            patch.object(
                gpu_mod, "gpu_backend_available",
                return_value=backend,
            ),
        )

    def test_no_gpus(self):
        p1, p2 = self._patch_gpu([], False)
        with p1, p2:
            desc, backend = _detect_gpu_info()
        assert desc == ""
        assert backend is False

    def test_with_gpu(self):
        from natshell.gpu import GpuInfo

        gpu = GpuInfo(
            name="Test GPU", vendor="nvidia",
            device_index=0, vram_mb=8192, is_discrete=True,
        )
        p1, p2 = self._patch_gpu([gpu], True)
        with p1, p2:
            desc, backend = _detect_gpu_info()
        assert "Test GPU" in desc
        assert "8192" in desc
        assert backend is True

    def test_exception_returns_defaults(self):
        """When gpu detection raises, returns empty defaults."""
        from natshell import gpu as gpu_mod

        gpu_mod.detect_gpus.cache_clear()
        with patch.object(
            gpu_mod, "detect_gpus",
            side_effect=RuntimeError("fail"),
        ):
            desc, backend = _detect_gpu_info()
        assert desc == ""
        assert backend is False


# ── TestWizardTrigger ────────────────────────────────────────────────────


class TestWizardTrigger:
    def test_triggers_when_no_config(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        assert should_run_wizard(config_path=config_path)

    def test_skips_when_config_exists(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        config_path.write_text("[model]\n")
        assert not should_run_wizard(config_path=config_path)

    def test_skips_with_no_setup(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        assert not should_run_wizard(
            config_path=config_path, no_setup=True
        )

    def test_skips_with_headless(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        assert not should_run_wizard(
            config_path=config_path, headless=True
        )

    def test_skips_with_mcp(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        assert not should_run_wizard(
            config_path=config_path, mcp=True
        )

    def test_skips_with_download(self, tmp_path: Path):
        config_path = tmp_path / "config.toml"
        assert not should_run_wizard(
            config_path=config_path, download=True
        )

    def test_default_path_used(self):
        """When no config_path given, uses default path."""
        fake_cfg = Path("/fake/config/natshell")
        with patch(
            "natshell.platform.config_dir",
            return_value=fake_cfg,
        ):
            result = should_run_wizard()
        assert result is True
