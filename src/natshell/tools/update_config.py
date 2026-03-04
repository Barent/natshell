"""Update a NatShell configuration value at runtime and persist to disk."""

from __future__ import annotations

import logging

from natshell.config import (
    CONFIG_ENUMS,
    VALID_CONFIG_KEYS,
    NatShellConfig,
    save_config_value,
)
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# ── Live config injection ─────────────────────────────────────────────────

_live_config: NatShellConfig | None = None


def set_live_config(config: NatShellConfig) -> None:
    """Inject the live NatShellConfig so updates take effect immediately."""
    global _live_config
    _live_config = config


# ── Tool definition ──────────────────────────────────────────────────────

DEFINITION = ToolDefinition(
    name="update_config",
    description=(
        "Update a NatShell configuration value. Changes are saved to "
        "~/.config/natshell/config.toml and take effect immediately. "
        "Use this when the user asks to change settings like temperature, "
        "max_steps, safety mode, GPU layers, etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "description": (
                    "The config section (e.g. 'agent', 'model', 'safety', 'engine', "
                    "'ui', 'backup', 'remote', 'ollama', 'mcp')"
                ),
            },
            "key": {
                "type": "string",
                "description": (
                    "The config key within the section "
                    "(e.g. 'temperature', 'n_gpu_layers')"
                ),
            },
            "value": {
                "type": "string",
                "description": "The new value (will be coerced to the correct type)",
            },
        },
        "required": ["section", "key", "value"],
    },
    requires_confirmation=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _coerce_value(value_str: str, type_str: str) -> int | float | bool | str:
    """Coerce a string value to the expected type.

    Raises ValueError on type mismatch.
    """
    if type_str == "int":
        try:
            return int(value_str)
        except ValueError:
            raise ValueError(f"Expected an integer, got: {value_str!r}")
    elif type_str == "float":
        try:
            return float(value_str)
        except ValueError:
            raise ValueError(f"Expected a float, got: {value_str!r}")
    elif type_str == "bool":
        low = value_str.lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"Expected a boolean, got: {value_str!r}")
    else:
        return value_str


def _apply_to_live_config(
    config: NatShellConfig, section: str, key: str, value: int | float | bool | str
) -> None:
    """Apply a value to the live config object."""
    section_obj = getattr(config, section, None)
    if section_obj is not None and hasattr(section_obj, key):
        setattr(section_obj, key, value)


# ── Handler ──────────────────────────────────────────────────────────────


async def update_config(section: str, key: str, value: str) -> ToolResult:
    """Validate, coerce, persist, and apply a config change."""
    # Validate section
    if section not in VALID_CONFIG_KEYS:
        valid_sections = ", ".join(sorted(VALID_CONFIG_KEYS.keys()))
        return ToolResult(
            error=f"Unknown config section: {section!r}. Valid sections: {valid_sections}",
            exit_code=1,
        )

    # Validate key
    section_keys = VALID_CONFIG_KEYS[section]
    if key not in section_keys:
        valid_keys = ", ".join(sorted(section_keys.keys()))
        return ToolResult(
            error=f"Unknown key {key!r} in [{section}]. Valid keys: {valid_keys}",
            exit_code=1,
        )

    # Coerce value
    type_str = section_keys[key]
    try:
        coerced = _coerce_value(value, type_str)
    except ValueError as e:
        return ToolResult(error=str(e), exit_code=1)

    # Enum validation
    if section in CONFIG_ENUMS and key in CONFIG_ENUMS[section]:
        allowed = CONFIG_ENUMS[section][key]
        if coerced not in allowed:
            return ToolResult(
                error=f"Invalid value {coerced!r} for [{section}].{key}. "
                f"Allowed values: {', '.join(allowed)}",
                exit_code=1,
            )

    # Persist to disk
    try:
        config_path = save_config_value(section, key, coerced)
    except Exception as e:
        return ToolResult(
            error=f"Failed to save config: {e}",
            exit_code=1,
        )

    # Apply to live config
    if _live_config is not None:
        _apply_to_live_config(_live_config, section, key, coerced)

    return ToolResult(
        output=f"Updated [{section}].{key} = {coerced!r} (saved to {config_path})"
    )
