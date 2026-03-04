"""Plugin discovery and loading via entry points and file-based plugins."""

from __future__ import annotations

import importlib.util
import logging
from importlib.metadata import entry_points
from pathlib import Path

from natshell.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "natshell.plugins"
PLUGIN_DIR = Path("~/.config/natshell/plugins").expanduser()


def _load_entry_point_plugins(registry: ToolRegistry) -> int:
    """Load plugins registered via the ``natshell.plugins`` entry-point group."""
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:
        # Python 3.11 fallback: entry_points() may not accept `group`
        # keyword on older stdlib versions — filter manually.
        eps = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]

    loaded = 0
    for ep in eps:
        try:
            plugin_fn = ep.load()
            plugin_fn(registry)
            logger.info("Loaded plugin: %s", ep.name)
            loaded += 1
        except Exception:
            logger.warning("Failed to load plugin %r", ep.name, exc_info=True)

    return loaded


def _load_file_plugins(registry: ToolRegistry) -> int:
    """Load plugins from ``*.py`` files in the plugin directory."""
    if not PLUGIN_DIR.is_dir():
        return 0

    loaded = 0
    for path in sorted(PLUGIN_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue

        module_name = f"natshell_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                logger.warning("Cannot load plugin file %s: invalid module spec", path.name)
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            logger.warning("Failed to import plugin file %s", path.name, exc_info=True)
            continue

        register_fn = getattr(module, "register", None)
        if not callable(register_fn):
            logger.warning("Plugin file %s has no register() callable, skipping", path.name)
            continue

        try:
            register_fn(registry)
            logger.info("Loaded file plugin: %s", path.name)
            loaded += 1
        except Exception:
            logger.warning("Failed to load file plugin %s", path.name, exc_info=True)

    return loaded


def load_plugins(registry: ToolRegistry) -> int:
    """Discover and load plugins from entry points and the plugin directory.

    Each plugin must provide a callable with the signature
    ``(registry: ToolRegistry) -> None``.  Errors are logged as warnings
    and never crash startup.

    Returns the number of plugins that loaded successfully.
    """
    loaded = _load_entry_point_plugins(registry)
    loaded += _load_file_plugins(registry)
    return loaded
