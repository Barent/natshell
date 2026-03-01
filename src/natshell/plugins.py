"""Plugin discovery and loading via entry points."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from natshell.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "natshell.plugins"


def load_plugins(registry: ToolRegistry) -> int:
    """Discover and load plugins from the ``natshell.plugins`` entry-point group.

    Each entry point must resolve to a callable with the signature
    ``(registry: ToolRegistry) -> None``.  Errors are logged as warnings
    and never crash startup.

    Returns the number of plugins that loaded successfully.
    """
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
