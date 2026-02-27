"""Tool registration, dispatch, and schema generation for the agent."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    output: str = ""
    error: str = ""
    exit_code: int = 0
    truncated: bool = False

    def to_message_content(self) -> str:
        """Format for inclusion in the LLM conversation as a tool result."""
        parts = []
        if self.exit_code != 0:
            parts.append(f"Exit code: {self.exit_code}")
        if self.output:
            parts.append(f"{self.output}")
        if self.error:
            parts.append(f"stderr:\n{self.error}")
        if self.truncated:
            parts.append("[output was truncated]")
        return "\n".join(parts) if parts else "(no output)"


@dataclass
class ToolDefinition:
    """Schema definition for a tool, used to generate the LLM prompt."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object
    requires_confirmation: bool = False


# Type alias for tool handler functions
ToolHandler = Callable[..., Awaitable[ToolResult]]


class ToolRegistry:
    """Manages tool registration, schema generation, and dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        """Register a tool with its definition and handler."""
        self._tools[definition.name] = handler
        self._definitions[definition.name] = definition
        logger.debug(f"Registered tool: {definition.name}")

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Generate OpenAI-compatible tool schemas for the LLM."""
        schemas = []
        for defn in self._definitions.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": defn.name,
                    "description": defn.description,
                    "parameters": defn.parameters,
                },
            })
        return schemas

    def get_definition(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._definitions.get(name)

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments."""
        handler = self._tools.get(name)
        if handler is None:
            return ToolResult(
                output="",
                error=f"Unknown tool: {name}",
                exit_code=1,
            )

        try:
            return await handler(**arguments)
        except TypeError:
            # LLM may hallucinate wrong argument names (e.g. "param" instead
            # of "topic").  Attempt to remap values to the schema's expected
            # parameter names by position before giving up.
            remapped = self._remap_arguments(name, arguments)
            if remapped is not None:
                logger.warning(
                    "Tool %s: remapped bad arg names %s → %s",
                    name, list(arguments.keys()), list(remapped.keys()),
                )
                try:
                    return await handler(**remapped)
                except Exception as e:
                    logger.exception(f"Tool {name} raised an exception after remap")
                    return ToolResult(
                        output="",
                        error=f"Tool error: {type(e).__name__}: {e}",
                        exit_code=1,
                    )
            # Remap not possible — report the original TypeError
            return ToolResult(
                output="",
                error=f"Tool error: wrong arguments for {name}. "
                       f"Got: {list(arguments.keys())}",
                exit_code=1,
            )
        except Exception as e:
            logger.exception(f"Tool {name} raised an exception")
            return ToolResult(
                output="",
                error=f"Tool error: {type(e).__name__}: {e}",
                exit_code=1,
            )

    def _remap_arguments(
        self, name: str, arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Try to map LLM-provided argument values to the schema's expected
        parameter names by position.  Returns None if remapping isn't possible
        (e.g. argument count mismatch)."""
        defn = self._definitions.get(name)
        if defn is None:
            return None
        props = defn.parameters.get("properties", {})
        expected_keys = list(props.keys())
        provided_values = list(arguments.values())
        if len(provided_values) != len(expected_keys):
            return None
        return dict(zip(expected_keys, provided_values))

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


def create_default_registry() -> ToolRegistry:
    """Create a registry with all built-in tools registered."""
    from natshell.tools.execute_shell import execute_shell, DEFINITION as EXEC_DEF
    from natshell.tools.read_file import read_file, DEFINITION as READ_DEF
    from natshell.tools.write_file import write_file, DEFINITION as WRITE_DEF
    from natshell.tools.list_directory import list_directory, DEFINITION as LIST_DEF
    from natshell.tools.search_files import search_files, DEFINITION as SEARCH_DEF
    from natshell.tools.natshell_help import natshell_help, DEFINITION as HELP_DEF

    registry = ToolRegistry()
    registry.register(EXEC_DEF, execute_shell)
    registry.register(READ_DEF, read_file)
    registry.register(WRITE_DEF, write_file)
    registry.register(LIST_DEF, list_directory)
    registry.register(SEARCH_DEF, search_files)
    registry.register(HELP_DEF, natshell_help)
    return registry
