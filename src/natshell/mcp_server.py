"""MCP (Model Context Protocol) server for NatShell.

Exposes NatShell's tools, system context, and help topics as an MCP server
that can be used by any MCP-compatible client (e.g., Claude Desktop).

Requires the optional ``mcp`` package: ``pip install 'natshell[mcp]'``
"""

from __future__ import annotations

import logging
from typing import Any

from natshell.config import McpConfig
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# All mcp imports are deferred / wrapped so the module can be imported
# even when the ``mcp`` package is not installed.  The public entry points
# (``create_mcp_server``, ``run_mcp_server``) raise ``ImportError`` at
# call time if the package is missing.


def _json_schema_to_mcp_input(parameters: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI-style JSON-Schema parameters dict to the format
    expected by ``mcp.types.Tool.inputSchema``."""
    # MCP expects a top-level JSON Schema object.  Our tool definitions
    # already store one (type: object, properties, required) so we can
    # pass it through directly.
    return parameters


def _build_tool_list(registry: ToolRegistry) -> list:
    """Build a list of ``mcp.types.Tool`` from the NatShell registry."""
    from mcp.types import Tool

    tools: list[Tool] = []
    for schema in registry.get_tool_schemas():
        func = schema["function"]
        tools.append(
            Tool(
                name=func["name"],
                description=func.get("description", ""),
                inputSchema=_json_schema_to_mcp_input(func.get("parameters", {})),
            )
        )
    return tools


async def _execute_tool(
    registry: ToolRegistry,
    safety: SafetyClassifier,
    mcp_config: McpConfig,
    name: str,
    arguments: dict[str, Any],
) -> list:
    """Execute a NatShell tool with safety classification, returning MCP content."""
    from mcp.types import TextContent

    # Safety check
    risk = safety.classify_tool_call(name, arguments)

    if risk == Risk.BLOCKED:
        raise ValueError(
            f"Tool call '{name}' is blocked by safety policy."
        )

    if risk == Risk.CONFIRM:
        if mcp_config.safety_mode != "permissive":
            raise ValueError(
                f"Tool call '{name}' requires confirmation (safety_mode='strict'). "
                f"Set safety_mode='permissive' in [mcp] config to auto-approve."
            )
        logger.info("MCP: auto-approving confirm-level tool call %s (permissive mode)", name)

    result = await registry.execute(name, arguments)
    content_text = result.to_message_content()

    return [TextContent(type="text", text=content_text)]


def _build_resource_list() -> list:
    """Build the list of MCP resources (system context + help topics)."""
    from mcp.types import Resource

    resources = []

    # System context resource
    resources.append(
        Resource(
            uri="natshell://system-context",
            name="System Context",
            description="Current system information (CPU, RAM, disk, network, tools)",
            mimeType="text/plain",
        )
    )

    # Help topic resources
    from natshell.tools.natshell_help import VALID_TOPICS

    for topic in VALID_TOPICS:
        resources.append(
            Resource(
                uri=f"natshell://help/{topic}",
                name=f"Help: {topic}",
                description=f"NatShell documentation for '{topic}'",
                mimeType="text/plain",
            )
        )

    return resources


async def _read_resource(uri: str) -> str:
    """Read the content of a NatShell MCP resource by URI."""
    uri_str = str(uri)

    if uri_str == "natshell://system-context":
        from natshell.agent.context import gather_system_context

        ctx = await gather_system_context()
        return ctx.to_prompt_text()

    if uri_str.startswith("natshell://help/"):
        topic = uri_str[len("natshell://help/"):]
        from natshell.tools.natshell_help import natshell_help

        result = await natshell_help(topic)
        if result.error:
            return f"Error: {result.error}"
        return result.output

    raise ValueError(f"Unknown resource URI: {uri_str}")


def create_mcp_server(
    tools_registry: ToolRegistry,
    safety_classifier: SafetyClassifier,
    mcp_config: McpConfig | None = None,
) -> Any:
    """Create and configure an MCP server exposing NatShell tools and resources.

    Parameters
    ----------
    tools_registry:
        The NatShell tool registry (with all tools registered).
    safety_classifier:
        The safety classifier for vetting tool calls.
    mcp_config:
        MCP-specific configuration.  Defaults to strict safety mode.

    Returns
    -------
    mcp.server.Server
        The configured (but not yet running) MCP server instance.

    Raises
    ------
    ImportError
        If the ``mcp`` package is not installed.
    """
    from mcp.server import Server

    if mcp_config is None:
        mcp_config = McpConfig()

    server = Server("natshell")

    # ── Tool listing ──────────────────────────────────────────────────────
    @server.list_tools()
    async def handle_list_tools() -> list:
        return _build_tool_list(tools_registry)

    # ── Tool execution ────────────────────────────────────────────────────
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None = None) -> list:
        if arguments is None:
            arguments = {}
        return await _execute_tool(
            tools_registry, safety_classifier, mcp_config, name, arguments
        )

    # ── Resource listing ──────────────────────────────────────────────────
    @server.list_resources()
    async def handle_list_resources() -> list:
        return _build_resource_list()

    # ── Resource reading ──────────────────────────────────────────────────
    @server.read_resource()
    async def handle_read_resource(uri) -> str:
        return await _read_resource(uri)

    return server


async def run_mcp_server(
    tools_registry: ToolRegistry,
    safety_classifier: SafetyClassifier,
    mcp_config: McpConfig | None = None,
) -> None:
    """Create the MCP server and run it over stdio.

    This is the main entry point for ``natshell --mcp``.

    Raises
    ------
    ImportError
        If the ``mcp`` package is not installed.
    """
    from mcp.server.stdio import stdio_server

    server = create_mcp_server(tools_registry, safety_classifier, mcp_config)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
