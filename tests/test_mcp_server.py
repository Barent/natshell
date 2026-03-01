"""Tests for the MCP server module.

These tests are structured to work without the ``mcp`` package installed.
We mock the mcp imports and test the logic that NatShell layers on top:
tool-to-MCP mapping, safety classification integration, resource listing,
and safety_mode behavior.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from natshell.config import McpConfig, SafetyConfig
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.tools.registry import ToolDefinition, ToolRegistry, ToolResult

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_mock_mcp_modules():
    """Create a minimal mock of the ``mcp`` package tree so the module can
    be imported without the real package being installed."""
    mcp = types.ModuleType("mcp")
    mcp_types = types.ModuleType("mcp.types")
    mcp_server = types.ModuleType("mcp.server")
    mcp_server_stdio = types.ModuleType("mcp.server.stdio")

    # mcp.types stubs
    class Tool:
        def __init__(self, name="", description="", inputSchema=None):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema or {}

    class Resource:
        def __init__(self, uri="", name="", description="", mimeType="text/plain"):
            self.uri = uri
            self.name = name
            self.description = description
            self.mimeType = mimeType

    class TextContent:
        def __init__(self, type="text", text=""):
            self.type = type
            self.text = text

    mcp_types.Tool = Tool
    mcp_types.Resource = Resource
    mcp_types.TextContent = TextContent

    # mcp.server.Server stub
    class FakeServer:
        def __init__(self, name):
            self.name = name
            self._handlers = {}

        def list_tools(self):
            def decorator(fn):
                self._handlers["list_tools"] = fn
                return fn
            return decorator

        def call_tool(self):
            def decorator(fn):
                self._handlers["call_tool"] = fn
                return fn
            return decorator

        def list_resources(self):
            def decorator(fn):
                self._handlers["list_resources"] = fn
                return fn
            return decorator

        def read_resource(self):
            def decorator(fn):
                self._handlers["read_resource"] = fn
                return fn
            return decorator

        def create_initialization_options(self):
            return {}

        async def run(self, read_stream, write_stream, init_options):
            pass

    mcp_server.Server = FakeServer
    mcp_server_stdio.stdio_server = AsyncMock()

    mcp.types = mcp_types
    mcp.server = mcp_server
    mcp.server.stdio = mcp_server_stdio

    return {
        "mcp": mcp,
        "mcp.types": mcp_types,
        "mcp.server": mcp_server,
        "mcp.server.stdio": mcp_server_stdio,
    }


@pytest.fixture(autouse=True)
def _mock_mcp(monkeypatch):
    """Inject mock mcp modules into sys.modules for every test, then clean up."""
    mocks = _make_mock_mcp_modules()
    original = {}
    for name, mod in mocks.items():
        original[name] = sys.modules.get(name)
        monkeypatch.setitem(sys.modules, name, mod)

    # Force-reload the mcp_server module so it picks up the mocks
    import importlib

    import natshell.mcp_server
    importlib.reload(natshell.mcp_server)

    yield mocks

    # importlib.reload would restore real state next time if needed


def _make_registry_with_tool() -> ToolRegistry:
    """Create a ToolRegistry with a single test tool."""
    registry = ToolRegistry()

    async def _echo_handler(message: str = "") -> ToolResult:
        return ToolResult(output=f"echo: {message}")

    defn = ToolDefinition(
        name="echo_test",
        description="Echo a message back",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to echo"},
            },
            "required": ["message"],
        },
    )
    registry.register(defn, _echo_handler)
    return registry


def _make_safety(mode: str = "confirm") -> SafetyClassifier:
    """Create a SafetyClassifier with default patterns."""
    return SafetyClassifier(SafetyConfig(mode=mode))


# ── Tests ──────────────────────────────────────────────────────────────────


class TestBuildToolList:
    """Test that NatShell tools are correctly mapped to MCP Tool objects."""

    def test_single_tool(self):
        from natshell.mcp_server import _build_tool_list

        registry = _make_registry_with_tool()
        tools = _build_tool_list(registry)

        assert len(tools) == 1
        assert tools[0].name == "echo_test"
        assert tools[0].description == "Echo a message back"
        assert "properties" in tools[0].inputSchema

    def test_multiple_tools(self):
        from natshell.mcp_server import _build_tool_list

        registry = _make_registry_with_tool()

        # Add a second tool
        async def _noop_handler() -> ToolResult:
            return ToolResult(output="ok")

        registry.register(
            ToolDefinition(
                name="noop",
                description="Do nothing",
                parameters={"type": "object", "properties": {}},
            ),
            _noop_handler,
        )

        tools = _build_tool_list(registry)
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"echo_test", "noop"}

    def test_default_registry_tools(self):
        """Ensure all default NatShell tools are exposed."""
        from natshell.mcp_server import _build_tool_list
        from natshell.tools.registry import create_default_registry

        registry = create_default_registry()
        tools = _build_tool_list(registry)

        names = {t.name for t in tools}
        assert "execute_shell" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names


class TestExecuteTool:
    """Test tool execution with safety classification."""

    async def test_safe_tool_executes(self):
        from natshell.mcp_server import _execute_tool

        registry = _make_registry_with_tool()
        safety = _make_safety()
        mcp_config = McpConfig()

        result = await _execute_tool(
            registry, safety, mcp_config, "echo_test", {"message": "hello"}
        )
        assert len(result) == 1
        assert "echo: hello" in result[0].text

    async def test_blocked_tool_raises(self):
        from natshell.mcp_server import _execute_tool

        registry = _make_registry_with_tool()
        safety = _make_safety()
        mcp_config = McpConfig()

        # Patch the classifier to return BLOCKED
        safety.classify_tool_call = MagicMock(return_value=Risk.BLOCKED)

        with pytest.raises(ValueError, match="blocked by safety policy"):
            await _execute_tool(
                registry, safety, mcp_config, "echo_test", {"message": "bad"}
            )

    async def test_confirm_strict_raises(self):
        """In strict mode, CONFIRM-level calls should raise."""
        from natshell.mcp_server import _execute_tool

        registry = _make_registry_with_tool()
        safety = _make_safety()
        mcp_config = McpConfig(safety_mode="strict")

        safety.classify_tool_call = MagicMock(return_value=Risk.CONFIRM)

        with pytest.raises(ValueError, match="requires confirmation"):
            await _execute_tool(
                registry, safety, mcp_config, "echo_test", {"message": "risky"}
            )

    async def test_confirm_permissive_executes(self):
        """In permissive mode, CONFIRM-level calls should auto-approve."""
        from natshell.mcp_server import _execute_tool

        registry = _make_registry_with_tool()
        safety = _make_safety()
        mcp_config = McpConfig(safety_mode="permissive")

        safety.classify_tool_call = MagicMock(return_value=Risk.CONFIRM)

        result = await _execute_tool(
            registry, safety, mcp_config, "echo_test", {"message": "ok"}
        )
        assert len(result) == 1
        assert "echo: ok" in result[0].text

    async def test_blocked_even_in_permissive(self):
        """BLOCKED is never auto-approved, even in permissive mode."""
        from natshell.mcp_server import _execute_tool

        registry = _make_registry_with_tool()
        safety = _make_safety()
        mcp_config = McpConfig(safety_mode="permissive")

        safety.classify_tool_call = MagicMock(return_value=Risk.BLOCKED)

        with pytest.raises(ValueError, match="blocked"):
            await _execute_tool(
                registry, safety, mcp_config, "echo_test", {"message": "bad"}
            )


class TestResourceListing:
    """Test MCP resource enumeration."""

    def test_lists_system_context(self):
        from natshell.mcp_server import _build_resource_list

        resources = _build_resource_list()
        uris = [str(r.uri) for r in resources]
        assert "natshell://system-context" in uris

    def test_lists_help_topics(self):
        from natshell.mcp_server import _build_resource_list
        from natshell.tools.natshell_help import VALID_TOPICS

        resources = _build_resource_list()
        uris = [str(r.uri) for r in resources]

        for topic in VALID_TOPICS:
            assert f"natshell://help/{topic}" in uris

    def test_resource_count(self):
        """1 system-context + N help topics."""
        from natshell.mcp_server import _build_resource_list
        from natshell.tools.natshell_help import VALID_TOPICS

        resources = _build_resource_list()
        assert len(resources) == 1 + len(VALID_TOPICS)


class TestReadResource:
    """Test reading MCP resource content."""

    async def test_read_help_topic(self):
        from natshell.mcp_server import _read_resource

        content = await _read_resource("natshell://help/overview")
        assert "NatShell" in content
        assert "natural language" in content

    async def test_read_unknown_uri_raises(self):
        from natshell.mcp_server import _read_resource

        with pytest.raises(ValueError, match="Unknown resource URI"):
            await _read_resource("natshell://nonexistent")


class TestCreateMcpServer:
    """Test that create_mcp_server produces a configured server."""

    def test_creates_server(self):
        from natshell.mcp_server import create_mcp_server

        registry = _make_registry_with_tool()
        safety = _make_safety()

        server = create_mcp_server(registry, safety)
        assert server.name == "natshell"
        # Should have all four handlers registered
        assert "list_tools" in server._handlers
        assert "call_tool" in server._handlers
        assert "list_resources" in server._handlers
        assert "read_resource" in server._handlers

    def test_default_mcp_config(self):
        """When no McpConfig is passed, defaults to strict."""
        from natshell.mcp_server import create_mcp_server

        registry = _make_registry_with_tool()
        safety = _make_safety()

        # Should not raise
        server = create_mcp_server(registry, safety)
        assert server is not None


class TestMcpConfig:
    """Test McpConfig integration with config loading."""

    def test_default_safety_mode(self):
        config = McpConfig()
        assert config.safety_mode == "strict"

    def test_permissive_mode(self):
        config = McpConfig(safety_mode="permissive")
        assert config.safety_mode == "permissive"

    def test_config_in_natshell_config(self):
        from natshell.config import NatShellConfig

        cfg = NatShellConfig()
        assert cfg.mcp.safety_mode == "strict"
