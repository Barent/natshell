"""Tests for the kiwix_search tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from natshell.tools.kiwix_search import (
    _strip_html,
    kiwix_search,
    reset_limits,
    set_kiwix_url,
    set_limits,
)
from natshell.tools.limits import ToolLimits
from natshell.tools.registry import PLAN_SAFE_TOOLS, create_default_registry

_PATCH_CLIENT = "natshell.tools.kiwix_search.httpx.AsyncClient"

_DEFAULT_KIWIX_URL = "http://localhost:8888"

_SEARCH_JSON = {
    "items": [
        {
            "title": "Albert Einstein",
            "url": "/A/Albert_Einstein",
            "snippet": "<b>Albert Einstein</b> was a theoretical physicist.",
        },
        {
            "title": "General relativity",
            "url": "/A/General_relativity",
            "snippet": "General relativity is Einstein's theory of gravity.",
        },
    ]
}

_EMPTY_SEARCH_JSON = {"items": []}


@pytest.fixture(autouse=True)
def _reset():
    set_kiwix_url(_DEFAULT_KIWIX_URL)
    yield
    reset_limits()
    set_kiwix_url(_DEFAULT_KIWIX_URL)


# ─── HTML stripping ───────────────────────────────────────────────────────────


class TestStripHtml:
    def test_strips_tags(self):
        assert _strip_html("<b>hello</b>") == "hello"

    def test_strips_script(self):
        result = _strip_html("<p>visible</p><script>evil()</script>")
        assert "evil" not in result
        assert "visible" in result

    def test_strips_style(self):
        result = _strip_html("<style>.cls { color: red }</style><p>text</p>")
        assert "color" not in result
        assert "text" in result

    def test_decodes_entities(self):
        result = _strip_html("<p>1 &amp; 2</p>")
        assert "&amp;" not in result
        assert "&" in result

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_plain_text_unchanged(self):
        assert _strip_html("hello world") == "hello world"


# ─── Search results ───────────────────────────────────────────────────────────


def _make_mock_client(json_data: dict, status_code: int = 200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = ""

    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    mock_client_cls = MagicMock()
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_client_cls, mock_client


class TestSearchResults:
    @patch(_PATCH_CLIENT)
    async def test_returns_titles(self, mock_client_cls):
        mock_client_cls_inst, _ = _make_mock_client(_SEARCH_JSON)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        result = await kiwix_search("Einstein")
        assert result.exit_code == 0
        assert "Albert Einstein" in result.output

    @patch(_PATCH_CLIENT)
    async def test_returns_snippets(self, mock_client_cls):
        mock_client_cls_inst, _ = _make_mock_client(_SEARCH_JSON)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        result = await kiwix_search("Einstein")
        assert "theoretical physicist" in result.output

    @patch(_PATCH_CLIENT)
    async def test_no_results_message(self, mock_client_cls):
        mock_client_cls_inst, _ = _make_mock_client(_EMPTY_SEARCH_JSON)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        result = await kiwix_search("xyzzy_no_match")
        assert result.exit_code == 0
        assert "No results" in result.output

    @patch(_PATCH_CLIENT)
    async def test_book_filter_in_url(self, mock_client_cls):
        """book parameter must appear in the search URL."""
        mock_client_cls_inst, mock_client = _make_mock_client(_SEARCH_JSON)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("Einstein", book="wikipedia_en_mini")
        call_args = mock_client.get.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        # If url is in positional or keyword args
        if not url and call_args:
            url = str(call_args)
        assert "wikipedia_en_mini" in url or "wikipedia_en_mini" in str(call_args)

    @patch(_PATCH_CLIENT)
    async def test_results_clamped_to_20(self, mock_client_cls):
        """results > 20 should be clamped."""
        mock_client_cls_inst, mock_client = _make_mock_client(_SEARCH_JSON)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("Einstein", results=999)
        call_args = mock_client.get.call_args
        assert "end=20" in str(call_args)

    @patch(_PATCH_CLIENT)
    async def test_uses_configured_url(self, mock_client_cls):
        """set_kiwix_url must be respected."""
        set_kiwix_url("http://myserver:9999")
        mock_client_cls_inst, mock_client = _make_mock_client(_SEARCH_JSON)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("Einstein")
        call_args = mock_client.get.call_args
        assert "myserver:9999" in str(call_args)


# ─── fetch_article ────────────────────────────────────────────────────────────


class TestFetchArticle:
    @patch(_PATCH_CLIENT)
    async def test_fetch_article_true(self, mock_client_cls):
        """When fetch_article=True, article text should appear in output."""
        article_html = "<html><body><p>Einstein was born in 1879.</p></body></html>"

        search_resp = MagicMock(spec=httpx.Response)
        search_resp.status_code = 200
        search_resp.json.return_value = _SEARCH_JSON
        search_resp.text = ""

        article_resp = MagicMock(spec=httpx.Response)
        article_resp.status_code = 200
        article_resp.json.return_value = {}
        article_resp.text = article_html

        mock_client = AsyncMock()
        mock_client.get.side_effect = [search_resp, article_resp]
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await kiwix_search("Einstein", fetch_article=True)
        assert result.exit_code == 0
        assert "1879" in result.output

    @patch(_PATCH_CLIENT)
    async def test_fetch_article_false_no_extra_request(self, mock_client_cls):
        mock_client_cls_inst, mock_client = _make_mock_client(_SEARCH_JSON)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("Einstein", fetch_article=False)
        assert mock_client.get.call_count == 1


# ─── Error handling ───────────────────────────────────────────────────────────


class TestErrorHandling:
    @patch(_PATCH_CLIENT)
    async def test_connection_refused(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await kiwix_search("Einstein")
        assert result.exit_code == 1
        assert "kiwix-serve" in result.error.lower() or "connect" in result.error.lower()
        assert "Is kiwix-serve running?" in result.error

    @patch(_PATCH_CLIENT)
    async def test_timeout(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await kiwix_search("Einstein")
        assert result.exit_code == 1
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    @patch(_PATCH_CLIENT)
    async def test_non_200_status(self, mock_client_cls):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 404
        resp.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.get.return_value = resp
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await kiwix_search("Einstein")
        assert result.exit_code == 1
        assert "404" in result.error

    @patch(_PATCH_CLIENT)
    async def test_non_json_response(self, mock_client_cls):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")

        mock_client = AsyncMock()
        mock_client.get.return_value = resp
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await kiwix_search("Einstein")
        assert result.exit_code == 1
        assert "json" in result.error.lower()


# ─── Output truncation ────────────────────────────────────────────────────────


class TestTruncation:
    @patch(_PATCH_CLIENT)
    async def test_output_truncated_at_limit(self, mock_client_cls):
        large_search = {
            "items": [
                {
                    "title": f"Article {i}",
                    "url": f"/A/Article_{i}",
                    "snippet": "x" * 200,
                }
                for i in range(20)
            ]
        }
        mock_client_cls_inst, _ = _make_mock_client(large_search)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        set_limits(ToolLimits(max_output_chars=100))
        result = await kiwix_search("test", results=20)
        assert result.truncated
        assert len(result.output) <= 100


# ─── Registration ─────────────────────────────────────────────────────────────


class TestRegistration:
    def test_registered_in_default_registry(self):
        registry = create_default_registry()
        assert "kiwix_search" in registry.tool_names

    def test_in_plan_safe_tools(self):
        assert "kiwix_search" in PLAN_SAFE_TOOLS

    def test_not_in_small_context_tools(self):
        from natshell.tools.registry import SMALL_CONTEXT_TOOLS

        assert "kiwix_search" not in SMALL_CONTEXT_TOOLS

    def test_schema_generated(self):
        registry = create_default_registry()
        schemas = registry.get_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "kiwix_search" in names

    def test_no_confirmation_required(self):
        registry = create_default_registry()
        defn = registry.get_definition("kiwix_search")
        assert defn is not None
        assert defn.requires_confirmation is False
