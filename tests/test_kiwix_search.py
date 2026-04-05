"""Tests for the kiwix_search tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from natshell.tools.kiwix_search import (
    _discover_kiwix_url,
    _looks_like_kiwix_catalog,
    _parse_search_html,
    _strip_html,
    discover_and_set_kiwix_url,
    kiwix_search,
    reset_limits,
    set_kiwix_url,
    set_limits,
)
from natshell.tools.limits import ToolLimits
from natshell.tools.registry import PLAN_SAFE_TOOLS, create_default_registry

_PATCH_CLIENT = "natshell.tools.kiwix_search.httpx.AsyncClient"

_DEFAULT_KIWIX_URL = "http://localhost:8080"

_SEARCH_HTML = """\
<html><body>
<ul class="results">
  <li>
    <a href="/content/wikipedia_en/Albert_Einstein">Albert Einstein</a>
    <cite>Albert Einstein was a theoretical physicist.</cite>
  </li>
  <li>
    <a href="/content/wikipedia_en/General_relativity">General relativity</a>
    <cite>General relativity is Einstein's theory of gravity.</cite>
  </li>
</ul>
</body></html>
"""

_EMPTY_SEARCH_HTML = "<html><body><p>No results found.</p></body></html>"

_CATALOG_XML = """\
<?xml version="1.0"?>
<feed>
  <entry><name>wikipedia_en_all_maxi_2026-02</name><title>Wikipedia</title></entry>
  <entry><name>stackexchange_en</name><title>Stack Exchange</title></entry>
</feed>
"""


def _make_mock_client(html_text: str, status_code: int = 200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = html_text

    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    mock_client_cls = MagicMock()
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_client_cls, mock_client


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


# ─── HTML search result parser ────────────────────────────────────────────────


class TestParseSearchHtml:
    def test_extracts_titles(self):
        items = _parse_search_html(_SEARCH_HTML)
        titles = [i["title"] for i in items]
        assert "Albert Einstein" in titles

    def test_extracts_urls(self):
        items = _parse_search_html(_SEARCH_HTML)
        assert items[0]["url"] == "/content/wikipedia_en/Albert_Einstein"

    def test_extracts_snippets(self):
        items = _parse_search_html(_SEARCH_HTML)
        assert "theoretical physicist" in items[0]["snippet"]

    def test_empty_html_returns_empty_list(self):
        assert _parse_search_html(_EMPTY_SEARCH_HTML) == []

    def test_ignores_non_content_links(self):
        html_text = '<a href="/other/link">Not a result</a>'
        assert _parse_search_html(html_text) == []

    def test_strips_bold_from_snippet(self):
        html_text = (
            '<a href="/content/wiki/Foo">Foo</a>'
            "<cite>term with <b>highlighted</b> word</cite>"
        )
        items = _parse_search_html(html_text)
        assert items[0]["snippet"] == "term with highlighted word"


# ─── Search results ───────────────────────────────────────────────────────────


class TestSearchResults:
    @patch(_PATCH_CLIENT)
    async def test_returns_titles(self, mock_client_cls):
        mock_client_cls_inst, _ = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        result = await kiwix_search("Einstein")
        assert result.exit_code == 0
        assert "Albert Einstein" in result.output

    @patch(_PATCH_CLIENT)
    async def test_returns_snippets(self, mock_client_cls):
        mock_client_cls_inst, _ = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        result = await kiwix_search("Einstein")
        assert "theoretical physicist" in result.output

    @patch(_PATCH_CLIENT)
    async def test_no_results_message(self, mock_client_cls):
        mock_client_cls_inst, _ = _make_mock_client(_EMPTY_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        result = await kiwix_search("xyzzy_no_match")
        assert result.exit_code == 0
        assert "No results" in result.output

    @patch(_PATCH_CLIENT)
    async def test_book_filter_uses_content_param(self, mock_client_cls):
        """book parameter must appear as 'content=' in the search URL."""
        mock_client_cls_inst, mock_client = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("Einstein", book="wikipedia_en_mini")
        call_args = str(mock_client.get.call_args)
        assert "content=" in call_args
        assert "wikipedia_en_mini" in call_args

    @patch(_PATCH_CLIENT)
    async def test_search_url_uses_pattern_param(self, mock_client_cls):
        """Search URL must use 'pattern=' not 'query='."""
        mock_client_cls_inst, mock_client = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("Einstein")
        call_args = str(mock_client.get.call_args)
        assert "pattern=" in call_args
        assert "query=" not in call_args

    @patch(_PATCH_CLIENT)
    async def test_results_clamped_to_20(self, mock_client_cls):
        """results > 20 should be clamped."""
        mock_client_cls_inst, mock_client = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("Einstein", results=999)
        call_args = str(mock_client.get.call_args)
        assert "pageLength=20" in call_args

    @patch(_PATCH_CLIENT)
    async def test_uses_configured_url(self, mock_client_cls):
        """set_kiwix_url must be respected."""
        set_kiwix_url("http://myserver:9999")
        mock_client_cls_inst, mock_client = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("Einstein")
        call_args = str(mock_client.get.call_args)
        assert "myserver:9999" in call_args

    @patch(_PATCH_CLIENT)
    async def test_url_output_is_absolute(self, mock_client_cls):
        """Result URLs must be absolute (http://localhost:8080/content/...), not relative."""
        mock_client_cls_inst, _ = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        result = await kiwix_search("Einstein")
        assert result.exit_code == 0
        assert "http://localhost:8080/content/" in result.output
        # Relative-only path must not appear as a bare URL line
        assert "Open: /content/" not in result.output


# ─── fetch_article ────────────────────────────────────────────────────────────


class TestFetchArticle:
    @patch(_PATCH_CLIENT)
    async def test_fetch_article_true(self, mock_client_cls):
        """When fetch_article=True, article text should appear in output."""
        article_html = "<html><body><p>Einstein was born in 1879.</p></body></html>"

        search_resp = MagicMock(spec=httpx.Response)
        search_resp.status_code = 200
        search_resp.text = _SEARCH_HTML

        article_resp = MagicMock(spec=httpx.Response)
        article_resp.status_code = 200
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
        mock_client_cls_inst, mock_client = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("Einstein", fetch_article=False)
        assert mock_client.get.call_count == 1


# ─── Error handling ───────────────────────────────────────────────────────────


class TestErrorHandling:
    @patch("natshell.tools.kiwix_search._discover_kiwix_url")
    @patch(_PATCH_CLIENT)
    async def test_connection_refused(self, mock_client_cls, mock_discover):
        """ConnectError with no server found → error message."""
        mock_discover.return_value = None
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await kiwix_search("Einstein")
        assert result.exit_code == 1
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
        resp.text = ""

        mock_client = AsyncMock()
        mock_client.get.return_value = resp
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await kiwix_search("Einstein")
        assert result.exit_code == 1
        assert "404" in result.error

    @patch("natshell.tools.kiwix_search._discover_kiwix_url")
    @patch(_PATCH_CLIENT)
    async def test_auto_discovery_on_connect_error(self, mock_client_cls, mock_discover):
        """ConnectError on first attempt triggers discovery and retries."""
        mock_discover.return_value = "http://localhost:8080"

        search_resp = MagicMock(spec=httpx.Response)
        search_resp.status_code = 200
        search_resp.text = _SEARCH_HTML

        mock_client = AsyncMock()
        mock_client.get.side_effect = [
            httpx.ConnectError("refused"),  # initial attempt
            search_resp,  # retry after discovery
        ]
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await kiwix_search("Einstein")
        assert result.exit_code == 0
        assert "Albert Einstein" in result.output
        mock_discover.assert_called_once()


# ─── open_articles ────────────────────────────────────────────────────────────


class TestOpenArticles:
    @patch("natshell.tools.kiwix_search.subprocess.run")
    @patch(_PATCH_CLIENT)
    async def test_open_zero_no_subprocess(self, mock_client_cls, mock_run):
        """open_articles=0 must not call subprocess.run."""
        mock_client_cls_inst, _ = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("Einstein", open_articles=0)
        mock_run.assert_not_called()

    @patch("natshell.tools.kiwix_search.is_wsl", return_value=False)
    @patch("natshell.tools.kiwix_search.is_macos", return_value=False)
    @patch("natshell.tools.kiwix_search.shutil.which", return_value="/usr/bin/xdg-open")
    @patch("natshell.tools.kiwix_search.subprocess.run")
    @patch(_PATCH_CLIENT)
    async def test_open_one_calls_subprocess(
        self, mock_client_cls, mock_run, mock_which, mock_is_macos, mock_is_wsl
    ):
        """open_articles=1 calls subprocess.run exactly once with the correct URL."""
        mock_client_cls_inst, _ = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("Einstein", open_articles=1)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "http://localhost:8080/content/wikipedia_en/Albert_Einstein" in call_args

    @patch("natshell.tools.kiwix_search.is_wsl", return_value=False)
    @patch("natshell.tools.kiwix_search.is_macos", return_value=False)
    @patch("natshell.tools.kiwix_search.shutil.which", return_value="/usr/bin/xdg-open")
    @patch("natshell.tools.kiwix_search.subprocess.run")
    @patch(_PATCH_CLIENT)
    async def test_open_capped_at_three(
        self, mock_client_cls, mock_run, mock_which, mock_is_macos, mock_is_wsl
    ):
        """open_articles=10 must call subprocess.run at most 3 times."""
        items = "".join(
            f'<li><a href="/content/wiki/Article_{i}">Article {i}</a>'
            f"<cite>Snippet {i}</cite></li>"
            for i in range(10)
        )
        large_html = f"<html><body><ul>{items}</ul></body></html>"
        mock_client_cls_inst, _ = _make_mock_client(large_html)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        await kiwix_search("test", results=10, open_articles=10)
        assert mock_run.call_count == 3

    @patch("natshell.tools.kiwix_search.is_wsl", return_value=False)
    @patch("natshell.tools.kiwix_search.is_macos", return_value=False)
    @patch("natshell.tools.kiwix_search.shutil.which", return_value="/usr/bin/xdg-open")
    @patch("natshell.tools.kiwix_search.subprocess.run")
    @patch(_PATCH_CLIENT)
    async def test_open_reported_in_output(
        self, mock_client_cls, mock_run, mock_which, mock_is_macos, mock_is_wsl
    ):
        """'Opened in browser' must appear in output when articles are opened."""
        mock_client_cls_inst, _ = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        result = await kiwix_search("Einstein", open_articles=1)
        assert "Opened in browser" in result.output
        assert "Albert Einstein" in result.output

    @patch("natshell.tools.kiwix_search.is_wsl", return_value=False)
    @patch("natshell.tools.kiwix_search.is_macos", return_value=False)
    @patch("natshell.tools.kiwix_search.shutil.which", return_value="/usr/bin/xdg-open")
    @patch("natshell.tools.kiwix_search.subprocess.run")
    @patch(_PATCH_CLIENT)
    async def test_open_cap_warning(
        self, mock_client_cls, mock_run, mock_which, mock_is_macos, mock_is_wsl
    ):
        """Requesting more than 3 must include the cap warning in output."""
        mock_client_cls_inst, _ = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        result = await kiwix_search("Einstein", open_articles=5)
        assert "Capped at 3" in result.output

    @patch("natshell.tools.kiwix_search.is_wsl", return_value=False)
    @patch("natshell.tools.kiwix_search.is_macos", return_value=False)
    @patch("natshell.tools.kiwix_search.shutil.which", return_value=None)
    @patch(_PATCH_CLIENT)
    async def test_open_xdg_missing_linux(
        self, mock_client_cls, mock_which, mock_is_macos, mock_is_wsl
    ):
        """Missing xdg-open on Linux → error in output, no crash."""
        mock_client_cls_inst, _ = _make_mock_client(_SEARCH_HTML)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        result = await kiwix_search("Einstein", open_articles=1)
        assert result.exit_code == 0
        assert "xdg-open not found" in result.output


# ─── Output truncation ────────────────────────────────────────────────────────


class TestTruncation:
    @patch(_PATCH_CLIENT)
    async def test_output_truncated_at_limit(self, mock_client_cls):
        items = "".join(
            f'<li><a href="/content/wiki/Article_{i}">Article {i}</a>'
            f'<cite>{"x" * 200}</cite></li>'
            for i in range(20)
        )
        large_html = f"<html><body><ul>{items}</ul></body></html>"
        mock_client_cls_inst, _ = _make_mock_client(large_html)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        set_limits(ToolLimits(max_output_chars=100))
        result = await kiwix_search("test", results=20)
        assert result.truncated
        assert len(result.output) <= 100


class TestSmallContextBudget:
    """Verify per-section budget split for small context windows."""

    @patch(_PATCH_CLIENT)
    async def test_listing_capped_at_half_budget(self, mock_client_cls):
        """Search listing is capped at max_output_chars // 2."""
        items = "".join(
            f'<li><a href="/content/wiki/Article_{i}">Article {i}</a>'
            f'<cite>{"x" * 200}</cite></li>'
            for i in range(20)
        )
        large_html = f"<html><body><ul>{items}</ul></body></html>"
        mock_client_cls_inst, _ = _make_mock_client(large_html)
        mock_client_cls.return_value = mock_client_cls_inst.return_value

        set_limits(ToolLimits(max_output_chars=4000))
        result = await kiwix_search("test", results=20, fetch_article=False)
        # Listing alone may not exceed half the budget
        assert len(result.output) <= 2000

    @patch(_PATCH_CLIENT)
    async def test_article_gets_remaining_budget(self, mock_client_cls):
        """When fetch_article=True, article content is capped at remaining budget."""
        search_html = (
            '<li><a href="/content/wiki/Python">Python</a>'
            "<cite>Programming language</cite></li>"
        )
        article_html = "<html><body><p>" + ("A" * 5000) + "</p></body></html>"

        # Two separate responses: search then article
        search_resp = MagicMock(spec=httpx.Response)
        search_resp.status_code = 200
        search_resp.text = f"<html><body><ul>{search_html}</ul></body></html>"

        article_resp = MagicMock(spec=httpx.Response)
        article_resp.status_code = 200
        article_resp.text = article_html

        mock_client = AsyncMock()
        mock_client.get.side_effect = [search_resp, article_resp]
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        set_limits(ToolLimits(max_output_chars=4000))
        result = await kiwix_search("Python", fetch_article=True)
        assert result.truncated
        assert len(result.output) <= 4000
        # Article section must be present
        assert "--- Article:" in result.output

    @patch(_PATCH_CLIENT)
    async def test_small_context_total_fits_budget(self, mock_client_cls):
        """At 4000-char budget, total output stays at or under 4000 chars."""
        search_html = (
            '<li><a href="/content/wiki/Python">Python</a>'
            "<cite>Programming language</cite></li>"
        )
        article_html = "<html><body><p>" + ("B" * 10000) + "</p></body></html>"

        search_resp = MagicMock(spec=httpx.Response)
        search_resp.status_code = 200
        search_resp.text = f"<html><body><ul>{search_html}</ul></body></html>"

        article_resp = MagicMock(spec=httpx.Response)
        article_resp.status_code = 200
        article_resp.text = article_html

        mock_client = AsyncMock()
        mock_client.get.side_effect = [search_resp, article_resp]
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        set_limits(ToolLimits(max_output_chars=4000))
        result = await kiwix_search("Python", fetch_article=True)
        assert len(result.output) <= 4000


# ─── Auto-discovery ───────────────────────────────────────────────────────────


class TestAutoDiscovery:
    @patch(_PATCH_CLIENT)
    async def test_discover_and_set_uses_current_url_if_reachable(self, mock_client_cls):
        """If current URL responds, discover_and_set_kiwix_url keeps it."""
        catalog_resp = MagicMock(spec=httpx.Response)
        catalog_resp.status_code = 200
        catalog_resp.text = _CATALOG_XML

        mock_client = AsyncMock()
        mock_client.get.return_value = catalog_resp
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from natshell.tools import kiwix_search as ks

        original_url = ks._kiwix_url
        await discover_and_set_kiwix_url()
        assert ks._kiwix_url == original_url

    @patch("natshell.tools.kiwix_search._discover_kiwix_url")
    @patch(_PATCH_CLIENT)
    async def test_discover_and_set_switches_on_unreachable(
        self, mock_client_cls, mock_discover
    ):
        """If current URL is unreachable, switch to discovered URL."""
        mock_discover.return_value = "http://localhost:9090"

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from natshell.tools import kiwix_search as ks

        await discover_and_set_kiwix_url()
        assert ks._kiwix_url == "http://localhost:9090"

    def test_looks_like_kiwix_catalog_valid(self):
        """A real OPDS catalog response is accepted."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.text = _CATALOG_XML
        assert _looks_like_kiwix_catalog(resp) is True

    def test_looks_like_kiwix_catalog_rejects_html(self):
        """An HTML page (e.g. IIS default) is rejected."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.text = "<html><body><h1>Welcome to IIS</h1></body></html>"
        assert _looks_like_kiwix_catalog(resp) is False

    def test_looks_like_kiwix_catalog_rejects_404(self):
        """A 404 response is rejected even with XML-like body."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 404
        resp.text = "<feed><entry></entry></feed>"
        assert _looks_like_kiwix_catalog(resp) is False

    @patch(_PATCH_CLIENT)
    async def test_discover_skips_non_kiwix_servers(self, mock_client_cls):
        """Ports with non-Kiwix HTTP servers are not detected as kiwix-serve."""
        iis_resp = MagicMock(spec=httpx.Response)
        iis_resp.status_code = 404
        iis_resp.text = "<html><body>404 - Not Found</body></html>"

        mock_client = AsyncMock()
        mock_client.get.return_value = iis_resp
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await _discover_kiwix_url()
        assert result is None


# ─── Hot-reload via update_config ─────────────────────────────────────────────


class TestUpdateConfigHotReload:
    @patch("natshell.tools.kiwix_search.set_kiwix_url")
    async def test_update_config_kiwix_url_calls_set_kiwix_url(self, mock_set):
        """update_config kiwix url must call set_kiwix_url immediately."""
        from natshell.tools.update_config import update_config

        result = await update_config("kiwix", "url", "http://localhost:9191")
        assert result.exit_code == 0
        mock_set.assert_called_once_with("http://localhost:9191")


# ─── Registration ─────────────────────────────────────────────────────────────


class TestRegistration:
    def test_registered_in_default_registry(self):
        registry = create_default_registry()
        assert "kiwix_search" in registry.tool_names

    def test_in_plan_safe_tools(self):
        assert "kiwix_search" in PLAN_SAFE_TOOLS

    def test_in_small_context_tools(self):
        from natshell.tools.registry import SMALL_CONTEXT_TOOLS

        assert "kiwix_search" in SMALL_CONTEXT_TOOLS

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
