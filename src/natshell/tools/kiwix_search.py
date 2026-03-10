"""Search a local kiwix-serve instance for offline Wikipedia and documentation."""

from __future__ import annotations

import html
import logging
from html.parser import HTMLParser
from urllib.parse import quote_plus

import httpx

from natshell.tools.limits import ToolLimits
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# ── Configurable kiwix-serve URL (injected at startup) ───────────────────

_kiwix_url = "http://localhost:8888"


def set_kiwix_url(url: str) -> None:
    global _kiwix_url
    _kiwix_url = url


# ── Shared limits (overwritten by agent loop once n_ctx is known) ─────────

_limits = ToolLimits()


def set_limits(limits: ToolLimits) -> None:
    global _limits
    _limits = limits


def reset_limits() -> None:
    global _limits
    _limits = ToolLimits()


# ── HTML stripping ────────────────────────────────────────────────────────


class _TextExtractor(HTMLParser):
    """Extract plain text from HTML, skipping script/style content."""

    def __init__(self) -> None:
        super().__init__()
        self._text: list[str] = []
        self._skip = 0  # depth inside script/style tags

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip > 0:
            self._skip -= 1
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "li", "br", "tr"):
            if not self._skip:
                self._text.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._text.append(data)

    def get_text(self) -> str:
        raw = "".join(self._text)
        # Collapse runs of whitespace while preserving paragraph breaks
        lines = [line.strip() for line in raw.splitlines()]
        # Remove blank-line runs (keep at most one)
        result: list[str] = []
        prev_blank = False
        for line in lines:
            if not line:
                if not prev_blank:
                    result.append("")
                prev_blank = True
            else:
                result.append(line)
                prev_blank = False
        return "\n".join(result).strip()


def _strip_html(html_text: str) -> str:
    parser = _TextExtractor()
    parser.feed(html.unescape(html_text))
    return parser.get_text()


# ── Tool definition ──────────────────────────────────────────────────────

DEFINITION = ToolDefinition(
    name="kiwix_search",
    description=(
        "Search a local kiwix-serve instance for offline Wikipedia articles, "
        "Stack Overflow answers, documentation, and other ZIM archive content. "
        "Requires kiwix-serve to be running locally. Use this for factual "
        "lookups, technical documentation, and encyclopedia queries without "
        "internet access."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "book": {
                "type": "string",
                "description": (
                    "Filter results to a specific ZIM book by name "
                    "(e.g. 'wikipedia_en_mini'). Omit to search all books."
                ),
            },
            "results": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 20).",
            },
            "fetch_article": {
                "type": "boolean",
                "description": (
                    "If true, fetch and return the plain-text content of the "
                    "top result in addition to the search listing."
                ),
            },
        },
        "required": ["query"],
    },
    requires_confirmation=False,
)


# ── Handler ──────────────────────────────────────────────────────────────


async def kiwix_search(
    query: str,
    book: str = "",
    results: int = 5,
    fetch_article: bool = False,
) -> ToolResult:
    """Search kiwix-serve and optionally fetch the top article."""
    results = max(1, min(results, 20))

    # Build search URL
    params = f"query={quote_plus(query)}&start=0&end={results}"
    if book:
        params += f"&books={quote_plus(book)}"
    search_url = f"{_kiwix_url}/search?{params}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                search_url,
                headers={"Accept": "application/json"},
            )
    except httpx.ConnectError:
        return ToolResult(
            error=(
                f"Could not connect to kiwix-serve at {_kiwix_url}. "
                "Is kiwix-serve running? "
                "Start it with: kiwix-serve /path/to/file.zim"
            ),
            exit_code=1,
        )
    except httpx.TimeoutException:
        return ToolResult(
            error=f"Request timed out connecting to kiwix-serve at {_kiwix_url}.",
            exit_code=1,
        )
    except httpx.HTTPError as e:
        return ToolResult(error=f"HTTP error querying kiwix-serve: {e}", exit_code=1)

    if response.status_code != 200:
        return ToolResult(
            error=(
                f"kiwix-serve returned HTTP {response.status_code} for search. "
                f"URL: {search_url}"
            ),
            exit_code=1,
        )

    try:
        data = response.json()
    except Exception:
        return ToolResult(
            error="kiwix-serve returned non-JSON response for search.",
            exit_code=1,
        )

    items = data.get("items", [])
    if not items:
        output = f"No results found for: {query}"
        if book:
            output += f" (book: {book})"
        return ToolResult(output=output)

    # Format search results
    lines = [f"Search results for '{query}'" + (f" in '{book}'" if book else "") + ":\n"]
    for i, item in enumerate(items, 1):
        title = item.get("title", "(no title)")
        snippet = item.get("snippet", "").strip()
        url = item.get("url", "")
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if snippet:
            clean = _strip_html(snippet)
            if clean:
                lines.append(f"   {clean}")
        lines.append("")

    output = "\n".join(lines).rstrip()

    # Optionally fetch the top article
    if fetch_article and items:
        top_url = items[0].get("url", "")
        if top_url:
            # Construct full article URL
            if top_url.startswith("/"):
                article_url = f"{_kiwix_url}{top_url}"
            else:
                article_url = f"{_kiwix_url}/{top_url}"

            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    article_resp = await client.get(article_url)
                if article_resp.status_code == 200:
                    article_text = _strip_html(article_resp.text)
                    output += f"\n\n--- Article: {items[0].get('title', '')} ---\n{article_text}"
                else:
                    output += f"\n\n(Could not fetch article: HTTP {article_resp.status_code})"
            except httpx.HTTPError as e:
                output += f"\n\n(Could not fetch article: {e})"

    # Apply context-adaptive output truncation
    max_chars = _limits.max_output_chars
    truncated = False
    if len(output) > max_chars:
        output = output[:max_chars]
        truncated = True

    return ToolResult(output=output, truncated=truncated)
