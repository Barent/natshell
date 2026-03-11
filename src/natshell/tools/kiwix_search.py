"""Search a local kiwix-serve instance for offline Wikipedia and documentation."""

from __future__ import annotations

import html
import logging
import re
import shutil
import subprocess
from html.parser import HTMLParser
from urllib.parse import quote_plus, urljoin

import httpx

from natshell.platform import is_macos, is_wsl
from natshell.tools.limits import ToolLimits
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# ── Configurable kiwix-serve URL (injected at startup) ───────────────────

_kiwix_url = "http://localhost:8080"
_known_books: list[str] = []

_DISCOVERY_PORTS = [8080, 8888, 80, 9090]
_MAX_OPEN = 3
_DISCOVERY_PATH = "/catalog/v2/entries"


def _open_in_browser(url: str) -> str | None:
    """Open a URL in the default browser. Returns error string or None on success."""
    if is_macos():
        cmd = ["open", url]
    elif is_wsl():
        cmd = ["cmd.exe", "/c", "start", url]
    else:
        xdg = shutil.which("xdg-open")
        if not xdg:
            return "xdg-open not found — install xdg-utils to open articles in browser"
        cmd = [xdg, url]
    try:
        subprocess.run(cmd, timeout=5, check=False, capture_output=True)
        return None
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)


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


# ── Search result HTML parser ─────────────────────────────────────────────


class _SearchResultParser(HTMLParser):
    """Parse kiwix-serve HTML search results into structured data."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_link = False
        self._in_cite = False
        self._pending_url = ""
        self._pending_title: list[str] = []
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs_dict = dict(attrs)
        if tag == "a":
            href = attrs_dict.get("href", "")
            if "/content/" in href:
                self._in_link = True
                self._pending_url = href
                self._pending_title = []
        elif tag == "cite":
            self._in_cite = True
            self._snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False
        elif tag == "cite" and self._in_cite:
            self._in_cite = False
            if self._pending_url:
                self.results.append(
                    {
                        "url": self._pending_url,
                        "title": "".join(self._pending_title).strip(),
                        "snippet": "".join(self._snippet_parts).strip(),
                    }
                )
                self._pending_url = ""
                self._pending_title = []

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._pending_title.append(data)
        elif self._in_cite:
            self._snippet_parts.append(data)


def _parse_search_html(html_text: str) -> list[dict[str, str]]:
    """Parse kiwix-serve search HTML and return list of {url, title, snippet}."""
    parser = _SearchResultParser()
    parser.feed(html_text)
    return parser.results


# ── Auto-discovery ────────────────────────────────────────────────────────


async def _discover_kiwix_url() -> str | None:
    """Probe common ports to find a running kiwix-serve instance."""
    async with httpx.AsyncClient(timeout=2.0) as client:
        for port in _DISCOVERY_PORTS:
            candidate = f"http://localhost:{port}"
            try:
                resp = await client.get(f"{candidate}{_DISCOVERY_PATH}")
                if resp.status_code < 500:
                    return candidate
            except (httpx.ConnectError, httpx.TimeoutException):
                continue
    return None


async def _discover_books(base_url: str) -> list[str]:
    """Fetch book/content IDs from the kiwix-serve catalog."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}{_DISCOVERY_PATH}")
        if resp.status_code == 200:
            return re.findall(r"<name>([^<]+)</name>", resp.text)
    except Exception:
        pass
    return []


async def discover_and_set_kiwix_url() -> None:
    """Auto-detect kiwix-serve at startup. Silently does nothing if not found."""
    global _kiwix_url, _known_books

    # Try the currently configured URL first
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{_kiwix_url}{_DISCOVERY_PATH}")
        if resp.status_code < 500:
            _known_books = await _discover_books(_kiwix_url)
            return
    except (httpx.ConnectError, httpx.TimeoutException):
        pass

    # Current URL unreachable — probe other ports
    discovered = await _discover_kiwix_url()
    if discovered:
        _kiwix_url = discovered
        print(f"Found kiwix-serve at {discovered}")
        _known_books = await _discover_books(discovered)


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
            "open_articles": {
                "type": "integer",
                "description": (
                    "Open the top N results in the default browser (max 3). "
                    "Requires xdg-open on Linux, works natively on macOS/WSL. "
                    "Default 0 (do not open)."
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
    open_articles: int = 0,
) -> ToolResult:
    """Search kiwix-serve and optionally fetch the top article."""
    global _kiwix_url
    results = max(1, min(results, 20))

    # Build search URL using the correct kiwix-serve API
    params = f"pattern={quote_plus(query)}&start=0&pageLength={results}"
    if book:
        params += f"&content={quote_plus(book)}"
    search_url = f"{_kiwix_url}/search?{params}"

    response = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(search_url)
    except httpx.ConnectError:
        # Try auto-discovery before giving up
        discovered = await _discover_kiwix_url()
        if discovered:
            _kiwix_url = discovered
            search_url = f"{_kiwix_url}/search?{params}"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(search_url)
            except httpx.HTTPError as e:
                return ToolResult(error=f"HTTP error querying kiwix-serve: {e}", exit_code=1)
        else:
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

    if response is None or response.status_code != 200:
        code = response.status_code if response is not None else "?"
        return ToolResult(
            error=(
                f"kiwix-serve returned HTTP {code} for search. "
                f"URL: {search_url}"
            ),
            exit_code=1,
        )

    # Parse HTML search results
    items = _parse_search_html(response.text)
    if not items:
        output = f"No results found for: {query}"
        if book:
            output += f" (book: {book})"
        return ToolResult(output=output)

    # Format search results
    header = f"Search results for '{query}'" + (f" in '{book}'" if book else "")
    if _known_books:
        header += f"\nAvailable books: {', '.join(_known_books[:5])}"
        if len(_known_books) > 5:
            header += f" (+{len(_known_books) - 5} more)"
    lines = [header + ":\n"]
    for i, item in enumerate(items, 1):
        title = item.get("title", "(no title)")
        snippet = item.get("snippet", "").strip()
        url = item.get("url", "")
        lines.append(f"{i}. {title}")
        if url:
            full_url = urljoin(_kiwix_url, url)
            lines.append(f"   Open: {full_url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    output = "\n".join(lines).rstrip()

    # Optionally open top results in the browser
    if open_articles > 0:
        n_open = min(open_articles, _MAX_OPEN)
        opened = []
        errors = []
        for item in items[:n_open]:
            item_url = urljoin(_kiwix_url, item["url"])
            err = _open_in_browser(item_url)
            if err:
                errors.append(err)
                break
            else:
                opened.append(item["title"])
        if opened:
            output += f"\n\nOpened in browser: {', '.join(opened)}"
        if open_articles > _MAX_OPEN:
            output += f"\n(Capped at {_MAX_OPEN} articles opened per call)"
        if errors:
            output += f"\n(Could not open in browser: {errors[0]})"

    # Apply per-section budget: listing gets at most half the total budget so
    # the article fetch has room.  This prevents a long search listing from
    # crowding out the article content on small context windows.
    max_chars = _limits.max_output_chars
    listing_cap = max_chars // 2
    truncated = False
    if len(output) > listing_cap:
        output = output[:listing_cap]
        truncated = True

    # Optionally fetch the top article
    if fetch_article and items:
        top_url = items[0].get("url", "")
        if top_url:
            article_url = urljoin(_kiwix_url, top_url)
            article_cap = max_chars - len(output)
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    article_resp = await client.get(article_url)
                if article_resp.status_code == 200:
                    article_text = _strip_html(article_resp.text)
                    title = items[0].get("title", "")
                    if len(article_text) > article_cap:
                        article_text = article_text[:article_cap]
                        output += f"\n\n--- Article: {title} (truncated) ---\n{article_text}"
                        truncated = True
                    else:
                        output += f"\n\n--- Article: {title} ---\n{article_text}"
                else:
                    output += f"\n\n(Could not fetch article: HTTP {article_resp.status_code})"
            except httpx.HTTPError as e:
                output += f"\n\n(Could not fetch article: {e})"

    # Final safety cap in case article + listing still exceeds budget
    if len(output) > max_chars:
        output = output[:max_chars]
        truncated = True

    return ToolResult(output=output, truncated=truncated)
