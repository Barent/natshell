---
name: web-research
description: Fetch and summarize web pages and offline kiwix docs. Use when the user asks for current information or external references.
---

# Web research skill

## When to use
- User asks to look up documentation, man pages, or external references
- User asks for current information that may not be in the model's training data
- User asks to summarize or extract key points from a URL

## When NOT to use
- The user wants to scrape many pages or crawl a site — use execute_shell with wget/curl
- The user asks about system configuration — use system-admin skill
- The user wants to download a file, not read content — use execute_shell with curl/wget

## Procedure
1. **Prefer `kiwix_search`** for offline documentation (man pages, Stack Overflow, Wikipedia, language docs).
2. Use `fetch_url` for live web when kiwix doesn't have the content.
3. Cite the source URL in every response.
4. Summarize — don't dump raw HTML at the user.
5. If the content is behind a login or paywalled, tell the user and stop.

## Recipes

**Search kiwix for documentation:**
```
kiwix_search(query="python asyncio event loop")
kiwix_search(query="bash find command examples")
```

**Fetch a web page:**
```
fetch_url(url="https://docs.example.com/api")
```

**Fetch a man page via kiwix:**
```
kiwix_search(query="man rsync")
kiwix_search(query="man page ssh_config")
```

**Fetch and extract from a specific section:**
After fetch_url returns, parse the content with Python stdlib if needed:
```python
import html.parser, urllib.request

class TextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self._skip = False
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer"):
            self._skip = True
    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer"):
            self._skip = False
    def handle_data(self, data):
        if not self._skip:
            self.text.append(data)

# Use after fetch_url has retrieved content
```

## Pitfalls
- `fetch_url` blocks internal IPs and localhost (SSRF protection) — expected behavior.
- `fetch_url` is GET-only and caps at 1 MB — not suitable for downloading large files.
- Some sites return JavaScript-only pages that render empty via fetch_url — note this and tell the user.
- Always cite source URLs. Do not present fetched content as your own knowledge.
- kiwix requires a running `kiwix-serve` instance with downloaded ZIM files. Check `kiwix_search` result for "unavailable" errors.
