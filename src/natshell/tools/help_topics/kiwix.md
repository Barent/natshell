kiwix_search — Offline Wikipedia and documentation search

The kiwix_search tool queries a local kiwix-serve instance, which serves ZIM archive files (offline copies of Wikipedia, Stack Overflow, documentation sets, and more) without internet access.

Requirements:
  kiwix-serve must be running before using this tool.
  Start it with: kiwix-serve /path/to/file.zim
  Download ZIM files from: https://download.kiwix.org/

Parameters:
  query          — Search query (required)
  book           — Filter to a specific ZIM book by name (e.g. 'wikipedia_en_mini')
  results        — Number of results (default 5, max 20)
  fetch_article  — If true, fetch and return full text of the top result

Example usage:
  'search kiwix for Albert Einstein'
  'what does Wikipedia say about the Eiffel Tower' (with fetch_article: true)
  'look up Python decorators in the documentation'

Configuration:
  Default URL: http://localhost:8888
  To change: update_config kiwix.url http://myserver:8888
  Or set in config.toml:
    [kiwix]
    url = "http://localhost:8888"