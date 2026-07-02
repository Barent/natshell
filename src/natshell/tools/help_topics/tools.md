Agent tools available during operation:
  execute_shell  — Run a bash command and return output (with safety classification)
  read_file      — Read file contents (line limit scales with context window)
  write_file     — Write/append to a file (always requires confirmation)
  edit_file      — Search-and-replace edit in an existing file (requires confirmation, unique match)
  list_directory — List directory contents with sizes and types
  search_files   — Text search (grep) or file search (find) in a directory
  run_code       — Execute code snippets in 10 languages (python, javascript, bash, ruby, perl, php, c, cpp, rust, go)
  git_tool       — Structured git operations (status, diff, log, branch, commit, stash)
  fetch_url      — Fetch a URL and return its content (blocks private/internal IPs)
  kiwix_search   — Search a local kiwix-serve instance for offline Wikipedia and documentation
  natshell_help  — Look up NatShell documentation by topic (this tool)
  update_config  — Update a NatShell config value (saves to disk + applies live)