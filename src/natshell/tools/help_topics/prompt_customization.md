Customize NatShell's system prompt via the [prompt] section in ~/.config/natshell/config.toml.

Available keys:
  persona             — Replace the default role description
  extra_instructions  — Append extra instructions to the prompt

Example config:
  [prompt]
  persona = "expert Python developer and DevOps engineer"
  extra_instructions = "Always suggest the simplest solution first"

Details:
  - 'persona' replaces the role in the opening line: "You are NatShell, a {persona}..."
  - 'extra_instructions' is appended as an "Additional Instructions" section at the end of the system prompt.
  - Core safety rules, behavior rules, and code editing guidance are always included regardless of customization.
  - Changes take effect on the next message (prompt is rebuilt each turn).

You can also set these at runtime:
  update_config section="prompt" key="persona" value="senior Rust developer"
  update_config section="prompt" key="extra_instructions" value="Prefer functional style"