Plugins let you add custom tools to NatShell.

Plugin directory: ~/.config/natshell/plugins/
Each .py file must define a register(registry) function.

Example plugin (~/.config/natshell/plugins/hello.py):

  from natshell.tools.registry import ToolDefinition, ToolResult

  def register(registry):
      registry.register(
          ToolDefinition(
              name="hello",
              description="Say hello",
              parameters={"type": "object", "properties": {}},
          ),
          handler=hello_handler,
      )

  async def hello_handler(**kwargs):
      return ToolResult(output="Hello from plugin!")

Plugins are loaded at startup. Restart NatShell after adding new plugins.