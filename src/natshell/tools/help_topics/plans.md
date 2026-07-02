Plans let you break complex tasks into structured steps.

Commands:
  /plan <description>    — Generate a plan from a description
  /exeplan <file>        — Preview a plan file (shows steps)
  /exeplan run <file>    — Execute all steps sequentially

Plan file format (Markdown):
  ## Step 1: Description
  Details of what to do in this step.

  ## Step 2: Another step
  More details here.

Each step is executed with a dedicated agent budget. The agent reads the step description and works autonomously to complete it before moving to the next step.