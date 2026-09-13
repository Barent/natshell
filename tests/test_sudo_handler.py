"""Tests for the ``natshell.tools.sudo`` extraction (SudoHandler unit).

The historical ``execute_shell`` sudo helpers moved to ``natshell.tools.sudo``
with byte-identical behaviour and a thin ``SudoHandler`` facade.  These tests
pin the *extraction* itself:

- the facade and the module-level function API share one password state;
- the historical ``execute_shell`` re-export names still resolve and behave
  exactly as before (aliasing, constants, the public set/clear pair);
- the facade exposes exactly the module's function API.

Behavioural parity of the helper *bodies* (injection rules, scrubbing,
needs-password detection, package-manager tail) is covered by the pre-existing
tests in ``test_tools.py`` and ``test_stream_execute_shell.py``.
"""

from __future__ import annotations

import natshell.tools.execute_shell as ex
import natshell.tools.sudo as sudo
from natshell.tools.execute_shell import SUDO, SudoHandler
from natshell.tools.registry import ToolResult


def test_module_and_exports_exist():
    assert sudo.SudoHandler is SudoHandler
    assert SUDO is sudo.SUDO
    assert isinstance(SUDO, SudoHandler)
    # Historical re-exports on execute_shell.
    assert ex.SudoHandler is SudoHandler
    assert ex.SUDO is SUDO
    for name in (
        "set_sudo_password",
        "clear_sudo_password",
        "needs_sudo_password",
        "_has_sudo_invocation",
        "_inject_sudo_dash_s",
        "_prepare_sudo",
        "_scrub_sudo_prompt",
        "_SUDO_NEEDS_PW",
        "_PKG_MANAGER_RE",
        "_SUDO_PW_TIMEOUT",
    ):
        assert hasattr(ex, name), f"execute_shell lost historical name {name}"


def test_facade_function_api_matches_module():
    """A SudoHandler instance must expose exactly the module function API."""
    handler = SudoHandler()
    for name in (
        "has_invocation",
        "needs_password",
        "prepare_for_run",
        "scrub_prompt",
        "get_password",
        "set_password",
        "clear_password",
        "inject_dash_s",
    ):
        assert hasattr(handler, name), f"SudoHandler lost method {name}"
        assert hasattr(sudo, name), f"sudo module lost function {name}"


def test_state_shared_between_facade_and_module_functions():
    """Setting a password on the facade is visible via the free-function API."""
    sudo.clear_password()
    try:
        SUDO.set_password("pw-facade")
        assert sudo.get_password() == "pw-facade"
        assert ex._get_sudo_password() == "pw-facade"

        ex.set_sudo_password("pw-alias")
        assert SUDO.get_password() == "pw-alias"

        sudo.clear_password()
        assert sudo.get_password() is None
        assert SUDO.get_password() is None
    finally:
        sudo.clear_password()


def test_both_clear_spellings_clear_the_same_state():
    """``clear_password`` and the historical ``clear_sudo_password`` are one."""
    sudo.set_password("pw")
    ex.clear_sudo_password()
    assert sudo.get_password() is None

    sudo.set_password("pw")
    sudo.clear_password()
    assert ex._get_sudo_password() is None


def test_facade_delegates_behaviourally():
    """Facade calls return identical results to the free-function calls."""
    cmd = "sudo apt update"
    handler = SudoHandler()
    assert handler.has_invocation(cmd) == sudo.has_invocation(cmd)

    injected, count = handler.inject_dash_s(cmd)
    assert (injected, count) == sudo.inject_dash_s(cmd)
    assert injected == "sudo -S apt update" and count == 1

    result = ToolResult(
        output="sudo: a password is required",
        error="",
        exit_code=1,
    )
    assert handler.needs_password(result) == sudo.needs_password(result) is True

    stderr = "[sudo] password for barent: \nok"
    scrubbed = handler.scrub_prompt(stderr, "pw")
    assert scrubbed == sudo.scrub_prompt(stderr, "pw") == "ok"
    assert sudo.scrub_prompt(stderr, None) == stderr  # no-op without a password


def test_prepare_for_run_facade_matches_module():
    """The facade's run preparation matches the module's, including tail."""
    sudo.clear_password()
    try:
        sudo.set_password("pw")
        cmd, payload = SUDO.prepare_for_run("sudo apt install nmap")
        assert (cmd, payload) == sudo.prepare_for_run("sudo apt install nmap")
        assert payload is not None and payload.endswith("y\ny\ny\n")
        assert cmd == "sudo -S apt install nmap"

        # Without a cached password → no injection, no stdin payload.
        sudo.clear_password()
        assert SUDO.prepare_for_run("sudo apt install nmap") == (
            "sudo apt install nmap",
            None,
        )

        # Explicit empty-string password also means "no injection".
        assert sudo.prepare_for_run("sudo ls", "") == ("sudo ls", None)
    finally:
        sudo.clear_password()


def test_no_import_cycle_sudo_does_not_import_execute_shell():
    """``sudo`` must stay a leaf: it cannot import back from execute_shell."""
    import pathlib

    source = pathlib.Path(sudo.__file__).read_text()
    assert "import" in source
    assert "execute_shell" not in "".join(
        line.split("#", 1)[0]
        for line in source.splitlines()
        if line.lstrip().startswith(("import", "from"))
    )
