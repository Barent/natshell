"""Tests for the backup system."""

from __future__ import annotations

import pytest

from natshell.backup import BackupManager, get_backup_manager, reset_backup_manager


@pytest.fixture
def backup_dir(tmp_path):
    """Provide a temporary backup directory."""
    return tmp_path / "backups"


@pytest.fixture
def manager(backup_dir):
    """Provide a BackupManager with a temp directory."""
    return BackupManager(backup_dir=backup_dir, max_per_file=3)


@pytest.fixture(autouse=True)
def _reset_singleton():
    yield
    reset_backup_manager()


# ─── Backup creation ────────────────────────────────────────────────────────


class TestBackupCreation:
    def test_backup_creates_file(self, manager, tmp_path):
        src = tmp_path / "test.txt"
        src.write_text("original content")
        result = manager.backup(src)
        assert result is not None
        assert result.exists()
        assert result.read_text() == "original content"

    def test_backup_returns_none_for_missing_file(self, manager):
        result = manager.backup("/nonexistent/file.txt")
        assert result is None

    def test_backup_returns_none_for_directory(self, manager, tmp_path):
        result = manager.backup(tmp_path)
        assert result is None

    def test_backup_preserves_original(self, manager, tmp_path):
        src = tmp_path / "test.txt"
        src.write_text("original content")
        manager.backup(src)
        assert src.read_text() == "original content"

    def test_backup_filename_contains_stem(self, manager, tmp_path, backup_dir):
        src = tmp_path / "myfile.py"
        src.write_text("code")
        result = manager.backup(src)
        assert "myfile" in result.name
        assert result.suffix == ".bak"


# ─── Undo ────────────────────────────────────────────────────────────────────


class TestUndo:
    def test_undo_restores_file(self, manager, tmp_path):
        src = tmp_path / "test.txt"
        src.write_text("original")
        manager.backup(src)
        src.write_text("modified")
        assert src.read_text() == "modified"
        success, msg = manager.undo_last()
        assert success
        assert src.read_text() == "original"
        assert "Restored" in msg

    def test_undo_removes_backup(self, manager, tmp_path):
        src = tmp_path / "test.txt"
        src.write_text("original")
        backup_path = manager.backup(src)
        src.write_text("modified")
        manager.undo_last()
        assert not backup_path.exists()

    def test_undo_empty_history(self, manager):
        success, msg = manager.undo_last()
        assert not success
        assert "No backups" in msg

    def test_undo_missing_backup_file(self, manager, tmp_path):
        src = tmp_path / "test.txt"
        src.write_text("original")
        backup_path = manager.backup(src)
        backup_path.unlink()  # Simulate missing backup
        success, msg = manager.undo_last()
        assert not success
        assert "missing" in msg.lower()

    def test_multiple_undos(self, manager, tmp_path):
        src = tmp_path / "test.txt"
        src.write_text("v1")
        manager.backup(src)
        src.write_text("v2")
        manager.backup(src)
        src.write_text("v3")

        success, _ = manager.undo_last()
        assert success
        assert src.read_text() == "v2"

        success, _ = manager.undo_last()
        assert success
        assert src.read_text() == "v1"


# ─── Pruning ─────────────────────────────────────────────────────────────────


class TestPruning:
    def test_prune_keeps_max_per_file(self, manager, tmp_path, backup_dir):
        src = tmp_path / "test.txt"
        for i in range(5):
            src.write_text(f"version {i}")
            manager.backup(src)

        backups = list(backup_dir.glob("test.*.bak"))
        assert len(backups) <= 3  # max_per_file=3


# ─── Integration with edit_file ──────────────────────────────────────────────


class TestEditFileIntegration:
    async def test_edit_file_creates_backup(self, tmp_path, monkeypatch):
        from natshell.tools.edit_file import edit_file
        from natshell.tools.file_tracker import reset_tracker

        reset_tracker()
        backup_dir = tmp_path / "backups"
        monkeypatch.setattr("natshell.backup._instance._backup_dir", backup_dir)

        src = tmp_path / "code.py"
        src.write_text("old_value = 1\n")

        result = await edit_file(str(src), "old_value = 1", "new_value = 2")
        assert result.exit_code == 0

        backups = list(backup_dir.glob("code.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "old_value = 1\n"
        reset_tracker()


# ─── Integration with write_file ─────────────────────────────────────────────


class TestWriteFileIntegration:
    async def test_write_file_creates_backup_on_overwrite(self, tmp_path, monkeypatch):
        from natshell.tools.write_file import write_file

        backup_dir = tmp_path / "backups"
        monkeypatch.setattr("natshell.backup._instance._backup_dir", backup_dir)

        src = tmp_path / "data.txt"
        src.write_text("original data")

        result = await write_file(str(src), "new data", mode="overwrite")
        assert result.exit_code == 0

        backups = list(backup_dir.glob("data.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "original data"

    async def test_write_file_new_file_no_backup(self, tmp_path, monkeypatch):
        from natshell.tools.write_file import write_file

        backup_dir = tmp_path / "backups"
        monkeypatch.setattr("natshell.backup._instance._backup_dir", backup_dir)

        src = tmp_path / "new_file.txt"
        result = await write_file(str(src), "brand new")
        assert result.exit_code == 0

        # No backup for a new file (didn't exist before)
        backups = list(backup_dir.glob("*.bak"))
        assert len(backups) == 0


# ─── Singleton ────────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_backup_manager_returns_same_instance(self):
        m1 = get_backup_manager()
        m2 = get_backup_manager()
        assert m1 is m2

    def test_reset_clears_history(self, tmp_path):
        mgr = get_backup_manager()
        src = tmp_path / "test.txt"
        src.write_text("data")
        mgr.backup(src)
        assert len(mgr._history) > 0
        reset_backup_manager()
        assert len(mgr._history) == 0


# ─── Security ────────────────────────────────────────────────────────────────


class TestBackupSecurity:
    def test_symlink_returns_none(self, manager, tmp_path):
        real_file = tmp_path / "real.txt"
        real_file.write_text("secret data")
        link = tmp_path / "link.txt"
        link.symlink_to(real_file)
        result = manager.backup(link)
        assert result is None

    def test_backup_directory_permissions_0o700(self, manager, tmp_path):
        src = tmp_path / "test.txt"
        src.write_text("data")
        manager.backup(src)
        mode = manager._backup_dir.stat().st_mode & 0o777
        assert mode == 0o700
