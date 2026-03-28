import pytest

from app.loaders.folder_loader import FolderLoader


def test_raises_if_folder_does_not_exist(tmp_path):
    loader = FolderLoader(str(tmp_path / "nonexistent"))
    with pytest.raises(FileNotFoundError):
        loader.load_files()


def test_returns_only_allowed_extensions(tmp_path):
    (tmp_path / "doc.pdf").write_text("pdf")
    (tmp_path / "doc.docx").write_bytes(b"docx")
    (tmp_path / "doc.txt").write_text("txt")
    (tmp_path / "image.png").write_bytes(b"png")
    (tmp_path / "data.csv").write_text("csv")

    loader = FolderLoader(str(tmp_path))
    files = loader.load_files()
    suffixes = {f.suffix for f in files}

    assert suffixes == {".pdf", ".docx", ".txt"}
    assert len(files) == 3


def test_returns_files_sorted(tmp_path):
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "c.txt").write_text("c")

    loader = FolderLoader(str(tmp_path))
    files = loader.load_files()

    assert [f.name for f in files] == ["a.txt", "b.txt", "c.txt"]


def test_discovers_files_recursively(tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    (tmp_path / "root.txt").write_text("root")
    (sub / "nested.pdf").write_text("nested")

    loader = FolderLoader(str(tmp_path))
    files = loader.load_files()

    assert len(files) == 2


def test_returns_empty_list_for_empty_folder(tmp_path):
    loader = FolderLoader(str(tmp_path))
    assert loader.load_files() == []


def test_extension_check_is_case_insensitive(tmp_path):
    (tmp_path / "doc.PDF").write_text("pdf upper")
    (tmp_path / "doc.TXT").write_text("txt upper")

    loader = FolderLoader(str(tmp_path))
    files = loader.load_files()

    assert len(files) == 2
