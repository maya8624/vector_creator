from pathlib import Path
from typing import ClassVar


class FolderLoader:

    ALLOWED_EXTENSIONS: ClassVar[set[str]] = {".pdf", ".docx", ".txt", "csv"}

    def __init__(self, folder_path: str) -> None:
        self.folder_path = Path(folder_path)

    def load_files(self) -> list[Path]:
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        return self.get_sorted_file_paths()

    def get_sorted_file_paths(self):
        files = []
        for file in self.folder_path.rglob("*"):
            if file.is_file() and file.suffix.lower() in self.ALLOWED_EXTENSIONS:
                files.append(file)
        return sorted(files)
