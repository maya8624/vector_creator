from pathlib import Path


class FolderLoader:
    def __init__(self, folder_path: str) -> None:
        self.folder_path = Path(folder_path)

    def load_files(self) -> list[Path]:
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        return sorted([file for file in self.folder_path.rglob("*") if file.is_file()])
