from pathlib import Path
from typing import List
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, Document
from app.core.config import settings


class LlamaParseService:
    """
    Responsible for parsing documents using LlamaParse.
    Returns LlamaIndex Document objects.
    """

    def __init__(self) -> None:
        self._parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",  # may switch to "json" in the future for more structured data
        )

        self._file_extractor = {
            ".pdf": self._parser,
            ".docx": self._parser,
            ".txt": self._parser,
            ".csv": self._parser,
        }

    def parse(self, file_path: Path) -> List[Document]:
        """
        Parse a single file into LlamaIndex Documents.
        """

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        reader = SimpleDirectoryReader(
            input_files=[str(file_path)],
            file_extractor=self._file_extractor,
        )

        documents = reader.load_data()
        return documents
