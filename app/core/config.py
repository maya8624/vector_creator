from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ValidationError

# Define the file path
env_path = Path(".env")

if not env_path.exists():
    raise FileNotFoundError(
        f"Critical error: Configuration file '{env_path}' not found. "
        "Please create one based on .env.example"
    )


class Settings(BaseSettings):
    """
    Docstring for Settings
    """
    POSTGRES_URL: str
    # SECRET_API_KEY: str
    CHROMA_DB_PATH: str
    DOCUMENT_PATH: str
    EMBEDDING_MODEL: str
    LLAMA_MODEL_NAME: str
    LLAMA_MODEL_PATH: str
    LLAMA_THREADS: str
    LLAMA_CONTEXT: str
    LLAMA_CLOUD_API_KEY: str
    VECTOR_TABLE: str = "document_chunks"
    EMBEDDING_DIM: int = 384  # This should match the dimension of the embedding model
    # Use SettingsConfigDict for better structure
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",         # This ignores extra variables like 'production'
        case_sensitive=False    # Usually best for env vars
    )


try:
    settings = Settings()
    print("Settings loaded successfully!")
except ValidationError as e:
    print(f"Configuration validation error:\n{e}")
