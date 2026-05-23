"""
Run: python generate_suggestions.py

Reads doc_type_manifest.json and generates property-agent suggestions for
every file listed. Saves each payload to data/suggestions/<doc_id>.json
— ready to POST to the .NET document-suggestions endpoint.
"""
import json
import logging
from pathlib import Path

from app.suggestions.suggestion_service import generate_suggestions_for_document

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_MANIFEST_PATH = Path(__file__).parent / "data" / "doc_type_manifest.json"


def main() -> None:
    manifest: dict[str, str] = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    file_names = list(manifest.keys())

    print(f"Found {len(file_names)} files in manifest\n")

    for file_name in file_names:
        print(f"--- {file_name} ---")
        try:
            payload = generate_suggestions_for_document(file_name=file_name)
            print(f"  suggestions: {len(payload['suggestions'])}")
            print(f"  saved to: data/suggestions/{payload['docId']}.json\n")
        except ValueError as e:
            print(f"  skipping — {e}\n")


if __name__ == "__main__":
    main()
