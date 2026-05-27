import json
import logging
from pathlib import Path

from sqlalchemy import Column, MetaData, Table, Text, select
from sqlalchemy.dialects.postgresql import JSONB

from app.core.config import settings
from app.database.postgres_service import PostgresService
from app.llm_classifier import llm

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve(
).parent.parent.parent / "data" / "suggestions"

_metadata = MetaData()
_vector_table = Table(
    f"data_{settings.VECTOR_TABLE}",
    _metadata,
    Column("text", Text),
    Column("metadata_", JSONB),
)


def generate_suggestions_for_document(
    file_name: str,
) -> dict[str, object]:
    """
    Fetch stored chunks for a document, generate 5 property-agent questions via LLM,
    and save the result payload to data/suggestions/<doc_id>.json.

    Args:
        file_name: Exact file_name stored in pgvector metadata (e.g. 'lease-template.pdf').

    Returns:
        Dict matching SaveDocumentSuggestionRequest.
    """
    chunks, stored_doc_id, stored_user_id = _fetch_document_data(file_name)

    if not chunks:
        raise ValueError(
            f"No chunks found in vector store for file: {file_name!r}")

    full_content = "\n\n".join(chunks)
    logger.info("Fetched %d chunks for %r (%d chars)",
                len(chunks), file_name, len(full_content))

    suggestions = _generate_with_llm(full_content)

    payload = {
        "docId": stored_doc_id,
        "userId": stored_user_id,
        "suggestions": suggestions,
        "modelUsed": settings.LLAMA_MODEL_NAME,
    }

    _save_payload(str(payload["docId"]), payload)

    return payload



def _fetch_document_data(file_name: str) -> tuple[list[str], str, str]:
    db = PostgresService()
    query = (
        select(
            _vector_table.c.text,
            _vector_table.c.metadata_["doc_id"].as_string(),
            _vector_table.c.metadata_["tenant_id"].as_string(),
        )
        .where(_vector_table.c.metadata_["file_name"].as_string() == file_name)
    )
    with db.create_session() as session:
        rows = session.execute(query).fetchall()

    if not rows:
        return [], "", ""

    chunks = [row[0] for row in rows if row[0]]
    doc_id = rows[0][1] or ""
    user_id = rows[0][2] or ""
    return chunks, doc_id, user_id


def _generate_with_llm(content: str) -> list[str]:
    prompt = (
        "Based on the document below, generate exactly 5 questions a property agent would ask a client.\n"
        "Return a JSON array of 5 strings only — no preamble, no explanation.\n\n"
        f"Document:\n{content}"
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()

    try:
        suggestions = json.loads(raw)
        if not isinstance(suggestions, list):
            raise ValueError("LLM did not return a JSON array")
        return suggestions
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("LLM response was not valid JSON: %s", raw[:200])
        raise RuntimeError(
            f"Failed to parse LLM suggestions response: {e}") from e


def _save_payload(doc_id: str, payload: dict[str, object]) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / f"{doc_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved suggestion payload to %s", out_path)
