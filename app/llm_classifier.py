# from langchain_ollama import ChatOllama
# Change LLAMA_MODEL_PATH to LLAMA_MODEL_NAME in config.py
from app.core.config import settings
from langchain_ollama import ChatOllama

# Initialize the model using the Ollama service
# In config.py, set LLAMA_MODEL_NAME = "llama3.2:3b-instruct-q4_K_M"
llm = ChatOllama(
    model=settings.LLAMA_MODEL_NAME,
    temperature=0,
    num_predict=1024,
)


def classify_with_llama(text: str, filename: str = "") -> str:
    """
    Use Ollama to classify documents.
    """
    filename_hint = f"\nFilename: {filename}" if filename else ""
    prompt = f"""
        Classify this document into one of the following types:
        - faq, policy, guide, contract, report, notice, invoice, generic
        {filename_hint}
        Document excerpt:
        {text}

        Respond with ONLY the type name.
        """

    try:
        # LangChain's standardized invoke method
        response = llm.invoke(prompt)
        doc_type = response.content.strip().lower()
    except Exception as e:
        print(f"Classification Error: {e}")
        doc_type = "generic"

    return doc_type
