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
    Classify this real estate document into one of the following types:

    Tenancy documents:
    - lease                  (residential tenancy agreement, lease agreement)
    - water_bill             (water usage invoice, utility bill, water charge notice)
    - inspection_notice      (routine inspection notice, entry notice, condition report)
    - renewal_offer          (lease renewal offer, rent renewal letter)
    - maintenance_log        (maintenance request, repair log, tradesperson invoice)
    - bond_lodgement         (bond receipt, bond lodgement confirmation, rental bond)
    - rent_ledger            (rent statement, payment history, rental ledger)
    - notice                 (rent increase notice, breach notice, vacating notice, termination notice)

    General documents:
    - legislation            (residential tenancies act, tenancy legislation, NSW RTA, government act)
    - faq                    (frequently asked questions)
    - policy                 (agency policy, terms of service, anything that does not fit the above)
    - guide                  (tenant guide, how-to guide)

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
        doc_type = "policy"

    return doc_type
