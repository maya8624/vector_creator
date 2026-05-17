from typing import Final

# Document types
DOC_TYPE_FAQ: Final[str] = "faq"
DOC_TYPE_POLICY: Final[str] = "policy"
DOC_TYPE_GUIDE: Final[str] = "guide"
DOC_TYPE_CONTRACT: Final[str] = "contract"
DOC_TYPE_REPORT: Final[str] = "report"
DOC_TYPE_NOTICE: Final[str] = "notice"
DOC_TYPE_INVOICE: Final[str] = "invoice"
DOC_TYPE_GENERIC: Final[str] = "generic"

ALLOWED_DOC_TYPES: Final[frozenset[str]] = frozenset({
    DOC_TYPE_FAQ,
    DOC_TYPE_POLICY,
    DOC_TYPE_GUIDE,
    DOC_TYPE_CONTRACT,
    DOC_TYPE_REPORT,
    DOC_TYPE_NOTICE,
    DOC_TYPE_INVOICE,
    DOC_TYPE_GENERIC,
    # DOC_TYPE_LEGISLATION
})

# Rule-based classification keywords
FAQ_KEYWORDS: Final[tuple[str, ...]] = ("q:", "a:")

POLICY_KEYWORDS: Final[tuple[str, ...]] = (
    "policy",
    "terms",
    "agency profile",
    "about us"
)

GUIDE_KEYWORDS: Final[tuple[str, ...]] = (
    "step 1",
    "guide",
    "suburb",
    "median rent",
    "vacancy rate",
    "transport",
    "schools"
)

# NOTE: lease-template.pdf and nsw-tenancy-act.pdf both classify as "contract"
# If legislation needs to be distinguished from agency leases in future,
# add DOC_TYPE_LEGISLATION = "legislation" and update CLASSIFICATION_RULES
CONTRACT_KEYWORDS: Final[tuple[str, ...]] = (
    "lease",
    "agreement",
    "signed by",
    "tenant",
    "landlord",
    "bond"
)

REPORT_KEYWORDS: Final[tuple[str, ...]] = (
    "market report",
    "valuation",
    "inspection",
    "summary report"
)

NOTICE_KEYWORDS: Final[tuple[str, ...]] = (
    "notice to",
    "hereby notified",
    "vacate",
    "breach"
)

INVOICE_KEYWORDS: Final[tuple[str, ...]] = (
    "invoice",
    "amount due",
    "payment receipt",
    "total:",
    "fee schedule",
    "fees payable",
    "schedule of fees"
)

# Ordered rules: (doc_type, keywords, match_all)
# match_all=True  → all keywords must be present (e.g. FAQ requires both "q:" and "a:")
# match_all=False → any keyword is sufficient
CLASSIFICATION_RULES: Final[tuple[tuple[str, tuple[str, ...], bool], ...]] = (
    (DOC_TYPE_FAQ,      FAQ_KEYWORDS,      True),
    (DOC_TYPE_NOTICE,   NOTICE_KEYWORDS,   False),
    (DOC_TYPE_CONTRACT, CONTRACT_KEYWORDS, False),
    (DOC_TYPE_INVOICE,  INVOICE_KEYWORDS,  False),
    (DOC_TYPE_REPORT,   REPORT_KEYWORDS,   False),
    (DOC_TYPE_POLICY,   POLICY_KEYWORDS,   False),
    (DOC_TYPE_GUIDE,    GUIDE_KEYWORDS,    False),
)
