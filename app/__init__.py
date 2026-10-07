"""
app package - Cred Domain Support Agent (Banking & FinTech)

Directory layout:
app/
├── __init__.py
├── db.py
├── main.py
├── memory.py
├── models.py
├── pipeline.py
└── tools.py
"""

from .models import (
    AgentResponseSchema,
    VerdictModel,
    AskRequest,
    AddDocumentRequest,
    LoanApplicationRecord,
)
from .db import (
    LOAN_APPLICATIONS,
    LOAN_APPLICATIONS_BY_ID,
    get_loan_application,
    generate_loan_applications,
    get_vector_store,
    VectorStore,
    LocalVectorStore,
    load_knowledge_base_documents,
)
from .memory import (
    get_session_memory,
    SessionMemoryManager,
    InMemoryChatMessageHistory,
)
from .tools import (
    check_loan_application_status,
    calculate_escalation_score,
    rag_lookup,
    ESCALATION_THRESHOLD,
)
from .pipeline import (
    CredSupportCrew,
    get_support_crew,
    run_pipeline,
    AutoGenReviewTeam,
    get_review_team,
)
from .main import app

__all__ = [
    "app",
    "AgentResponseSchema",
    "VerdictModel",
    "AskRequest",
    "AddDocumentRequest",
    "LoanApplicationRecord",
    "LOAN_APPLICATIONS",
    "LOAN_APPLICATIONS_BY_ID",
    "get_loan_application",
    "generate_loan_applications",
    "get_vector_store",
    "VectorStore",
    "LocalVectorStore",
    "load_knowledge_base_documents",
    "get_session_memory",
    "SessionMemoryManager",
    "InMemoryChatMessageHistory",
    "check_loan_application_status",
    "calculate_escalation_score",
    "rag_lookup",
    "ESCALATION_THRESHOLD",
    "CredSupportCrew",
    "get_support_crew",
    "run_pipeline",
    "AutoGenReviewTeam",
    "get_review_team",
]
