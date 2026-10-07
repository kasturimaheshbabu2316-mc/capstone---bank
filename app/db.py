"""
app/db.py - Database Layer: Seeded Loan Applications & ChromaDB Vector Store
Track: Banking & FinTech (Cred)
Consolidated storage module for loan application transactional data and policy vector stores.
"""

from typing import List, Dict, Any, Optional
import os
import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import (
    LOAN_APPLICATIONS,
    LOAN_APPLICATIONS_BY_ID,
    get_loan_application,
    generate_loan_applications,
    LOAN_CATEGORIES,
    APPLICATION_STATUSES,
)
from rag.vector_store import (
    LocalVectorStore,
    LocalVectorStore as VectorStore,
    DeterministicEmbedder,
    get_vector_store,
)
from rag.chunking import (
    load_knowledge_base_documents,
    chunk_fixed_size,
    chunk_sentence_boundary,
)

__all__ = [
    "LOAN_APPLICATIONS",
    "LOAN_APPLICATIONS_BY_ID",
    "get_loan_application",
    "generate_synthetic_dataset",
    "LOAN_CATEGORIES",
    "APPLICATION_STATUSES",
    "LocalVectorStore",
    "VectorStore",
    "DeterministicEmbedder",
    "get_vector_store",
    "load_knowledge_base_documents",
    "chunk_fixed_size",
    "chunk_sentence_boundary",
]
