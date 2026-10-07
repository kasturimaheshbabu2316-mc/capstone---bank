"""
app/tools.py - Application Status Tool, Continuous Escalation Engine, and RAG Tool
Track: Banking & FinTech (Cred)
Task 6: Loan status lookup, continuous risk scoring, and knowledge base retrieval.
"""

from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.tools import (
    calculate_escalation_score,
    check_loan_application_status,
    rag_lookup,
    W_FRAUD,
    W_RECENCY,
    ESCALATION_THRESHOLD,
)

__all__ = [
    "calculate_escalation_score",
    "check_loan_application_status",
    "rag_lookup",
    "W_FRAUD",
    "W_RECENCY",
    "ESCALATION_THRESHOLD",
]
