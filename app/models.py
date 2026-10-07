"""
app/models.py - Pydantic Contracts and Data Schemas
Track: Banking & FinTech (Cred)
Consolidated models for API requests, agent responses, compliance verdicts, and database records.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentResponseSchema(BaseModel):
    """
    Standardized, validated output contract for Cred Domain Support Agent responses.
    Task 9 Specification.
    """
    query: str = Field(description="Original user inquiry")
    intent: str = Field(description="Policy inquiry or Application lookup")
    direct_answer: str = Field(description="Authoritative, grounded customer-facing answer")
    citations: List[str] = Field(default_factory=list, description="List of source policy document identifiers")
    escalation_triggered: bool = Field(default=False, description="Whether inquiry was escalated to supervisor")
    escalation_score: Optional[float] = Field(default=None, description="Continuous escalation risk score in [0.0, 1.0]")


class VerdictModel(BaseModel):
    """
    Structured compliance verdict emitted by AutoGen FinalEditor.
    Task 14 Specification.
    """
    approved: bool = Field(description="Compliance approval flag")
    final_answer: str = Field(description="Authoritative customer-facing answer")
    reason: str = Field(description="Audit explanation")


class AskRequest(BaseModel):
    """Input payload contract for /ask endpoint."""
    query: str = Field(description="Customer query text", min_length=1)
    session_id: Optional[str] = Field(default=None, description="Optional conversational session ID")


class AddDocumentRequest(BaseModel):
    """Input payload contract for /add-document endpoint."""
    doc_id: str = Field(description="Unique document identifier")
    content: str = Field(description="Policy text content", min_length=1)
    topic: Optional[str] = Field(default="Policy", description="Topic category")


class LoanApplicationRecord:
    """Represents a single immutable customer loan application record."""

    def __init__(
        self,
        record_id: str,
        category: str,
        status: str,
        loan_amount_inr: float,
        days_since_created: int,
        flagged_for_fraud_review: bool,
    ):
        self.record_id = record_id
        self.category = category
        self.status = status
        self.loan_amount_inr = round(float(loan_amount_inr), 2)
        self.days_since_created = int(days_since_created)
        self.flagged_for_fraud_review = bool(flagged_for_fraud_review)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "category": self.category,
            "status": self.status,
            "loan_amount_inr": self.loan_amount_inr,
            "days_since_created": self.days_since_created,
            "flagged_for_fraud_review": self.flagged_for_fraud_review,
        }
