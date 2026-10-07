"""
agents/schemas.py - Pydantic Response Models
Track: Banking & FinTech (Cred)
Task 9: Structured Output Validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class AgentResponseSchema(BaseModel):
    """
    Standardized, validated output contract for Cred Domain Support Agent responses.
    """
    query: str = Field(description="Original user inquiry")
    intent: str = Field(description="Policy inquiry or Application lookup")
    direct_answer: str = Field(description="Authoritative, grounded customer-facing answer")
    citations: List[str] = Field(default_factory=list, description="List of source policy document identifiers")
    escalation_triggered: bool = Field(default=False, description="Whether inquiry was escalated to supervisor")
    escalation_score: Optional[float] = Field(default=None, description="Continuous escalation risk score in [0.0, 1.0]")
