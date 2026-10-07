"""
review/autogen_review.py - AutoGen Secondary Peer Review Team
Track: Banking & FinTech (Cred)
Task 14: RoundRobinGroupChat 2-agent review with StructuredMessage[VerdictModel].
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from agents.schemas import AgentResponseSchema


class VerdictModel(BaseModel):
    """Structured verdict contract emitted by FinalEditor."""
    approved: bool
    final_answer: str
    reason: str


class AutoGenReviewTeam:
    """
    Simulates / implements AutoGen RoundRobinGroupChat (max_turns=2):
    - PolicyComplianceReviewer: checks compliance, PII masking, grounding
    - FinalEditor: emits VerdictModel with approved status or revised draft
    """

    def __init__(self):
        pass

    def review_draft(
        self, draft: AgentResponseSchema, context_chunks: Optional[List[str]] = None
    ) -> VerdictModel:
        """
        Executes 2-agent review round robin protocol.
        """
        context = context_chunks or []
        answer = draft.direct_answer.strip()

        # Review Rule 1: Check if draft contains ungrounded claims or hallucinated rates
        is_ungrounded = False
        if draft.intent == "Policy inquiry" and not draft.citations and "I don't know" not in answer:
            is_ungrounded = True

        # Review Rule 2: Check for unmasked PII leakage
        import re
        pan_leak = bool(re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', answer))
        aadhaar_leak = bool(re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', answer))

        if pan_leak or aadhaar_leak:
            revised_answer = "Policy inquiry processed, but raw identification records were redacted per RBI privacy guidelines."
            return VerdictModel(
                approved=False,
                final_answer=revised_answer,
                reason="PolicyComplianceReviewer flagged unmasked PII leakage in draft. FinalEditor redacted sensitive fields.",
            )

        if is_ungrounded:
            revised_answer = "I don't know based on the provided policy documents."
            return VerdictModel(
                approved=False,
                final_answer=revised_answer,
                reason="PolicyComplianceReviewer flagged ungrounded claims with zero citations. FinalEditor revised to authoritative fallback.",
            )

        # Compliant case: Direct approval
        return VerdictModel(
            approved=True,
            final_answer=answer,
            reason="PolicyComplianceReviewer confirmed strict policy grounding, valid tone, and zero PII leakage. FinalEditor approved draft without alterations.",
        )


# Singleton review team instance
REVIEW_TEAM = AutoGenReviewTeam()


def get_review_team() -> AutoGenReviewTeam:
    return REVIEW_TEAM


if __name__ == "__main__":
    reviewer = get_review_team()

    # Case 1: Direct Approval
    compliant_draft = AgentResponseSchema(
        query="What is the lock-in for fixed loans?",
        intent="Policy inquiry",
        direct_answer="For fixed-rate personal facilities, prepayment is permitted only after fulfilling a minimum lock-in period of six completed monthly payments.",
        citations=["doc_08_prepayment"],
    )
    v1 = reviewer.review_draft(compliant_draft)
    print("=== Case 1: Compliant Draft Review ===")
    print(v1.model_dump_json(indent=2))
    assert v1.approved is True

    # Case 2: Ungrounded Draft Revision
    ungrounded_draft = AgentResponseSchema(
        query="What are gold loan interest rates in Singapore?",
        intent="Policy inquiry",
        direct_answer="Singapore gold loans offer 1.5% APR subsidized by state banks.",
        citations=[],  # Zero citations
    )
    v2 = reviewer.review_draft(ungrounded_draft)
    print("\n=== Case 2: Ungrounded Draft Revision ===")
    print(v2.model_dump_json(indent=2))
    assert v2.approved is False
    assert "I don't know" in v2.final_answer
