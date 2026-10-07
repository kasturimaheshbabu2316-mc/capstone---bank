"""
agents/tools.py - Application Status Tool, Escalation Engine, and RAG Tool
Track: Banking & FinTech (Cred)
Task 6: Application Status Tool & Continuous Escalation Scoring.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Any, Optional
from dataset import get_loan_application, LOAN_APPLICATIONS
from rag.evaluation import query_with_fallback

W_FRAUD = 0.65
W_RECENCY = 0.35
ESCALATION_THRESHOLD = 0.70


def calculate_escalation_score(
    flagged_for_fraud_review: bool, days_since_created: int
) -> float:
    """
    Computes continuous escalation score S in [0.0, 1.0].
    S = w_fraud * I(fraud) + w_recency * (days_since_created / 30)
    """
    fraud_indicator = 1.0 if flagged_for_fraud_review else 0.0
    recency_ratio = min(max(float(days_since_created) / 30.0, 0.0), 1.0)
    score = (W_FRAUD * fraud_indicator) + (W_RECENCY * recency_ratio)
    return round(float(score), 4)


def check_loan_application_status(record_id: str) -> Dict[str, Any]:
    """
    Task 6 Tool: Retrieves loan application record, status, amount, and computes escalation score.
    Returns:
      {
        "record_id": str,
        "status": str,
        "loan_amount_inr": float,
        "escalation_score": float,
        "escalation_triggered": bool,
        "message": str
      }
    """
    rec = get_loan_application(record_id)
    if not rec:
        return {
            "record_id": record_id,
            "status": "NOT_FOUND",
            "loan_amount_inr": 0.0,
            "escalation_score": 0.0,
            "escalation_triggered": False,
            "message": f"Loan application record '{record_id}' not found in active records database.",
        }

    score = calculate_escalation_score(
        flagged_for_fraud_review=rec["flagged_for_fraud_review"],
        days_since_created=rec["days_since_created"],
    )
    escalated = score >= ESCALATION_THRESHOLD

    return {
        "record_id": rec["record_id"],
        "category": rec["category"],
        "status": rec["status"],
        "loan_amount_inr": rec["loan_amount_inr"],
        "days_since_created": rec["days_since_created"],
        "flagged_for_fraud_review": rec["flagged_for_fraud_review"],
        "escalation_score": score,
        "escalation_triggered": escalated,
        "message": (
            f"Application {rec['record_id']} for {rec['category']} (₹{rec['loan_amount_inr']:,.2f}) "
            f"is currently '{rec['status']}'. Escalation score is {score:.2f}."
            + (" [PRIORITY SUPERVISOR ESCALATION TRIGGERED]" if escalated else "")
        ),
    }


def rag_lookup(query: str, tau: float = 0.20) -> Dict[str, Any]:
    """
    Retrieves grounded policy context from local ChromaDB store.
    """
    return query_with_fallback(query_text=query, tau=tau)


if __name__ == "__main__":
    print("Testing check_loan_application_status:")
    sample = LOAN_APPLICATIONS[0]
    out = check_loan_application_status(sample["record_id"])
    print("Result for", sample["record_id"], ":", out)

    # Demonstrate high escalation score calculation
    high_score = calculate_escalation_score(flagged_for_fraud_review=True, days_since_created=25)
    print(f"High risk sample score (fraud=True, days=25): {high_score} (Escalated: {high_score >= ESCALATION_THRESHOLD})")
