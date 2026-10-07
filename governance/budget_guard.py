"""
governance/budget_guard.py - AI Governance & Runtime Budget Limiter
Track: Banking & FinTech (Cred)
Task 15: 4-Layer AI Governance Framework & Runtime Budget Guard.
"""

from typing import Tuple, Dict, Any

# =============================================================================
# SYSTEM RISK CLASSIFICATION RATIONALE (Task 15 Documented Requirement)
#
# Classification: HIGH RISK
# Justification: The Cred Domain Support Agent operates within retail banking and
# consumer lending operations. Automated outputs directly influence customer
# understanding of loan eligibility, credit-score impacts, fee structures, and
# application escalation statuses. Inaccurate or ungrounded outputs risk severe
# financial harm, regulatory non-compliance under RBI fair practices guidelines,
# and customer dispute exposure. Strict 4-layer governance controls are mandatory.
# =============================================================================

MAX_CHARACTER_LIMIT = 2048
MAX_ESTIMATED_TOKENS = 512


def enforce_budget_limits(query: str) -> Tuple[bool, str, int]:
    """
    Runtime Layer Budget Guard:
    Enforces maximum character and token budget ceilings per request.
    Returns: (allowed: bool, reason: str, status_code: int)
    """
    if not query:
        return False, "Query cannot be empty.", 400

    char_count = len(query)
    if char_count > MAX_CHARACTER_LIMIT:
        return (
            False,
            f"Payload entity too large: Query length ({char_count} chars) exceeds maximum threshold ({MAX_CHARACTER_LIMIT} chars).",
            413,
        )

    # Fast token estimation (~4 characters per token)
    estimated_tokens = len(query.split()) * 2
    if estimated_tokens > MAX_ESTIMATED_TOKENS:
        return (
            False,
            f"Payload entity too large: Estimated tokens ({estimated_tokens}) exceeds maximum token budget ({MAX_ESTIMATED_TOKENS}).",
            413,
        )

    return True, "Within budget limits.", 200


def verify_least_autonomy_access(agent_role: str, tool_name: str) -> bool:
    """
    Application Layer Governance:
    Mechanically blocks unauthorized agent tool invocations under RBAC least autonomy.
    """
    role = agent_role.lower()
    tool = tool_name.lower()

    if tool == "check_loan_application_status":
        return "lookup" in role or "operations" in role

    if tool == "rag_lookup":
        return "retrieval" in role or "policy" in role

    return False


if __name__ == "__main__":
    # Test valid query
    ok, msg, code = enforce_budget_limits("What is the interest rate slab?")
    print(f"Normal query test: OK={ok}, Code={code}")
    assert ok is True and code == 200

    # Test oversized query exceeding budget
    huge_query = "A" * 3000
    ok, msg, code = enforce_budget_limits(huge_query)
    print(f"Oversized query test: OK={ok}, Code={code}, Msg='{msg}'")
    assert ok is False and code == 413

    # Test least privilege RBAC
    assert verify_least_autonomy_access("Lookup Agent", "check_loan_application_status") is True
    assert verify_least_autonomy_access("Retrieval Agent", "check_loan_application_status") is False
    print("AI Governance Budget Guard & RBAC verified successfully.")
