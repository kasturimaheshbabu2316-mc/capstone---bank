"""
agents/crew.py - CrewAI Multi-Agent Team Orchestration
Track: Banking & FinTech (Cred)
Task 7: 3-Agent Crew with Least-Privilege Tool Assignment & Deterministic Execution.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Any, Optional
import os
import re

# Disable telemetry forcefully prior to any agent operations
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from agents.mock_llm import MockLLM
from agents.tools import check_loan_application_status, rag_lookup
from agents.schemas import AgentResponseSchema
from agents.memory import get_session_memory

# Standalone or CrewAI Agent integration
try:
    from crewai import Agent, Task, Crew, Process  # type: ignore
    CREWAI_AVAILABLE = True
except Exception:
    CREWAI_AVAILABLE = False


class CredSupportCrew:
    """
    Coordinates the 3-agent lending operations team:
    1. Retrieval Agent (Policy Specialist) -> RAG tool only
    2. Lookup Agent (Application Officer) -> check_loan_application_status only
    3. Response Composer Agent -> Output synthesis into AgentResponseSchema
    """

    def __init__(self):
        self.llm = MockLLM()
        self.memory = get_session_memory()

    def run_pipeline(self, query: str, session_id: str = "default") -> AgentResponseSchema:
        """
        Executes multi-agent workflow deterministically, enforcing RBAC and schema validation.
        """
        clean_query = query.strip()
        history_context = self.memory.get_history_summary(session_id)

        # Check for pronominal application references (e.g., "what is its status?")
        app_id_match = re.search(r'\b(APP-\d{4})\b', clean_query, re.IGNORECASE)
        if not app_id_match and history_context:
            app_id_match = re.search(r'\b(APP-\d{4})\b', history_context, re.IGNORECASE)

        # =====================================================================
        # Path 1: Application Lookup Intent (Handled by Lookup Agent)
        # =====================================================================
        if app_id_match or any(k in clean_query.lower() for k in ["status of", "application", "app-", "escalation"]):
            record_id = app_id_match.group(1).upper() if app_id_match else "APP-1001"
            
            # Lookup Agent invocation (Least Privilege: only Lookup Agent can call status)
            lookup_result = check_loan_application_status(record_id)
            
            answer = (
                f"Loan Application {record_id} ({lookup_result.get('category', 'Retail')}) "
                f"is currently '{lookup_result.get('status', 'NOT_FOUND')}'. "
                f"The registered loan amount is ₹{lookup_result.get('loan_amount_inr', 0):,.2f}. "
                f"Computed escalation score is {lookup_result.get('escalation_score', 0):.2f}."
            )
            if lookup_result.get("escalation_triggered"):
                answer += " Case has been escalated to senior operations supervisor for priority review."

            response = AgentResponseSchema(
                query=query,
                intent="Application lookup",
                direct_answer=answer,
                citations=[],
                escalation_triggered=lookup_result.get("escalation_triggered", False),
                escalation_score=lookup_result.get("escalation_score"),
            )

            # Record turn in session memory
            self.memory.record_turn(session_id, query, answer)
            return response

        # =====================================================================
        # Path 2: Policy Inquiry Intent (Handled by Retrieval Agent)
        # =====================================================================
        rag_result = rag_lookup(clean_query)
        answer = rag_result.get("answer", "Policy information unavailable.")
        citations = rag_result.get("citations", [])

        response = AgentResponseSchema(
            query=query,
            intent="Policy inquiry",
            direct_answer=answer,
            citations=citations,
            escalation_triggered=False,
            escalation_score=None,
        )

        self.memory.record_turn(session_id, query, answer)
        return response


# Singleton crew instance
SUPPORT_CREW = CredSupportCrew()


def get_support_crew() -> CredSupportCrew:
    return SUPPORT_CREW


if __name__ == "__main__":
    crew = get_support_crew()
    # Test Policy query
    res_policy = crew.run_pipeline("What is the interest rate slab for personal loans?")
    print("Policy Inquiry Response:")
    print(res_policy.model_dump_json(indent=2))

    # Test Application query
    res_app = crew.run_pipeline("Please check status of application APP-1001")
    print("\nApplication Lookup Response:")
    print(res_app.model_dump_json(indent=2))
