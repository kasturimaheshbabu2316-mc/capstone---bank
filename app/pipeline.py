"""
app/pipeline.py - Multi-Agent Pipeline & Compliance Review Coordination
Track: Banking & FinTech (Cred)
Task 7 & Task 14: CrewAI 3-agent crew orchestration with AutoGen secondary peer review.
"""

from typing import Dict, Any, Optional, List
import os
import sys
from pathlib import Path

# Disable telemetry forcefully
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.crew import CredSupportCrew, get_support_crew, SUPPORT_CREW
from agents.mock_llm import MockLLM
from review.autogen_review import (
    AutoGenReviewTeam,
    get_review_team,
    VerdictModel,
)
from app.models import AgentResponseSchema


def run_pipeline(
    query: str, session_id: str = "default", enable_review: bool = True
) -> AgentResponseSchema:
    """
    Executes the multi-agent lending support pipeline:
    1. Primary CrewAI workflow (Retrieval Agent, Lookup Agent, Response Composer).
    2. Secondary AutoGen compliance peer review (PolicyComplianceReviewer, FinalEditor).
    3. Emits validated AgentResponseSchema.
    """
    crew = get_support_crew()
    response = crew.run_pipeline(query=query, session_id=session_id)

    if enable_review:
        review_team = get_review_team()
        verdict = review_team.review_draft(draft=response)
        # Apply approved or revised final answer
        response.direct_answer = verdict.final_answer

    return response


__all__ = [
    "CredSupportCrew",
    "get_support_crew",
    "SUPPORT_CREW",
    "MockLLM",
    "run_pipeline",
    "AutoGenReviewTeam",
    "get_review_team",
    "VerdictModel",
]
