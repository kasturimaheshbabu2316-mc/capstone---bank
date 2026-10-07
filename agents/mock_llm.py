"""
agents/mock_llm.py - Deterministic Offline BaseLLM Implementation
Track: Banking & FinTech (Cred)
Task 7: CrewAI Custom LLM Extension with ReAct Template & Tool Routing Pitfall Mitigations.
"""

from typing import Any, List, Optional, Dict
import re

# BaseLLM import fallback for air-gapped environments
try:
    from crewai.llms.base_llm import BaseLLM  # type: ignore
except Exception:
    class BaseLLM:  # type: ignore
        """Lightweight stand-in if CrewAI BaseLLM cannot be loaded."""
        def __init__(self, *args, **kwargs):
            pass


class MockLLM(BaseLLM):
    """
    Deterministic offline LLM subclass.
    Zero external network calls.
    Addresses Pitfall 1 (ReAct Template Collision) and Pitfall 2 (Tool Routing Collision).
    """

    def __init__(self, model_name: str = "mock-deterministic-cred-llm", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def call(
        self,
        messages: Any,
        tools: Optional[List[Dict[str, Any]]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> str:
        """
        Main inference entrypoint called by CrewAI agents during ReAct reasoning loops.
        """
        # Convert messages / prompt into plain string
        prompt_str = ""
        if isinstance(messages, str):
            prompt_str = messages
        elif isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    prompt_str += m.get("content", "") + "\n"
                elif hasattr(m, "content"):
                    prompt_str += str(m.content) + "\n"
                else:
                    prompt_str += str(m) + "\n"
        else:
            prompt_str = str(messages)

        return self.generate_response(prompt_str)

    def generate_response(self, prompt: str) -> str:
        """
        Generates deterministic ReAct outputs or final answers based on prompt content.
        Safeguards against Pitfall 1 and Pitfall 2.
        """
        # =========================================================================
        # Pitfall 1 Mitigation: ReAct Template Collision
        # Slices off system prompt so template text "Observation: the result of the action"
        # does not cause premature pattern matching.
        # =========================================================================
        system_marker = "Observation: the result of the action"
        working_prompt = prompt
        if system_marker in prompt:
            # Inspect only the text after the system prompt definition
            parts = prompt.split(system_marker, 1)
            working_prompt = parts[1] if len(parts) > 1 else prompt

        # Check if an observation has already been provided in the conversation
        has_recent_observation = "Observation:" in working_prompt

        # Application ID pattern (e.g., APP-1001)
        app_id_match = re.search(r'\b(APP-\d{4})\b', prompt, re.IGNORECASE)

        # =========================================================================
        # Scenario A: Inbound Application Status Lookup
        # =========================================================================
        if app_id_match:
            record_id = app_id_match.group(1).upper()

            if not has_recent_observation:
                # =====================================================================
                # Pitfall 2 Mitigation: Exact Tool Routing Identity
                # Strictly evaluate exact declared tool name: 'check_loan_application_status'
                # DO NOT use fuzzy substring matching on 'lookup'
                # =====================================================================
                return (
                    f"Thought: The user is requesting the status of loan application {record_id}. "
                    f"I need to query the application status tool.\n"
                    f"Action: check_loan_application_status\n"
                    f"Action Input: {{\"record_id\": \"{record_id}\"}}"
                )
            else:
                # Synthesize final response from observation
                return (
                    f"Thought: I have retrieved the application details for {record_id}.\n"
                    f"Final Answer: Loan application {record_id} status has been verified. "
                    f"Please refer to the application record for current approval stage and escalation metrics."
                )

        # =========================================================================
        # Scenario B: Policy Inquiries (RAG Flow)
        # =========================================================================
        if not has_recent_observation:
            # Emit exact RAG tool call
            return (
                "Thought: The user inquiry is a lending policy inquiry. "
                "I must retrieve grounded context from the authoritative knowledge base.\n"
                "Action: rag_lookup\n"
                "Action Input: {\"query\": \"policy inquiry\"}"
            )
        else:
            # Observation present - synthesize final grounded answer
            return (
                "Thought: I have reviewed the retrieved policy documentation.\n"
                "Final Answer: According to Cred lending policy documents, loan eligibility, "
                "repayment schedules, interest rates, and fee structures are governed by standard banking criteria."
            )

    # CrewAI / LangChain compatibility methods
    def predict(self, text: str, **kwargs) -> str:
        return self.generate_response(text)

    def invoke(self, input_val: Any, **kwargs) -> Any:
        text = input_val if isinstance(input_val, str) else str(input_val)
        return self.generate_response(text)


if __name__ == "__main__":
    llm = MockLLM()
    # Test Pitfall 1 mitigation with literal template text
    react_prompt = "System: Format your answer as Action: ... Observation: the result of the action\nUser: Check status of APP-1005"
    resp = llm.generate_response(react_prompt)
    print("Test Response:")
    print(resp)
    assert "check_loan_application_status" in resp
    assert "rag_lookup" not in resp  # Exact match verification
    print("MockLLM Pitfall safeguards verified successfully.")
