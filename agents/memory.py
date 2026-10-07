"""
agents/memory.py - Multi-Turn Conversational Session Memory
Track: Banking & FinTech (Cred)
Task 8: In-Memory Multi-Turn Session Memory with LangChain Primitives.
"""

from typing import Dict, List, Any, Optional

try:
    from langchain_core.chat_history import InMemoryChatMessageHistory  # type: ignore
    from langchain_core.messages import HumanMessage, AIMessage  # type: ignore
except Exception:
    # Standalone implementation conforming to LangChain memory interface
    class InMemoryChatMessageHistory:  # type: ignore
        def __init__(self):
            self.messages: List[Dict[str, str]] = []

        def add_user_message(self, message: str):
            self.messages.append({"role": "user", "content": message})

        def add_ai_message(self, message: str):
            self.messages.append({"role": "ai", "content": message})

        def clear(self):
            self.messages.clear()


class SessionMemoryManager:
    """
    Manages session-isolated conversational memory buffers.
    """

    def __init__(self):
        self._sessions: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session(self, session_id: str) -> InMemoryChatMessageHistory:
        """Retrieves or initializes an isolated chat history buffer."""
        clean_id = session_id.strip() if session_id else "default_session"
        if clean_id not in self._sessions:
            self._sessions[clean_id] = InMemoryChatMessageHistory()
        return self._sessions[clean_id]

    def record_turn(self, session_id: str, user_query: str, agent_response: str):
        """Records a completed turn in session memory."""
        history = self.get_session(session_id)
        if hasattr(history, "add_user_message"):
            history.add_user_message(user_query)
            history.add_ai_message(agent_response)
        elif hasattr(history, "add_message"):
            from langchain_core.messages import HumanMessage, AIMessage  # type: ignore
            history.add_message(HumanMessage(content=user_query))
            history.add_message(AIMessage(content=agent_response))

    def get_history_summary(self, session_id: str) -> str:
        """Formats session history into an unambiguous string context."""
        history = self.get_session(session_id)
        formatted: List[str] = []

        if hasattr(history, "messages"):
            for m in history.messages:
                if isinstance(m, dict):
                    role = m.get("role", "speaker").title()
                    content = m.get("content", "")
                    formatted.append(f"{role}: {content}")
                elif hasattr(m, "content"):
                    role = "User" if "Human" in type(m).__name__ else "Assistant"
                    formatted.append(f"{role}: {m.content}")

        return "\n".join(formatted)

    def reset_session(self, session_id: str):
        """Clears memory for a specific session."""
        clean_id = session_id.strip()
        if clean_id in self._sessions:
            self._sessions[clean_id].clear()
            del self._sessions[clean_id]


# Singleton session memory manager
SESSION_MEMORY = SessionMemoryManager()


def get_session_memory() -> SessionMemoryManager:
    return SESSION_MEMORY


if __name__ == "__main__":
    mem = get_session_memory()
    # Test session continuity
    mem.record_turn("sess-1", "Check status of APP-1002", "APP-1002 is Under Review.")
    mem.record_turn("sess-1", "What is its loan amount?", "The loan amount is ₹350,000.")
    print("Session 1 History:\n", mem.get_history_summary("sess-1"))

    # Test session isolation
    print("\nSession 2 History (Should be empty):\n", mem.get_history_summary("sess-2"))
    assert len(mem.get_history_summary("sess-2")) == 0
    print("Session memory isolation verified successfully.")
