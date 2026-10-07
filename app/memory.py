"""
app/memory.py - Multi-Turn Conversational Session Memory
Track: Banking & FinTech (Cred)
Task 8: Session isolation, pronominal reference continuity, and clean state reset.
"""

from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.memory import (
    InMemoryChatMessageHistory,
    SessionMemoryManager,
    SESSION_MEMORY,
    get_session_memory,
)

__all__ = [
    "InMemoryChatMessageHistory",
    "SessionMemoryManager",
    "SESSION_MEMORY",
    "get_session_memory",
]
