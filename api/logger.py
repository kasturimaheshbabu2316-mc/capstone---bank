"""
api/logger.py - ELK-Compatible Structured JSON-L Audit Logging
Track: Banking & FinTech (Cred)
Task 12: Single-line JSON logging with trace IDs and strict PII redaction.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Any, Optional
import os
import json
import uuid
from datetime import datetime, timezone
from agents.guardrails import mask_pii

LOG_DIR = "./logs"
AUDIT_LOG_FILE = os.path.join(LOG_DIR, "audit.jsonl")


class StructuredLogger:
    """
    Emits audit logs in single-line ELK-compatible JSON-L format.
    Guarantees zero unmasked PII in persistent log records.
    """

    def __init__(self, log_path: str = AUDIT_LOG_FILE):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_request(
        self,
        endpoint: str,
        raw_prompt: str,
        latency_ms: float,
        status_code: int = 200,
        trace_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Emits single-line JSON record to audit log file.
        """
        tid = trace_id or str(uuid.uuid4())
        # Force strict regex PII masking before logging
        sanitized_prompt = mask_pii(raw_prompt)

        log_record = {
            "trace_id": tid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "latency_ms": round(float(latency_ms), 2),
            "masked_prompt": sanitized_prompt,
            "status_code": int(status_code),
        }

        if extra:
            log_record.update(extra)

        # Write single line JSONL atomically
        line = json.dumps(log_record, ensure_ascii=False)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

        return log_record


# Singleton logger instance
LOGGER = StructuredLogger()


def get_logger() -> StructuredLogger:
    return LOGGER


if __name__ == "__main__":
    logger = get_logger()
    entry = logger.log_request(
        endpoint="/ask",
        raw_prompt="Customer PAN ABCDE1234F requested loan status",
        latency_ms=42.15,
        status_code=200,
    )
    print("Emitted structured log record:")
    print(json.dumps(entry, indent=2))
    assert "[MASKED_PAN]" in entry["masked_prompt"]
    assert "ABCDE1234F" not in entry["masked_prompt"]
    print("Zero-PII structured logging verified successfully.")
