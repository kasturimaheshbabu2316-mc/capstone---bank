"""
app/main.py - Production FastAPI Application Gateway
Track: Banking & FinTech (Cred)
Task 11: Production endpoints for /ask, /add-document, and duplex /ws/chat.
"""

from typing import Dict, Any, Optional
import time
import uuid
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.models import AgentResponseSchema, AskRequest, AddDocumentRequest
from app.pipeline import run_pipeline, get_support_crew
from app.db import get_vector_store, chunk_fixed_size, chunk_sentence_boundary
from agents.guardrails import apply_input_guardrails, validate_groundedness
from governance.budget_guard import enforce_budget_limits
from governance.cache import get_query_cache
from api.logger import get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure ChromaDB store has indexed documents upon startup
    vs = get_vector_store()
    try:
        vs.index_all()
    except Exception:
        pass
    yield


app = FastAPI(
    title="Cred Domain Support Agent API",
    description="Multi-Agent Lending Operations Platform for Retail Credit Support",
    version="1.0.0",
    lifespan=lifespan,
)

logger = get_logger()
query_cache = get_query_cache()


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Basic health check and liveness probe."""
    return {"status": "ok", "service": "cred-domain-support-agent"}


@app.post("/ask", response_model=AgentResponseSchema)
def ask_question(payload: AskRequest) -> AgentResponseSchema:
    """
    Synchronous lending inquiry endpoint with end-to-end safety & governance.
    """
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    raw_query = payload.query.strip()
    session_id = payload.session_id or f"sess_{trace_id[:8]}"

    # Step 1: Runtime Layer Budget Guard (HTTP 413)
    allowed, budget_msg, status_code = enforce_budget_limits(raw_query)
    if not allowed:
        latency = (time.time() - start_time) * 1000
        logger.log_request("/ask", raw_query, latency, status_code=status_code, trace_id=trace_id)
        raise HTTPException(status_code=status_code, detail=budget_msg)

    # Step 2: Verification Layer - Input Guardrails (PII Masking & Injection Check)
    guardrail_res = apply_input_guardrails(raw_query)
    if guardrail_res.get("injection_detected"):
        latency = (time.time() - start_time) * 1000
        logger.log_request("/ask", raw_query, latency, status_code=400, trace_id=trace_id)
        raise HTTPException(status_code=400, detail="Security rejection: Adversarial prompt injection detected.")

    sanitized_query = guardrail_res.get("sanitized_query", raw_query)

    # Step 3: Optimization Layer - Normalized Query Response Cache
    cached_response = query_cache.get(sanitized_query)
    if cached_response is not None:
        latency = (time.time() - start_time) * 1000
        logger.log_request(
            "/ask",
            sanitized_query,
            latency,
            status_code=200,
            trace_id=trace_id,
            cache_hit=True,
        )
        return cached_response

    # Step 4: Multi-Agent Execution & Peer Review Pipeline
    response = run_pipeline(query=sanitized_query, session_id=session_id)

    # Step 5: Verification Layer - Output Grounding Guardrail
    if response.intent == "Policy inquiry" and response.direct_answer != "I don't know based on the provided policy documents.":
        vs = get_vector_store()
        retrieved = vs.query(sanitized_query, collection_type="sentence", top_k=3)
        context_corpus = " ".join(r["text"] for r in retrieved)
        if not validate_groundedness(response.direct_answer, context_corpus):
            response.direct_answer = "I don't know based on the provided policy documents."
            response.citations = []

    # Step 6: Store in Normalized Cache & Emit Audit Log
    query_cache.set(sanitized_query, response)
    latency = (time.time() - start_time) * 1000
    logger.log_request(
        "/ask",
        sanitized_query,
        latency,
        status_code=200,
        trace_id=trace_id,
        cache_hit=False,
    )

    return response


@app.post("/add-document")
def add_document(payload: AddDocumentRequest) -> Dict[str, Any]:
    """
    Ingests and indexes a new policy document into ChromaDB collections.
    """
    start_time = time.time()
    vs = get_vector_store()

    # Generate fixed and sentence chunks
    fixed_chunks = chunk_fixed_size(payload.doc_id, payload.content)
    sent_chunks = chunk_sentence_boundary(payload.doc_id, payload.content)

    # Upsert into collections
    vs.upsert_chunks(fixed_chunks, collection_type="fixed")
    vs.upsert_chunks(sent_chunks, collection_type="sentence")

    latency = (time.time() - start_time) * 1000
    logger.log_request("/add-document", f"Indexed document {payload.doc_id}", latency, status_code=200)

    return {
        "status": "indexed",
        "doc_id": payload.doc_id,
        "fixed_chunks": len(fixed_chunks),
        "sentence_chunks": len(sent_chunks),
    }


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    Duplex streaming conversational endpoint with graceful disconnect handling.
    """
    await websocket.accept()
    session_id = f"ws_{uuid.uuid4().hex[:8]}"

    try:
        while True:
            data = await websocket.receive_text()
            if not data.strip():
                continue

            # Process inbound message through pipeline
            guardrail_res = apply_input_guardrails(data)
            if guardrail_res.get("injection_detected"):
                await websocket.send_json({
                    "error": "Adversarial injection rejected.",
                    "status_code": 400,
                })
                continue

            clean_query = guardrail_res.get("sanitized_query", data)
            response = run_pipeline(query=clean_query, session_id=session_id)
            await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        logger.log_request(
            "/ws/chat",
            f"Session {session_id} disconnected cleanly.",
            0.0,
            status_code=1000,
        )
    except Exception:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
