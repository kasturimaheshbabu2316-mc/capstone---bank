"""
api/app.py - Production FastAPI Gateway
Track: Banking & FinTech (Cred)
Task 11: Endpoints for /ask, /add-document, and duplex /ws/chat.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, Any, Optional
import time
import uuid
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.schemas import AgentResponseSchema
from agents.guardrails import apply_input_guardrails, validate_groundedness
from agents.crew import get_support_crew
from review.autogen_review import get_review_team
from governance.budget_guard import enforce_budget_limits
from governance.cache import get_query_cache
from api.logger import get_logger
from rag.vector_store import get_vector_store
from rag.chunking import chunk_fixed_size, chunk_sentence_boundary


class AskRequest(BaseModel):
    query: str = Field(..., description="Customer inquiry or application reference")
    session_id: Optional[str] = Field(default="default_session", description="Session identifier for multi-turn memory")


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., description="Unique alphanumeric document key (e.g., doc_13_foreclosure)")
    content: str = Field(..., description="Authoritative policy document text (2-5 sentences)")
    topic: Optional[str] = Field(default="Policy Update", description="Policy subject title")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure knowledge base is indexed
    vs = get_vector_store()
    vs.index_all()
    yield
    # Shutdown cleanup if needed


app = FastAPI(
    title="Cred Domain Support Agent",
    description="Production-grade AI operations gateway for lending policy and application support.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/ask", response_model=AgentResponseSchema)
async def ask_endpoint(payload: AskRequest):
    """
    POST /ask: Primary structured conversational query endpoint.
    Pipeline: Budget Guard -> Input Guardrails -> Query Cache -> CrewAI -> AutoGen Review -> Output Guardrail -> Audit Logger.
    """
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4())
    logger = get_logger()
    cache = get_query_cache()
    crew = get_support_crew()
    reviewer = get_review_team()

    raw_query = payload.query
    session_id = payload.session_id or "default_session"

    # Step 1: Runtime Layer Budget Guard
    is_valid_size, size_msg, status_code = enforce_budget_limits(raw_query)
    if not is_valid_size:
        latency = (time.perf_counter() - start_time) * 1000.0
        logger.log_request("/ask", raw_query, latency, status_code=status_code, trace_id=trace_id)
        raise HTTPException(status_code=status_code, detail=size_msg)

    # Step 2: Input Guardrails (PII Masking & Injection Detection)
    guard_res = apply_input_guardrails(raw_query)
    if not guard_res["allowed"]:
        latency = (time.perf_counter() - start_time) * 1000.0
        logger.log_request("/ask", raw_query, latency, status_code=400, trace_id=trace_id)
        raise HTTPException(status_code=400, detail=guard_res["error"])

    sanitized_query = guard_res["masked_query"]

    # Step 3: Normalized Query Response Cache
    cached_resp = cache.get(sanitized_query)
    if cached_resp is not None:
        latency = (time.perf_counter() - start_time) * 1000.0
        logger.log_request(
            "/ask",
            sanitized_query,
            latency,
            status_code=200,
            trace_id=trace_id,
            extra={"cache_hit": True},
        )
        return cached_resp

    # Step 4: CrewAI Primary Multi-Agent Orchestration
    agent_draft = crew.run_pipeline(query=sanitized_query, session_id=session_id)

    # Step 5: AutoGen Secondary Peer Review
    verdict = reviewer.review_draft(agent_draft)
    final_answer = verdict.final_answer

    # Step 6: Output Guardrail Grounding Verification
    is_grounded, ground_msg = validate_groundedness(final_answer, agent_draft.citations)
    if not is_grounded:
        final_answer = "I don't know based on the provided policy documents."

    final_response = AgentResponseSchema(
        query=sanitized_query,
        intent=agent_draft.intent,
        direct_answer=final_answer,
        citations=agent_draft.citations,
        escalation_triggered=agent_draft.escalation_triggered,
        escalation_score=agent_draft.escalation_score,
    )

    # Cache response
    cache.set(sanitized_query, final_response)

    # Step 7: Structured JSON-L Logging
    latency = (time.perf_counter() - start_time) * 1000.0
    logger.log_request(
        "/ask",
        sanitized_query,
        latency,
        status_code=200,
        trace_id=trace_id,
        extra={"cache_hit": False, "approved": verdict.approved},
    )

    return final_response


@app.post("/add-document")
async def add_document_endpoint(payload: AddDocumentRequest):
    """
    POST /add-document: Dynamically ingests and indexes new policy documents into ChromaDB.
    """
    vs = get_vector_store()
    doc_id = payload.doc_id.strip()
    content = payload.content.strip()

    fixed_chunks = chunk_fixed_size(doc_id, content)
    sent_chunks = chunk_sentence_boundary(doc_id, content)

    vs.upsert_chunks(fixed_chunks, collection_type="fixed")
    vs.upsert_chunks(sent_chunks, collection_type="sentence")

    return {
        "status": "success",
        "doc_id": doc_id,
        "fixed_chunks_added": len(fixed_chunks),
        "sentence_chunks_added": len(sent_chunks),
        "message": f"Document '{doc_id}' successfully indexed into ChromaDB collections.",
    }


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    WebSocket /ws/chat: Real-time duplex conversational streaming endpoint.
    Handles multi-turn memory and catches WebSocketDisconnect cleanly without crashing server.
    """
    await websocket.accept()
    session_id = f"ws_session_{uuid.uuid4().hex[:8]}"
    crew = get_support_crew()
    logger = get_logger()

    try:
        while True:
            # Receive inbound frame
            user_message = await websocket.receive_text()
            start_time = time.perf_counter()

            # Budget check
            is_valid, msg, code = enforce_budget_limits(user_message)
            if not is_valid:
                await websocket.send_json({"error": msg, "status_code": code})
                continue

            # Input guardrails
            guard_res = apply_input_guardrails(user_message)
            if not guard_res["allowed"]:
                await websocket.send_json({"error": guard_res["error"], "status_code": 400})
                continue

            masked_query = guard_res["masked_query"]

            # Crew execution with session persistence
            response = crew.run_pipeline(query=masked_query, session_id=session_id)

            latency = (time.perf_counter() - start_time) * 1000.0
            logger.log_request("/ws/chat", masked_query, latency, status_code=200)

            # Send back structured frame
            await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        # Clean shutdown and resource freeing per Task 11 specification
        logger.log_request(
            "/ws/chat",
            f"Session {session_id} disconnected cleanly.",
            0.0,
            status_code=1000,
        )
    except Exception as e:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False)
