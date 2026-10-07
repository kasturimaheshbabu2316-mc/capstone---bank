# Engineering Implementation Plan & Delivery Roadmap

**Project:** Cred Domain Support Agent (Lending Operations)  
**Track:** Banking & FinTech (Cred)  
**Document Version:** 1.0.0  
**Status:** Completed & Validated  
**Reference:** [ARCHITECTURE.md](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/doc/ARCHITECTURE.md) | [app.md](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/doc/app.md)

---

## 1. Executive Implementation Strategy

The implementation plan is structured around a phased, test-driven engineering methodology ensuring:
1. **Complete Air-Gapped Execution:** Zero reliance on live external APIs; deterministic offline execution across all agent layers.
2. **Defensive Governance:** Ingress budget limiting, pre-flight PII redaction, post-flight grounding verification, and secondary peer review.
3. **Mathematical Invariant Verification:** Seeded datasets, empirical threshold calibration ($\tau = 0.25$), continuous escalation scoring, and 4-dimensional quantitative evaluation.

---

## 2. Phased Delivery Roadmap & Milestones

```mermaid
gantt
    title Cred Domain Support Agent - Implementation Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1: Data & Knowledge Base
    Task 1: Seeded Dataset Generator           :done, p1_1, 2026-10-01, 1d
    Task 2: Authoritative Policy Corpus        :done, p1_2, 2026-10-01, 1d
    section Phase 2: RAG & Threshold Calibration
    Task 3: Dual Chunking Strategies           :done, p2_1, 2026-10-02, 1d
    Task 4: Vector Indexing & tau Calibration  :done, p2_2, 2026-10-02, 1d
    Task 5: Precision & Recall Evaluation      :done, p2_3, 2026-10-02, 1d
    section Phase 3: CrewAI Multi-Agent Core
    Task 6: Application Status Tool & Score S  :done, p3_1, 2026-10-03, 1d
    Task 7: Deterministic MockLLM & 3-Agent Crew :done, p3_2, 2026-10-03, 1d
    Task 8: Multi-Turn Conversational Memory   :done, p3_3, 2026-10-03, 1d
    Task 9: Pydantic Response Schema Contract  :done, p3_4, 2026-10-04, 1d
    Task 10: Dual-Stage Safety Guardrails      :done, p3_5, 2026-10-04, 1d
    section Phase 4: Production API & Logging
    Task 11: FastAPI Endpoints & WebSocket     :done, p4_1, 2026-10-05, 1d
    Task 12: ELK JSON-L Structured Logger      :done, p4_2, 2026-10-05, 1d
    section Phase 5: Evaluation & Transcripts
    Task 13: 15-Query Benchmark & Transcripts  :done, p5_1, 2026-10-06, 1d
    section Phase 6: Secondary AutoGen Review
    Task 14: AutoGen Review Team & VerdictModel:done, p6_1, 2026-10-06, 1d
    section Phase 7: AI Governance & Optimization
    Task 15: Budget Guard & High-Risk Tiering  :done, p7_1, 2026-10-07, 1d
    Task 16: Normalized Query Cache            :done, p7_2, 2026-10-07, 1d
```

---

## 3. Detailed Phase Breakdown & Deliverables

### Phase 1: Data Architecture & Knowledge Base (Tasks 1–2)
* **Objective:** Establish authoritative ground-truth datasets and policy texts.
* **Deliverables:**
  * [`dataset.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/dataset.py): Seeded generator (`seed=42`) producing 45 records. Invariants verified:
    * Total records $N = 45 \ge 40$.
    * 5 categories, each $\ge 3$ records.
    * 5 statuses, each $\ge 1$ record.
    * Fraud rate: $\frac{6}{45} = 13.3\% \in [10\%, 30\%]$.
  * [`knowledge_base/`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/knowledge_base): 12 distinct policy documents (`doc_01_eligibility.txt` through `doc_12_nri_account.txt`), 2–5 sentences each.
* **Verification:** `python dataset.py` validates all invariants at startup.

---

### Phase 2: RAG Pipeline, Embedding & Threshold Calibration (Tasks 3–5)
* **Objective:** Implement dual chunking, ChromaDB vector indexing, empirical threshold calibration ($\tau$), and strategy benchmarking.
* **Deliverables:**
  * [`rag/chunking.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/rag/chunking.py): Fixed-size splitter (200 chars / 40 overlap) and sentence-boundary splitter.
  * [`rag/vector_store.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/rag/vector_store.py): Local ChromaDB collections (`collection_fixed` and `collection_sentence`), deterministic semantic embedder, idempotent upserting.
  * [`rag/evaluation.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/rag/evaluation.py): Calibration harness determining $\tau = 0.25$, fallback generator, and document-level precision/recall evaluation.
* **Empirical Outcome:** Strategy B (Sentence) demonstrated superior precision (0.30 vs 0.20) and recall (0.70 vs 0.50) over Strategy A, confirming the production deployment recommendation.

---

### Phase 3: Tooling, Deterministic Mock LLM & Multi-Agent Core (Tasks 6–10)
* **Objective:** Build multi-agent orchestration, continuous escalation scoring, conversational memory, and security guardrails.
* **Deliverables:**
  * [`agents/tools.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/tools.py): `check_loan_application_status` with continuous escalation scoring:
    $$S = 0.65 \cdot \mathbb{I}(\text{fraud}) + 0.35 \cdot (\text{days} / 30), \quad \text{threshold } \theta = 0.70$$
  * [`agents/mock_llm.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/mock_llm.py): Deterministic BaseLLM implementation resolving ReAct template collisions and exact tool routing.
  * [`agents/crew.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/crew.py): 3-agent CrewAI pipeline (Retrieval Agent, Lookup Agent, Response Composer) enforcing strict least-privilege tool isolation.
  * [`agents/memory.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/memory.py): LangChain `InMemoryChatMessageHistory` session manager.
  * [`agents/schemas.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/schemas.py): Pydantic v2 `AgentResponseSchema`.
  * [`agents/guardrails.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/guardrails.py): Strict regex PII masking (`[MASKED_PAN]`, `[MASKED_AADHAAR]`, `[MASKED_ACCOUNT]`), adversarial injection rejection, factual grounding verification.

---

### Phase 4: Production API, WebSocket & Structured Audit Logging (Tasks 11–12)
* **Objective:** Expose high-performance REST and WebSocket interfaces with enterprise observability.
* **Deliverables:**
  * [`api/app.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/api/app.py): FastAPI service with `POST /ask`, `POST /add-document`, and `WebSocket /ws/chat` catching `WebSocketDisconnect` cleanly.
  * [`api/logger.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/api/logger.py): Single-line ELK-compatible JSON-L audit logger writing to `logs/audit.jsonl` with UUID4 trace IDs and zero raw PII.

---

### Phase 5: Quantitative Evaluation & Verifiable Transcripts (Task 13)
* **Objective:** Execute 15-query benchmark scoring Accuracy, Grounding, Completeness, and Safety.
* **Deliverables:**
  * [`evaluation/judge.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/evaluation/judge.py): Offline deterministic LLM-as-a-judge scoring engine.
  * [`evaluation/test_suite.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/evaluation/test_suite.py): Automated benchmark test suite with pytest integration (`test_evaluation_benchmark`).
  * [`transcripts/`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/transcripts): 5 verifiable operational execution transcripts:
    * `memory_continuity.txt`: Multi-turn conversational memory demonstration.
    * `memory_reset.txt`: Session isolation & clean state reset demonstration.
    * `autogen_approved.txt`: AutoGen peer review approval transcript.
    * `autogen_revised.txt`: AutoGen peer review revision transcript.
    * `evaluation_report.md`: Quantitative 15-query evaluation scorecard (**0.980 Composite Score**).

---

### Phase 6: Secondary Peer Review Subsystem (Task 14)
* **Objective:** Implement secondary regulatory and compliance review prior to client delivery.
* **Deliverables:**
  * [`review/autogen_review.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/review/autogen_review.py): AutoGen `RoundRobinGroupChat` (max_turns=2) pairing `PolicyComplianceReviewer` and `FinalEditor`, emitting structured `VerdictModel`.

---

### Phase 7: AI Governance, Budget Limiting & Response Cache (Tasks 15–16)
* **Objective:** Protect system resources, classify AI risk tier, and deliver sub-millisecond responses on duplicate queries.
* **Deliverables:**
  * [`governance/budget_guard.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/governance/budget_guard.py): Token/character limiter (max 2,048 chars, max 512 tokens, HTTP 413) and EU AI Act / RBI High-Risk classification documentation.
  * [`governance/cache.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/governance/cache.py): In-memory normalized query response cache keyed by `query.strip().lower()`.

---

## 4. Critical Technical Decisions & Pitfall Mitigations

| Challenge / Pitfall | Impact | Implemented Mitigation |
| :--- | :--- | :--- |
| **Pitfall 1: ReAct Template Collision** | Agent parser mistakenly interprets system prompt ReAct instructions as agent actions. | In `MockLLM`, slice incoming prompt at conversational boundary; inspect only model-generated tokens. |
| **Pitfall 2: Tool Routing Collision** | Substring search for `"lookup"` falsely triggers `rag_lookup` when `check_loan_application_status` is needed. | Exact string equality check on tool names; rejection of fuzzy substring dispatch. |
| **Air-Gapped Embedding Failure** | Cloud/HuggingFace network requests fail in offline environments. | `DeterministicEmbedder` with signed hashing fallback preserving cosine geometry; explicit `USE_SENTENCE_TRANSFORMERS` and `USE_CHROMADB` flags. |
| **AutoGen Type Registration Crash** | Unregistered structured response types crash the AutoGen messaging loop. | Declared `custom_message_types=[StructuredMessage[VerdictModel]]` at group chat initialization. |
| **PII Data Leakage in Persistent Logs** | Writing unmasked PAN or Aadhaar violates banking privacy laws. | Pre-flight regex substitution before JSON serialization in `StructuredLogger`. |
