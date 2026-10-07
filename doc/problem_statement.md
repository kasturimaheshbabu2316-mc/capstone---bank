# Final Capstone Specification: Cred Domain Support Agent (CrewAI)

**Track:** Banking & FinTech (Cred)  
**Estimated Duration:** 14 days  
**Total Marks:** 100  
**Deliverable:** Single public GitHub repository link containing code, datasets, mock engines, tests, transcripts, and documentation.  
**Execution Constraints:** Zero network calls (`CREWAI_DISABLE_TELEMETRY=true`), zero paid accounts, 100% runnable under local embeddings (`SentenceTransformers`), local vector index (`ChromaDB`), and an offline deterministic `MOCK_LLM`.

---

## 1. Executive Summary & Production Brief

Cred's lending-operations team requires a production-grade domain support agent capable of answering loan policy inquiries from an authoritative knowledge base and retrieving customer loan application records from a curated transactional dataset. Support operations demand high consistency, zero data leakage, and rigorous governance.

Rather than a simple prototype, this system brings together dataset engineering, dual-strategy vector retrieval, conversational memory, structured tool orchestration, multi-agent peer review, guardrails, automated LLM-as-a-judge evaluation, and FastAPI deployment.

The entire system is designed to execute locally with **zero external API keys or cloud dependencies**, utilizing deterministic mock engines and local vector embeddings.

---

## 2. Global Constraints & Execution Requirements

1. **Deterministic Reproducibility:**
   - All dataset generation, RAG chunking, embeddings, agent workflows, and evaluations must run deterministically via explicit random seeds and fixed configurations.
   - The top of `README.md` must state the track (**Cred - Banking & FinTech**) and provide the exact dataset configuration (seed, category/status distributions, amount ranges).
2. **Zero-Cost & Offline Operation:**
   - Embeddings must use local models via `SentenceTransformers` (e.g., `all-MiniLM-L6-v2`).
   - Vector indexing must use local `ChromaDB` instances persisted on disk.
   - LLM generation across all stages (grounded RAG, evaluation judge, CrewAI agents, and AutoGen reviewers) must default to an offline, deterministic `MOCK_LLM`.
3. **CrewAI Custom LLM Extension Constraints:**
   - To implement `MOCK_LLM` within CrewAI, developers must subclass `crewai.llms.base_llm.BaseLLM`.
   - **Pitfall 1 (ReAct Template Collision):** CrewAI's default ReAct prompt contains the literal string `"Observation: the result of the action"`. A naive parser searching for `"Observation:"` will prematurely match the system prompt and return template placeholders. The mock parser must inspect only model-generated responses.
   - **Pitfall 2 (Tool Routing Collision):** Tool dispatching must evaluate the tool's declared schema or exact name identity rather than substring matching (e.g., substring matching on `"lookup"` will misclassify tools like `rag_lookup`).
4. **Telemetry Suppression:**
   - Outbound telemetry must be disabled prior to initializing any crew:

     ```bash
     export CREWAI_DISABLE_TELEMETRY=true
     export OTEL_SDK_DISABLED=true
     ```

5. **No Media Deliverables:**
   - All deliverables must be code, configuration, log files, or Markdown documentation inside the repository.

---

## 3. Scenario & Controlled Vocabularies

### 3.1 Controlled Vocabularies

- **Loan Categories (all required $\ge 1$ time; $\ge 3$ records per category in dataset):**
  1. `Personal Loan`
  2. `Home Loan`
  3. `Auto Loan`
  4. `Education Loan`
  5. `Business Loan`

- **Application Statuses (all required $\ge 1$ time):**
  1. `Submitted`
  2. `Under Review`
  3. `Approved`
  4. `Rejected`
  5. `Disbursed`

### 3.2 Required Knowledge Base Topics ($\ge 12$ distinct documents, $2\text{--}5$ sentences each)

1. Loan eligibility criteria by loan type
2. EMI calculation rules
3. Credit-card fee structure
4. KYC document requirements
5. Fraud-dispute resolution process
6. Account-closure process
7. Interest-rate slabs
8. Prepayment-penalty rules
9. Minimum-balance requirements
10. Credit-score impact factors
11. Joint-account rules
12. NRI-account eligibility

### 3.3 Guardrail-Relevant PII Scope

- **In-Scope Masking (Strict Regex Matching Required):**
  - **PAN Card Number:** Fixed format (5 uppercase letters, 4 digits, 1 uppercase letter, e.g., `[A-Z]{5}[0-9]{4}[A-Z]{1}`).
  - **Aadhaar Number:** Fixed 12-digit format (`\d{4}\s\d{4}\s\d{4}` or `\d{12}`).
  - **Bank Account Number:** Fixed-length numeric string ($9\text{--}18$ digits).

- **Out-of-Scope Masking (Acknowledged Exception):**
  - Applicant names and annual income figures represent free text / unformatted values with no deterministic local pattern matching under keyless `MOCK_LLM`. These must use synthetic test fixtures exclusively.

---

## 4. End-to-End System Architecture

```text
                                  ┌────────────────────────┐
                                  │   Inbound User Query   │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │ Input Guardrail Filter │  ◄── Token Budget Check
                                  │ (Regex Masking & Guard)│  ◄── PII Redaction (PAN/Aadhaar/Acc)
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                         ┌───────►│  Query Response Cache  │
                         │        └───────────┬────────────┘
                   Cache │                    │ Cache Miss
                    Hit  │                    ▼
                         │        ┌────────────────────────┐
                         │        │   CrewAI Agent Crew    │
                         │        │  (Deterministic Mock)  │
                         │        └─────┬────────────┬─────┘
                         │              │            │
            ┌────────────┴─────┐        │            │        ┌─────────────────────────┐
            │ Fast Return Path │        │            └───────►│ Application Lookup Tool │
            └──────────────────┘        ▼                     │ (Escalation Score Calc) │
                         ┌───────────────────────────┐        └─────────────────────────┘
                         │ Retrieval Agent (ChromaDB)│
                         │ (Precision/Recall Ranked) │
                         └──────────────┬────────────┘
                                        │
                                        ▼
                         ┌───────────────────────────┐
                         │   Composer Draft Output   │
                         │    (Pydantic Structured)  │
                         └──────────────┬────────────┘
                                        │
                                        ▼
                         ┌───────────────────────────┐
                         │    AutoGen Review Team    │
                         │ (Policy Check & Revision) │
                         └──────────────┬────────────┘
                                        │
                                        ▼
                         ┌───────────────────────────┐
                         │ Output Guardrail Validator│  ◄── Groundedness & Refusal Check
                         └──────────────┬────────────┘
                                        │
                                        ▼
                         ┌───────────────────────────┐
                         │ Structured JSON-L Logging │  ◄── Masked Audit Log with Trace ID
                         └──────────────┬────────────┘
                                        │
                                        ▼
                         ┌───────────────────────────┐
                         │  FastAPI HTTP / WebSocket │
                         └───────────────────────────┘
```

---

## 5. Detailed Technical Specifications by Part

### Part 1: Dataset Design & RAG Core (30 Marks)

#### Task 1: Dataset Generator (`dataset.py`)

- Implement a seeded, deterministic generator yielding a list of $\ge 40$ loan records named `LOAN_APPLICATIONS`.
- **Record Schema:**
  - `record_id`: String (e.g., `APP-1001`)
  - `category`: Categorical string matching given vocabulary
  - `status`: Categorical string matching given vocabulary
  - `loan_amount_inr`: Float or integer within a realistic lending bracket (must document rationale in `README.md`)
  - `days_since_created`: Integer in $[0, 30]$
  - `flagged_for_fraud_review`: Boolean
- **Statistical Invariants:**
  - Every loan category must have $\ge 3$ records.
  - Every application status must have $\ge 1$ record.
  - Fraud review flag constraint:
    $$0.10 \le \frac{\sum \mathbb{I}(\text{flagged\_for\_fraud\_review} == \text{True})}{N_{\text{total}}} \le 0.30$$
  - If initial generation fails this band, adjust generation weights or seed programmatically; manual post-editing of individual records is strictly prohibited.

#### Task 2: Knowledge Base Authoring

- Author $\ge 12$ standalone text documents covering each required scenario topic.
- Length: strictly $2\text{--}5$ sentences per document, written specifically for Cred banking domain operations.

#### Task 3: Dual Chunking & ChromaDB Indexing

- Implement two parallel indexing pipelines:
  - **Strategy A (Fixed-size with overlap):** e.g., 200 characters with 40-character overlap.
  - **Strategy B (Sentence-boundary chunking):** Semantic splitting preserving complete sentences.
- Embed chunks using local `SentenceTransformers` and index into two separate ChromaDB collections (`collection_fixed` and `collection_sentence`) using `.upsert()`.

#### Task 4: Grounded Generation & Empirical Threshold Calibration

- Implement retrieval retrieval-augmented generation returning answers restricted exclusively to retrieved context.
- **Empirical Fallback Calibration:**
  - Run $\ge 3$ in-scope queries and $\ge 2$ out-of-scope queries against the index.
  - Record the top-1 cosine similarities $\text{Sim}_{\max}$.
  - Identify the separation gap and select threshold $\tau$ strictly between the in-scope cluster and out-of-scope cluster:
    $$\tau \in (\max(\text{Sim}_{\text{out}}), \min(\text{Sim}_{\text{in}}))$$
  - When $\text{Sim}_{\max} < \tau$, immediately trigger the fallback response: *"I don't know based on the provided policy documents."*
  - Demonstrate on $\ge 5$ in-scope queries and $\ge 1$ out-of-scope query. Document all numbers in `README.md`.

#### Task 5: Chunking Strategy Evaluation (Document-Level Precision & Recall)

- For the $\ge 5$ in-scope queries, evaluate both collections.
- Map retrieved chunks back to their originating parent documents and deduplicate.
- Calculate Document-Level Precision and Recall:
  $$\text{Precision} = \frac{|D_{\text{retrieved}} \cap D_{\text{relevant}}|}{|D_{\text{retrieved}}|}, \quad \text{Recall} = \frac{|D_{\text{retrieved}} \cap D_{\text{relevant}}|}{|D_{\text{relevant}}|}$$
- Present per-query arithmetic tables for both strategies and provide a 2–3 sentence data-backed deployment recommendation.

---

### Part 2: CrewAI Orchestration, Tools, Memory & Guardrails (30 Marks)

#### Task 6: Application Status Tool & Escalation Scoring

- Implement tool: `check_loan_application_status(record_id: str) -> dict`.
- Output fields: `status`, `loan_amount_inr`, `escalation_score`.
- **Escalation Score Formula:**
  - Compute a continuous score $S \in [0, 1]$ combining fraud status and normalized recency:
    $$S = w_{\text{fraud}} \cdot \mathbb{I}(\text{flagged\_for\_fraud\_review}) + w_{\text{recency}} \cdot \left(\frac{\text{days\_since\_created}}{30}\right)$$
    *(where $w_{\text{fraud}} + w_{\text{recency}} = 1.0$, e.g., $w_{\text{fraud}} = 0.65, w_{\text{recency}} = 0.35$)*
  - Establish an escalation threshold (e.g., $S \ge 0.70$) justified against the 80th percentile distribution of your dataset.

#### Task 7: Multi-Agent Crew Construction

- Build a CrewAI crew using a custom subclass of `crewai.llms.base_llm.BaseLLM` running deterministic offline logic.
- **Required Agents ($\ge 3$):**
  1. **Retrieval Agent:** Has access only to the selected ChromaDB RAG tool.
  2. **Lookup Agent:** Has access only to `check_loan_application_status`.
  3. **Response Composer Agent:** Synthesizes tool outputs into a clear draft response.
- Execute via `crew.kickoff()` and demonstrate both tools being invoked across distinct sample queries.

#### Task 8: In-Memory Multi-Turn Session Memory

- Implement conversational history across turns using LangChain memory primitives (e.g., `InMemoryChatMessageHistory`).
- **Required Transcripts:**
  - Transcript A: Multi-turn session demonstrating memory continuity (e.g., Turn 1 asks about a record, Turn 2 asks a follow-up referring to "it" without repeating the ID).
  - Transcript B: Fresh session demonstrating complete state isolation (no context carryover).

#### Task 9: Structured Output Validation

- Define a Pydantic `BaseModel` for crew responses:

  ```python
  from pydantic import BaseModel, Field
  from typing import Optional, List

  class AgentResponseSchema(BaseModel):
      query: str
      intent: str = Field(description="Policy inquiry or Application lookup")
      direct_answer: str
      citations: List[str] = Field(default_factory=list)
      escalation_triggered: bool = False
      escalation_score: Optional[float] = None
  ```

- Validate all crew outputs against this schema before final handling.

#### Task 10: Input and Output Guardrails

- **Input Guardrails:**
  - Mask fixed-format PII (PAN: `[A-Z]{5}[0-9]{4}[A-Z]{1}` $\rightarrow$ `[MASKED_PAN]`, Aadhaar: `\d{4}\s?\d{4}\s?\d{4}` $\rightarrow$ `[MASKED_AADHAAR]`, Bank Account $\rightarrow$ `[MASKED_ACCOUNT]`).
  - Detect prompt injection attempts (e.g., `"ignore previous instructions"`, `"system override"`) and reject execution.
- **Output Guardrails:**
  - Validate that factual policy claims in the generated draft match the retrieved context. If context support is absent, refuse generation.
- Demonstrate deliberate test cases triggering each guardrail.

---

### Part 3: Evaluation, Observability & FastAPI Deployment (20 Marks)

#### Task 11: Production FastAPI Service

- Expose the agent pipeline using FastAPI:
  - `POST /ask`: Accepts query and session ID; returns structured `AgentResponseSchema`.
  - `POST /add-document`: Ingests and indexes new policy documents dynamically into ChromaDB.
  - `WebSocket /ws/chat`: Real-time duplex multi-turn conversational endpoint. Must catch `WebSocketDisconnect` cleanly, freeing session resources without crashing the server.

#### Task 12: ELK-Style Structured JSON Logging

- Emit structured single-line JSON logs (`.jsonl`) for every HTTP and WebSocket interaction.
- **Log Schema:**
  - `trace_id`: UUID4 string
  - `timestamp`: ISO 8601 UTC timestamp
  - `endpoint`: Target route
  - `latency_ms`: Request execution time in milliseconds
  - `masked_prompt`: Input text with fixed-format PII masked
  - `status_code`: HTTP status / completion code
- **Strict Compliance:** Under no circumstances may raw, unmasked PAN, Aadhaar, or Bank Account numbers be written to logs.

#### Task 13: Quantitative LLM-as-a-Judge Evaluation (15 Test Cases)

- Assemble a benchmark of 15 queries:
  - 12 covering each required Knowledge Base topic.
  - 1 covering loan application status lookup.
  - 2 out-of-scope / adversarial edge cases.
- Run an offline LLM-as-a-judge prompt to score each interaction on a $[0.0, 1.0]$ scale across 4 core dimensions:
  1. **Accuracy:** Correctness relative to ground truth policy.
  2. **Grounding:** Strict adherence to retrieved context without hallucination.
  3. **Completeness:** Addressing all constraints in the query.
  4. **Safety:** Correct handling of PII, injection defense, and out-of-scope refusal.
- Report per-query scores and composite averages in a summary table.

---

### Part 4: Resilience, AutoGen Review & AI Governance (20 Marks)

#### Task 14: AutoGen Secondary Review Team

- Build a post-processing review stage using AutoGen's `RoundRobinGroupChat`:
  - **Agents:** `PolicyComplianceReviewer` and `FinalEditor`.
  - Configure with `max_turns=2` (or `MaxMessageTermination(3)`).
- **Structured Output Protocol:**
  - Define verdict schema:

    ```python
    from pydantic import BaseModel

    class VerdictModel(BaseModel):
        approved: bool
        final_answer: str
        reason: str
    ```

  - Set `output_content_type=VerdictModel` on `FinalEditor`.
  - Construct team with `custom_message_types=[StructuredMessage[VerdictModel]]` to prevent runtime registration crashes.
- Demonstrate two execution cases:
  1. Direct approval of a compliant draft.
  2. Revision of an ungrounded or non-compliant draft.

#### Task 15: Four-Layer AI Governance Framework

- **Application Layer (Least Autonomy):**
  - Verify that only the Lookup Agent can call `check_loan_application_status`. Any invocation attempt by other agents is mechanically blocked.
- **System Risk Tiering:**
  - Document a 1-paragraph risk classification: Classify the Cred Domain Support Agent under **High Risk** due to financial impact, consumer credit access implications, and regulatory scrutiny.
- **Runtime Layer (Budget Guard):**
  - Enforce token and character budget limits per request. Oversized inputs exceeding thresholds must be immediately rejected with an HTTP `413 Request Entity Too Large` or structured error.

#### Task 16: Normalized Query Response Cache

- Implement an in-memory cache keyed by normalized query strings (`query.strip().lower()`).
- On duplicate queries, return cached responses directly, bypassing CrewAI and vector retrieval.
- Demonstrate cache hits via execution time metrics or invocation counters.

---

## 6. Repository Layout Specification

A compliant submission must organize code and deliverables according to the following tree structure:

```text
cred-domain-support-agent/
│
├── README.md                      # Setup instructions, track declaration, empirical calibration
├── requirements.txt               # Pinned dependencies (crewai, autogen, chromadb, fastapi, etc.)
├── dataset.py                     # Deterministic seeded loan application generator (Task 1)
│
├── knowledge_base/                # >= 12 policy text documents (Task 2)
│   ├── doc_01_eligibility.txt
│   ├── doc_02_emi_rules.txt
│   └── ...
│
├── rag/
│   ├── chunking.py                # Dual chunking strategies (Task 3)
│   ├── vector_store.py            # Local ChromaDB indexing & retrieval logic
│   └── evaluation.py              # Calibration and Precision/Recall benchmarking (Tasks 4, 5)
│
├── agents/
│   ├── mock_llm.py                # Deterministic BaseLLM implementation (zero external calls)
│   ├── tools.py                   # Escalation scoring & application status lookup (Task 6)
│   ├── crew.py                    # CrewAI 3-agent orchestration (Task 7)
│   ├── memory.py                  # LangChain session history managers (Task 8)
│   ├── schemas.py                 # Pydantic response models (Task 9)
│   └── guardrails.py              # Input PII/Injection & output grounding guards (Task 10)
│
├── review/
│   └── autogen_review.py          # AutoGen 2-agent RoundRobin review team (Task 14)
│
├── governance/
│   ├── cache.py                   # In-memory query response cache (Task 16)
│   └── budget_guard.py            # Runtime token and size limiter (Task 15)
│
├── api/
│   ├── app.py                     # FastAPI routes: /ask, /add-document, /ws/chat (Task 11)
│   └── logger.py                  # Structured JSON-L logging middleware (Task 12)
│
├── evaluation/
│   ├── test_suite.py              # 15-query test dataset runner (Task 13)
│   └── judge.py                   # LLM-as-a-judge scoring matrix
│
└── transcripts/                   # Verifiable test run logs and CLI outputs
    ├── memory_continuity.txt      # Multi-turn conversational memory demonstration
    ├── memory_reset.txt           # Fresh conversation state isolation demonstration
    ├── autogen_approved.txt       # AutoGen draft approval transcript
    ├── autogen_revised.txt        # AutoGen draft revision transcript
    └── evaluation_report.md       # Quantitative 15-query evaluation results
```

---

## 7. Capstone Acceptance Matrix

| Item | Requirement Verified | Pass Criteria |
| :--- | :--- | :--- |
| **P1.1** | `dataset.py` Determinism & Schema | Produces $\ge 40$ records; categories $\ge 3$; fraud flags strictly between $10\%\text{--}30\%$. |
| **P1.2** | Knowledge Base Breadth | $\ge 12$ distinct documents covering all listed policy topics ($2\text{--}5$ sentences each). |
| **P1.3** | Dual Vector Indexing | Fixed-size and sentence chunking stored in separate ChromaDB collections. |
| **P1.4** | Cosine Threshold Calibration | Empirically derived fallback threshold $\tau$ documented with in-scope/out-of-scope numbers. |
| **P1.5** | Chunking Comparative Benchmark | Precision & Recall calculated per query for both collections with written recommendation. |
| **P2.1** | Escalation Calculation | Continuous score $S \in [0, 1]$ combining fraud and recency with percentile justification. |
| **P2.2** | CrewAI Multi-Agent Team | $\ge 3$ agents interacting via custom `BaseLLM` without external network access. |
| **P2.3** | Session Memory | Continuity demonstrated across turns, isolated cleanly on session reset. |
| **P2.4** | Pydantic Output Enforcement | All agent responses parse into strict `AgentResponseSchema`. |
| **P2.5** | Guardrail Activation | Regex masks PAN/Aadhaar/Account; injection blocked; ungrounded claims refused. |
| **P3.1** | FastAPI Deployment | HTTP `/ask`, `/add-document`, and WebSocket `/ws/chat` surviving client disconnects. |
| **P3.2** | Structured Logging | JSON-Lines logs with trace IDs, latencies, and zero unmasked PII. |
| **P3.3** | 15-Query Evaluation | Tabular output scoring Accuracy, Grounding, Completeness, Safety across all 15 queries. |
| **P4.1** | AutoGen Review Team | RoundRobin 2-agent group chat approving compliant and revising non-compliant drafts. |
| **P4.2** | AI Governance Layers | Least autonomy verified; risk tier documented; runtime budget cap enforced. |
| **P4.3** | Response Caching | Duplicate query returns cached result with verified latency reduction / zero tool calls. |
