# Task Breakdown & Work Breakdown Structure (WBS)

**Project:** Cred Domain Support Agent (Banking & FinTech)  
**Track:** Banking & FinTech (Cred)  
**Total Tasks:** 16 Tasks across 4 Core Parts  
**Status:** 100% Completed & Verified  
**Reference:** [ARCHITECTURE.md](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/doc/ARCHITECTURE.md) | [problem_statement.md](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/doc/problem_statement.md)

---

## 1. Master Task Checklist & Completion Status

| Task ID | Component Name | Primary File | Key Technical Requirement | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Task 1** | Seeded Dataset Generator | [`dataset.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/dataset.py) | Seed=42, $N=45 \ge 40$, 5 categories, fraud rate 13.3% $\in [10\%, 30\%]$ | **Completed** |
| **Task 2** | Authoritative Knowledge Base | [`knowledge_base/`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/knowledge_base) | 12 policy files, 2–5 sentences each, covering retail lending | **Completed** |
| **Task 3** | Dual Chunking Strategies | [`rag/chunking.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/rag/chunking.py) | Strategy A (200/40 fixed) vs Strategy B (sentence boundary) | **Completed** |
| **Task 4** | Vector Store & Fallback Calibration | [`rag/vector_store.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/rag/vector_store.py), [`rag/evaluation.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/rag/evaluation.py) | ChromaDB persistence, $\tau = 0.25$ cutoff, authoritative fallback | **Completed** |
| **Task 5** | Chunking Strategy Benchmark | [`rag/evaluation.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/rag/evaluation.py) | Precision & Recall evaluation across 5 benchmark queries | **Completed** |
| **Task 6** | Application Status Tool & Escalation | [`agents/tools.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/tools.py) | Status lookup, $S = 0.65\cdot\mathbb{I}(\text{fraud}) + 0.35\cdot(\text{days}/30)$, $\theta = 0.70$ | **Completed** |
| **Task 7** | Deterministic MockLLM & CrewAI Crew | [`agents/mock_llm.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/mock_llm.py), [`agents/crew.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/crew.py) | 3-agent least privilege, ReAct template & tool routing mitigations | **Completed** |
| **Task 8** | Multi-Turn Conversational Memory | [`agents/memory.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/memory.py) | LangChain `InMemoryChatMessageHistory`, session isolation & continuity | **Completed** |
| **Task 9** | Structured Pydantic Response Schema | [`agents/schemas.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/schemas.py) | Pydantic v2 `AgentResponseSchema` contract validation | **Completed** |
| **Task 10** | Dual-Stage Safety Guardrails | [`agents/guardrails.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/guardrails.py) | Regex PII masking (PAN, Aadhaar, Account), injection defense, grounding | **Completed** |
| **Task 11** | Production FastAPI Service | [`api/app.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/api/app.py) | `POST /ask`, `POST /add-document`, duplex `WebSocket /ws/chat` | **Completed** |
| **Task 12** | Structured ELK Audit Logging | [`api/logger.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/api/logger.py) | Single-line JSON-L (`logs/audit.jsonl`), UUID4 trace IDs, zero raw PII | **Completed** |
| **Task 13** | LLM Judge & 15-Query Evaluation | [`evaluation/judge.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/evaluation/judge.py), [`evaluation/test_suite.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/evaluation/test_suite.py) | 15-query test suite, 4 scoring dimensions, 5 verifiable transcripts | **Completed** |
| **Task 14** | AutoGen Secondary Peer Review Team | [`review/autogen_review.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/review/autogen_review.py) | `RoundRobinGroupChat` (max_turns=2), `VerdictModel` structured output | **Completed** |
| **Task 15** | AI Governance Budget Guard | [`governance/budget_guard.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/governance/budget_guard.py) | Max 2,048 chars, max 512 tokens (HTTP 413), High-Risk AI tiering | **Completed** |
| **Task 16** | Normalized Query Response Cache | [`governance/cache.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/governance/cache.py) | Normalized string key (`query.strip().lower()`), sub-millisecond bypass | **Completed** |

---

## 2. Granular Task Breakdown & Technical Specifications

### Task 1: Synthetic Transactional Dataset Generator

* **File:** [`dataset.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/dataset.py)
* **Scope:** Deterministic generation of synthetic loan application records.
* **Invariants Enforced:**
  1. Fixed seed (`seed=42`).
  2. Dataset count $N = 45 \ge 40$.
  3. Controlled Categories: `Personal Loan`, `Home Loan`, `Auto Loan`, `Education Loan`, `Business Loan` ($\ge 3$ records per category).
  4. Controlled Statuses: `Submitted`, `Under Review`, `Approved`, `Rejected`, `Disbursed` ($\ge 1$ record per status).
  5. Fraud Flag Rate: $6/45 = 13.3\%$, strictly within $[0.10, 0.30]$.
* **Verification:**

  ```powershell
  python dataset.py
  # Output: Generated 45 loan applications. Fraud flagged: 6/45 (13.3%)
  ```

---

### Task 2: Authoritative Knowledge Base Corpus

* **Directory:** [`knowledge_base/`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/knowledge_base)
* **Scope:** 12 authoritative banking policy text documents ($2\text{--}5$ sentences each):
  * `doc_01_eligibility.txt`: Age bounds (21–60), CIBIL $\ge 750$.
  * `doc_02_emi_rules.txt`: Reducing balance formula, auto-debit on 5th.
  * `doc_03_card_fees.txt`: Annual fees and spend-based fee waiver rules.
  * `doc_04_kyc_docs.txt`: OVD requirements (PAN, Aadhaar, Passport).
  * `doc_05_fraud_dispute.txt`: 45-day dispute window, provisional credit.
  * `doc_06_account_closure.txt`: Zero-dues certificate, NOC issuance.
  * `doc_07_interest_slabs.txt`: APR slabs from 8.5% to 24.0%.
  * `doc_08_prepayment.txt`: Zero penalty on floating rate loans.
  * `doc_09_min_balance.txt`: AMB requirements and shortfall penalties.
  * `doc_10_credit_score.txt`: Credit score factor weightings.
  * `doc_11_joint_account.txt`: Co-borrower liability and survivorship.
  * `doc_12_nri_account.txt`: NRE/NRO accounts and FEMA regulations.

---

### Task 3: Dual Chunking Strategies

* **File:** [`rag/chunking.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/rag/chunking.py)
* **Scope:** Implementation of two contrasting chunking approaches:
  * Strategy A: `chunk_fixed_size` (window: 200 chars, overlap: 40 chars).
  * Strategy B: `chunk_sentence_boundary` (regex `(?<=[.!?])\s+`, 1–2 complete sentences).

---

### Task 4: Vector Indexing & Empirical Fallback Calibration

* **Files:** [`rag/vector_store.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/rag/vector_store.py), [`rag/evaluation.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/rag/evaluation.py)
* **Scope:** Local ChromaDB indexing into `collection_fixed` and `collection_sentence`, deterministic embedding generation, and empirical cutoff calibration.
* **Findings:**
  * Minimum in-scope similarity: 0.2560
  * Maximum out-of-scope similarity: 0.2087
  * Calibrated threshold: $\tau = 0.25$
  * Fallback string: *"I don't know based on the provided policy documents."*

---

### Task 5: Chunking Strategy Precision & Recall Benchmark

* **File:** [`rag/evaluation.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/rag/evaluation.py)
* **Scope:** Evaluated 5 in-scope queries across both collections against ground truth document sets.
* **Results:**
  * Strategy A (Fixed): Mean Precision = 0.20, Mean Recall = 0.50
  * Strategy B (Sentence): Mean Precision = 0.30, Mean Recall = 0.70
* **Recommendation:** Strategy B selected for production deployment due to grammatical completeness.

---

### Task 6: Application Status Tool & Continuous Escalation Scoring

* **File:** [`agents/tools.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/tools.py)
* **Scope:** Retrieval of loan records and computation of escalation risk metric:
  $$S = 0.65 \cdot \mathbb{I}(\text{fraud}) + 0.35 \cdot \left(\frac{\text{days\_since\_created}}{30}\right)$$
* **Threshold:** $S \ge 0.70$ triggers supervisor escalation.

---

### Task 7: Deterministic MockLLM & CrewAI Multi-Agent Team

* **Files:** [`agents/mock_llm.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/mock_llm.py), [`agents/crew.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/crew.py)
* **Scope:** 3-agent team with least-privilege tool isolation:
  * Retrieval Agent: `rag_lookup` only.
  * Lookup Agent: `check_loan_application_status` only.
  * Response Composer Agent: Synthesis only (no tools).
* **Pitfall Mitigations:** ReAct template boundary slicing and exact string tool matching.

---

### Task 8: In-Memory Multi-Turn Conversational Memory

* **File:** [`agents/memory.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/memory.py)
* **Scope:** Session management with LangChain `InMemoryChatMessageHistory`.
* **Verification:**
  * Demonstrates cross-turn reference resolution (APP-1002).
  * Verified session isolation and clean memory reset via `reset_session`.

---

### Task 9: Structured Pydantic Response Validation

* **File:** [`agents/schemas.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/schemas.py)
* **Scope:** Strict validation of agent outputs against `AgentResponseSchema`:
  * `query`, `intent`, `direct_answer`, `citations`, `escalation_triggered`, `escalation_score`.

---

### Task 10: Dual-Stage Guardrail & Safety Pipeline

* **File:** [`agents/guardrails.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/agents/guardrails.py)
* **Scope:**
  * Pre-flight regex masking: PAN (`[MASKED_PAN]`), Aadhaar (`[MASKED_AADHAAR]`), Account (`[MASKED_ACCOUNT]`).
  * Prompt injection detection and rejection.
  * Post-flight context grounding gate.

---

### Task 11: Production FastAPI Service & WebSocket Stream

* **File:** [`api/app.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/api/app.py)
* **Scope:** Endpoints `POST /ask`, `POST /add-document`, and `WebSocket /ws/chat`. Clean socket disconnect handling on client close.

---

### Task 12: Single-Line ELK-Compatible Audit Logging

* **File:** [`api/logger.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/api/logger.py)
* **Scope:** Structured logging to `logs/audit.jsonl` with UUID4 trace IDs, latency measurements, and 100% PII masking.

---

### Task 13: Quantitative LLM-as-a-Judge Evaluation & Transcripts

* **Files:** [`evaluation/judge.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/evaluation/judge.py), [`evaluation/test_suite.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/evaluation/test_suite.py), [`transcripts/`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/transcripts)
* **Scope:** Automated 15-query benchmark scoring Accuracy (35%), Grounding (35%), Completeness (15%), Safety (15%).
* **Outcome:** Composite Average = **0.980**, Safety = **1.000**.
* **Transcripts Generated:**
  * `transcripts/memory_continuity.txt`
  * `transcripts/memory_reset.txt`
  * `transcripts/autogen_approved.txt`
  * `transcripts/autogen_revised.txt`
  * `transcripts/evaluation_report.md`

---

### Task 14: AutoGen Secondary Peer Review Team

* **File:** [`review/autogen_review.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/review/autogen_review.py)
* **Scope:** 2-agent `RoundRobinGroupChat` (`PolicyComplianceReviewer` + `FinalEditor`) emitting `StructuredMessage[VerdictModel]`.

---

### Task 15: AI Governance Budget Guard & High-Risk Tiering

* **File:** [`governance/budget_guard.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/governance/budget_guard.py)
* **Scope:** Payload limits (2,048 chars / 512 tokens, HTTP 413) and regulatory classification under EU AI Act & RBI guidelines.

---

### Task 16: Sub-Millisecond Normalized Query Response Cache

* **File:** [`governance/cache.py`](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/governance/cache.py)
* **Scope:** In-memory caching keyed by `query.strip().lower()`. Delivers $< 1\text{ ms}$ latency on duplicate inquiries.
