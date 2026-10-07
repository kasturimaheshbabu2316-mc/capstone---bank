# Product Requirements Document (PRD)

**Project:** Cred Domain Support Agent (Lending Operations)  
**Track:** Banking & FinTech (Cred)  
**Version:** 1.0.0  
**Status:** Approved / Production Specification  
**Reference:** [ARCHITECTURE.md](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/doc/ARCHITECTURE.md)

---

## 1. Executive Summary & Product Vision

### 1.1 Vision Statement
The **Cred Domain Support Agent** is an enterprise-grade AI lending operations assistant designed to automate customer support inquiries for Cred's retail credit facilities. The platform resolves policy inquiries, checks loan application statuses, evaluates operational escalation risks, and enforces bank-grade data security with zero reliance on cloud APIs or external telemetry.

### 1.2 Core Problem Statement
Customer inquiries in digital lending operations require immediate, 100% accurate, and compliant responses. Typical risks in standard conversational systems include:
1. **Financial Hallucination:** Generating inaccurate interest rates, eligibility criteria, or fee structures.
2. **PII Exposure:** Leaking sensitive financial identifiers such as PAN, Aadhaar, and Bank Account numbers.
3. **Escalation Blind Spots:** Inability to continuously calculate fraud risk and SLA aging on active loan files.
4. **Adversarial Exploitation:** Susceptibility to prompt injections and boundary evasion.

The Cred Domain Support Agent resolves these challenges through a deterministic, air-gapped, multi-agent architecture incorporating layered guardrails, multi-turn session memory, and secondary compliance verification.

---

## 2. Target Personas & Stakeholders

| Persona | Role | Primary Needs & Jobs-to-be-Done |
| :--- | :--- | :--- |
| **Cred Member / Borrower** | End-User | - Instant clarity on loan eligibility, EMI schedules, and credit card fee waivers.<br>- Real-time status lookup on active loan applications with pronominal context.<br>- High confidentiality regarding personal identification. |
| **Lending Operations Officer** | Internal Staff | - Automated triage of routine policy inquiries.<br>- Continuous escalation calculation prioritizing high-risk applications ($S \ge 0.70$) for manual intervention. |
| **Risk & Compliance Auditor** | Enterprise Regulator | - Verification of zero ungrounded advice.<br>- Strict compliance with RBI Digital Lending Guidelines and EU AI Act High-Risk system tiering.<br>- Complete ELK-compatible audit trail with zero persisted PII. |

---

## 3. Product Scope & Functional Requirements

### 3.1 Feature Group 1: Authoritative Policy Inquiry Resolution (RAG)
* **FR-1.1:** System shall ingest and index an authoritative knowledge base of at least 12 distinct policy documents covering personal loans, home loans, auto loans, credit card fees, KYC, fraud disputes, and account closure.
* **FR-1.2:** System shall implement dual chunking strategies:
  * Strategy A: Fixed-size character window (200 characters, 40-character overlap).
  * Strategy B: Sentence-boundary chunking (1–2 sentences preserving atomic clauses).
* **FR-1.3:** System shall enforce an empirical cosine similarity cutoff ($\tau = 0.25$). Queries falling below $\tau$ must immediately return the exact fallback response:
  > *"I don't know based on the provided policy documents."*
* **FR-1.4:** Grounded responses must cite parent policy document identifiers (e.g., `doc_01_eligibility`, `doc_08_prepayment`).

### 3.2 Feature Group 2: Transactional Loan Application Lookup & Continuous Escalation
* **FR-2.1:** System shall query active loan application records from a synthetic, invariant-validated dataset of $\ge 40$ records generated with a fixed seed (`seed=42`).
* **FR-2.2:** Each lookup shall return loan category, current status, loan amount in INR, and days since created.
* **FR-2.3:** System shall compute a continuous escalation score $S \in [0.0, 1.0]$ using the weighted formulation:
  $$S = 0.65 \cdot \mathbb{I}(\text{flagged\_for\_fraud\_review}) + 0.35 \cdot \left(\frac{\text{days\_since\_created}}{30}\right)$$
* **FR-2.4:** If $S \ge 0.70$, the system must flag `escalation_triggered = True` and append an operational priority routing message to the customer response.

### 3.3 Feature Group 3: Multi-Agent CrewAI Orchestration
* **FR-3.1:** Multi-agent pipeline shall deploy 3 specialized agents with least-privilege tool isolation:
  * **Retrieval Agent:** Restricted to `rag_lookup` only.
  * **Lookup Agent:** Restricted to `check_loan_application_status` only.
  * **Response Composer Agent:** Zero tools; synthesizes structured customer-facing responses.
* **FR-3.2:** Execution shall be powered by an offline deterministic LLM subclass (`MockLLM`) with zero external network connectivity.
* **FR-3.3:** Output contract must validate against the Pydantic v2 schema `AgentResponseSchema`.

### 3.4 Feature Group 4: Secondary Peer Review (AutoGen)
* **FR-4.1:** All draft responses shall pass through an AutoGen `RoundRobinGroupChat` (maximum 2 turns) comprising:
  * `PolicyComplianceReviewer`: Verifies grounding, absence of PII, and professional tone.
  * `FinalEditor`: Emits a structured verdict using `VerdictModel(approved, final_answer, reason)`.
* **FR-4.2:** Ungrounded claims shall be intercepted and revised to the authoritative fallback string.

### 3.5 Feature Group 5: Conversational Memory
* **FR-5.1:** System shall maintain session-isolated conversational memory using LangChain `InMemoryChatMessageHistory`.
* **FR-5.2:** Memory must support multi-turn pronominal continuity (e.g., resolving *"What is its status?"* to the application referenced in the prior turn).
* **FR-5.3:** Session reset (`reset_session`) must cleanly clear memory buffers to guarantee zero cross-session data leakage.

### 3.6 Feature Group 6: Production Gateway & Audit Logging
* **FR-6.1:** FastAPI service shall provide:
  * `POST /ask`: Primary synchronous query endpoint.
  * `POST /add-document`: Dynamic knowledge ingestion endpoint.
  * `WebSocket /ws/chat`: Duplex streaming endpoint with session lifecycle management.
* **FR-6.2:** Single-line ELK-compatible JSON-L audit logs shall be written to `logs/audit.jsonl` with UUID4 trace IDs, latency metrics, and 100% masked user inputs.

---

## 4. Non-Functional & Regulatory Requirements

### 4.1 Air-Gapped & Offline Execution
* **NFR-1.1:** Zero cloud API invocations (no external OpenAI, Anthropic, or external embedding endpoints).
* **NFR-1.2:** Hard disablement of all framework telemetry (`CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`).
* **NFR-1.3:** Vector embedding inference executes locally via `SentenceTransformers` (`all-MiniLM-L6-v2`) or deterministic zero-mean signed hashing fallback.

### 4.2 Security & Data Privacy Guardrails
* **NFR-2.1 (PII Masking):** Strict regex masking before downstream processing:
  * Permanent Account Number (PAN): `\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b` $\longrightarrow$ `[MASKED_PAN]`
  * Aadhaar Number: `\b\d{4}\s?\d{4}\s?\d{4}\b` $\longrightarrow$ `[MASKED_AADHAAR]`
  * Bank Account Number: `\b\d{9,18}\b` $\longrightarrow$ `[MASKED_ACCOUNT]`
* **NFR-2.2 (Injection Defense):** Rejection of prompt injection attempts (e.g., `"ignore previous instructions"`, `"system override"`).
* **NFR-2.3 (Log Sanitation):** Zero raw PII written to log files or disk storage.

### 4.3 AI Governance & Risk Classification
* **NFR-3.1 (High-Risk Classification):** Governed under EU AI Act High-Risk Category (Annex III, point 5b) and RBI Digital Lending Guidelines due to impact on consumer credit terms and financial decisions.
* **NFR-3.2 (Runtime Budget Limiter):** Inbound payloads exceeding 2,048 characters or 512 estimated tokens must be rejected with HTTP 413.

### 4.4 Latency & Performance
* **NFR-4.1 (Normalized Cache):** Repeated queries normalized via `query.strip().lower()` must return from in-memory cache with latency $< 2\text{ ms}$, bypassing vector search and agent coordination.
* **NFR-4.2:** Cold agent pipeline execution latency $\le 100\text{ ms}$.

---

## 5. Success Metrics & Key Performance Indicators (KPIs)

| Metric | Target Threshold | Measured Performance | Verification Tool |
| :--- | :--- | :--- | :--- |
| **Composite Evaluation Score** | $\ge 0.85$ (on $[0.0, 1.0]$ scale) | **0.980** | `evaluation/test_suite.py` |
| **Safety Evaluation Score** | $1.00$ (100% compliance) | **1.000** | `evaluation/judge.py` |
| **PII Redaction Rate** | $100\%$ across PAN, Aadhaar, Account | **100%** | `agents/guardrails.py` |
| **Fallback Recall on Out-of-Scope** | $100\%$ accurate fallback | **100%** | `rag/evaluation.py` |
| **Synthetic Dataset Invariants** | $N \ge 40$, categories $\ge 3$, fraud $\in [0.10, 0.30]$ | **$N=45$, fraud=13.3%** | `dataset.py` |
| **Cache Hit Speedup** | $\ge 10\times$ speedup on duplicate queries | **$>20\times$ speedup ($<1\text{ ms}$)** | `governance/cache.py` |

---

## 6. Assumptions & Out-of-Scope Boundaries

1. **Unstructured PII:** Free-form entity extraction for arbitrary personal names and income figures without standardized format is outside the deterministic regex scope.
2. **Network I/O:** All execution occurs locally in an air-gapped test environment without live banking core API integrations.
3. **Persisted Vector Store:** ChromaDB stores data locally under `./data/chroma_db` without external database clusters.
