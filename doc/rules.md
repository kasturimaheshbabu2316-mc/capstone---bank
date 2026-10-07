# System Operating Rules & AI Governance Directives

**Project:** Cred Domain Support Agent (Banking & FinTech)  
**System:** Multi-Agent Lending Operations Support Platform  
**Document Version:** 1.0.0  
**Status:** Mandatory Operational Directives  
**Reference:** [ARCHITECTURE.md](file:///c:/Users/kastu/Desktop/captstone%20-%20bank/doc/ARCHITECTURE.md)

---

## 1. Regulatory Governance & Risk Classification

### Rule 1.1: EU AI Act High-Risk System Classification

* **Classification:** The Cred Domain Support Agent is formally designated as a **High-Risk AI System** under the European Union Artificial Intelligence Act (Annex III, point 5b: *"AI systems intended to be used to evaluate the creditworthiness of natural persons or establish their credit score"*).
* **Operational Mandate:** The system must maintain complete auditability, continuous risk management, verifiable human-in-the-loop escalation paths, and zero-hallucination factual grounding.

### Rule 1.2: RBI Digital Lending Compliance

* In accordance with Reserve Bank of India (RBI) Digital Lending Guidelines:
  * All disclosed loan terms (interest slabs, fees, prepayment penalties, EMI auto-debit dates) must derive verbatim from authoritative policy documents.
  * Disclosures of interest rates or fee waivers not present in the indexed policy repository are strictly prohibited.

---

## 2. Statistical Invariants & Dataset Integrity Rules

### Rule 2.1: Deterministic Dataset Generation (`dataset.py`)

* The dataset generator must utilize a fixed random seed (`seed=42`).
* Total record volume must satisfy $N_{\text{total}} \ge 40$ (system implements $N = 45$).
* **Category Diversity:** Every loan category (`Personal Loan`, `Home Loan`, `Auto Loan`, `Education Loan`, `Business Loan`) must contain at least 3 records.
* **Status Coverage:** Every application status (`Submitted`, `Under Review`, `Approved`, `Rejected`, `Disbursed`) must contain at least 1 record.
* **Fraud Proportion Bounds:** The fraction of records flagged for fraud review must strictly satisfy:
  $$0.10 \le \frac{\sum_{i=1}^{N} \mathbb{I}(\text{fraud}_i)}{N} \le 0.30$$
  *Violation of this bound invalidates the dataset. Manual tampering or hardcoding records is forbidden.*

---

## 3. Retrieval & Fallback Policy Rules

### Rule 3.1: Empirical Cosine Similarity Threshold ($\tau$)

* Vector retrieval executes against the calibrated threshold $\tau = 0.25$.
* **Authoritative Fallback Rule:** If the top-1 cosine similarity between a user query and indexed policy chunks is strictly less than $\tau$ ($\text{Sim}_{\max} < 0.25$), the retrieval engine must abort semantic generation and emit the verbatim fallback response:
  > *"I don't know based on the provided policy documents."*
* Under no circumstances may the system attempt to generate speculative answers on topics outside the indexed knowledge base corpus (e.g., cryptocurrency, gold loans, equity trading).

### Rule 3.2: Mandatory Document Citations

* Any policy inquiry that yields similarity $\ge \tau$ must include parent document identifiers in the `citations` list of the response schema (e.g., `["doc_01_eligibility", "doc_07_interest_slabs"]`).

---

## 4. Multi-Agent Least-Privilege & RBAC Rules

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        LEAST PRIVILEGE RBAC MATRIX                     │
├───────────────────────┬───────────────────┬────────────────────────────┤
│ Agent Role            │ Allowed Tool(s)   │ Strictly Prohibited Tool(s)│
├───────────────────────┼───────────────────┼────────────────────────────┤
│ Retrieval Agent       │ rag_lookup        │ check_loan_app_status      │
│ Lookup Agent          │ check_loan_app_...│ rag_lookup                 │
│ Response Composer     │ None (Synthesis)  │ rag_lookup, check_status   │
└───────────────────────┴───────────────────┴────────────────────────────┘
```

### Rule 4.1: Role-Based Tool Isolation

* **Retrieval Agent (Policy Specialist):** Permitted access to `rag_lookup` only. It must possess zero capability to access or inspect the customer loan application database.
* **Lookup Agent (Application Officer):** Permitted access to `check_loan_application_status` only. It must possess zero access to the ChromaDB vector collections.
* **Response Composer Agent:** Least privilege: zero tool execution privileges. Operates strictly as an aggregator and structured response synthesizer.

### Rule 4.2: Deterministic Mock LLM Mitigations

* **Mitigation 1 (ReAct Template Collision):** In `MockLLM.call()`, the model must slice incoming prompt strings at the conversational turn boundary and inspect only model-generated tokens, preventing false parser matching against CrewAI's system prompt `"Observation:"`.
* **Mitigation 2 (Tool Routing Collision):** Tool matching must evaluate exact string equality (`tool_name == "rag_lookup"` vs `tool_name == "check_loan_application_status"`). Substring matching on substrings like `"lookup"` is prohibited.

---

## 5. Continuous Escalation Scoring Rules

### Rule 5.1: Mathematical Formulation

* Every loan application query must compute an objective risk metric $S \in [0.0, 1.0]$:
  $$S = 0.65 \cdot \mathbb{I}(\text{flagged\_for\_fraud\_review}) + 0.35 \cdot \left(\frac{\text{days\_since\_created}}{30}\right)$$
* Invariant: $w_{\text{fraud}} + w_{\text{recency}} = 0.65 + 0.35 = 1.00$.

### Rule 5.2: Escalation Threshold

* If $S \ge 0.70$, the tool must set `escalation_triggered = True`.
* The final synthesized customer response must append:
  > *"Case has been escalated to senior operations supervisor for priority review."*

---

## 6. Privacy & Security Guardrail Rules

### Rule 6.1: Strict Regex PII Masking

* Prior to dispatching inbound queries to downstream agents or persisting logs, all sensitive identifiers must be masked:
  1. **Permanent Account Number (PAN):** `\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b` $\longrightarrow$ `[MASKED_PAN]`
  2. **Aadhaar Number:** `\b\d{4}\s?\d{4}\s?\d{4}\b` $\longrightarrow$ `[MASKED_AADHAAR]`
  3. **Bank Account Number:** `\b\d{9,18}\b` (in account context) $\longrightarrow$ `[MASKED_ACCOUNT]`
* **Zero-Leakage Guarantee:** Writing unmasked PAN, Aadhaar, or Bank Account numbers to disk, files, or audit logs is strictly prohibited.

### Rule 6.2: Prompt Injection Interception

* Inbound queries matching adversarial prompt injection patterns:

  ```regex
  (?i)(ignore\s+previous\s+instructions|system\s+override|disregard\s+all\s+prior|act\s+as\s+dan|reveal\s+system\s+prompt)
  ```

  must be intercepted immediately, returning a 400 Bad Request / security refusal without invoking LLM reasoning.

### Rule 6.3: Post-Generation Grounding Verification

* Before response transmission, claims asserted in the draft answer must be verified against retrieved context chunks. If unsupported factual assertions are detected, the response must be substituted with the authoritative fallback string.

---

## 7. AI Governance & Runtime Budget Rules

### Rule 7.1: Runtime Ingress Limits (`governance/budget_guard.py`)

* **Maximum Inbound Characters:** 2,048 characters.
* **Maximum Estimated Tokens:** 512 tokens ($\text{word\_count} \times 2$).
* **Enforcement:** Payloads exceeding either ceiling must be rejected immediately with HTTP 413 (Payload Entity Too Large).

### Rule 7.2: Normalized Query Response Caching (`governance/cache.py`)

* Repeated inquiries normalized by `query.strip().lower()` must return cached `AgentResponseSchema` payloads directly, achieving latency $< 2\text{ ms}$ and completely bypassing multi-agent orchestration.

---

## 8. Offline & Engineering Execution Rules

### Rule 8.1: Air-Gapped Zero-Telemetry Execution

* External network calls are strictly prohibited.
* Telemetry environment flags must be forcefully set prior to importing agent frameworks:

  ```python
  os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
  os.environ["OTEL_SDK_DISABLED"] = "true"
  ```

### Rule 8.2: Pydantic Contract Enforcement

* All outputs emitted by the primary multi-agent crew must validate against `AgentResponseSchema`.
* All review outputs emitted by the AutoGen secondary team must validate against `VerdictModel`.

### Rule 8.3: Consolidated Source Code Documentation (`app`)

* **Mandatory Single Document Principle:** All Python (`.py`) source code files across the project must be compiled and maintained inside a single consolidated document named **`app`** (`app.md` / `doc/app.md`).
* **Folder Hierarchy Preservation:** In the `app` document, each `.py` file must be grouped, titled, and indexed under its originating folder (e.g., `rag/`, `agents/`, `api/`, `review/`, `governance/`, `evaluation/`).
* **Operational Purpose:** Enables immediate full-text searchability, streamlined code audits, and rapid project comprehension without navigating multiple fragmented directories.
