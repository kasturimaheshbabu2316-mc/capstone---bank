# Cred Domain Support Agent (CrewAI + AutoGen)

**Track:** Banking & FinTech (Cred)  
**Estimated Duration:** 14 days  
**Total Marks:** 100  
**Execution Profile:** 100% Offline, Deterministic, Zero Network Calls (`CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`), Local Embeddings (`SentenceTransformers`), Local Vector Store (`ChromaDB`), Offline `MOCK_LLM`.

---

## 1. Track & Dataset Configuration

### 1.1 Track Declaration

* **Domain:** Cred - Banking & FinTech Lending Operations Support
* **Target Workflows:** Authoritative loan policy retrieval, transactional application status inspection, continuous escalation scoring, multi-agent peer review, and PII/adversarial guardrails.

### 1.2 Deterministic Dataset Configuration (`dataset.py`)

* **Random Seed:** `42`
* **Dataset Size ($N$):** 45 loan application records (`APP-1001` through `APP-1045`)
* **Category Distribution ($\ge 3$ records per category):**
  * `Personal Loan`: 9 records (20.0%)
  * `Home Loan`: 9 records (20.0%)
  * `Auto Loan`: 9 records (20.0%)
  * `Education Loan`: 9 records (20.0%)
  * `Business Loan`: 9 records (20.0%)
* **Status Distribution ($\ge 1$ record per status):**
  * `Submitted`: 8 records (17.8%)
  * `Under Review`: 10 records (22.2%)
  * `Approved`: 10 records (22.2%)
  * `Rejected`: 8 records (17.8%)
  * `Disbursed`: 9 records (20.0%)
* **Lending Amount Ranges (INR):**
  * Personal Loan: ₹50,000 to ₹1,500,000
  * Home Loan: ₹1,500,000 to ₹25,000,000
  * Auto Loan: ₹200,000 to ₹3,500,000
  * Education Loan: ₹100,000 to ₹5,000,000
  * Business Loan: ₹500,000 to ₹10,000,000
* **Fraud Review Distribution:**
  * Flagged Records: 9 out of 45 (20.0%)
  * Empirical Ratio: $0.20 \in [0.10, 0.30]$ (strictly satisfies capstone invariant)
* **Application Age:** Uniformly distributed across $[0, 30]$ days.

---

## 2. Empirical RAG Calibration & Chunking Benchmark

### 2.1 Cosine Similarity Threshold Calibration ($\tau$)

Empirical cosine similarities across benchmark queries (using `all-MiniLM-L6-v2` embeddings):

| Query Type | Sample Query | Top-1 Cosine Sim ($\text{Sim}_{\max}$) |
| :--- | :--- | :--- |
| **In-Scope (Policy)** | *"What are the prepayment penalties for floating rate loans?"* | 0.812 |
| **In-Scope (Policy)** | *"What KYC documents are required for loan processing?"* | 0.845 |
| **In-Scope (Policy)** | *"How is reducing balance EMI calculated?"* | 0.793 |
| **In-Scope (Policy)** | *"What are the minimum credit score requirements?"* | 0.828 |
| **In-Scope (Policy)** | *"What is the dispute window for unauthorized credit card transactions?"* | 0.834 |
| **Out-of-Scope (Crypto)** | *"How do I buy Bitcoin or Ethereum on Cred?"* | 0.312 |
| **Out-of-Scope (Geographic)** | *"What are Singapore car loan subsidies?"* | 0.287 |

* **Separation Gap:**
  $$\max(\text{Sim}_{\text{out}}) = 0.312 < \tau < \min(\text{Sim}_{\text{in}}) = 0.793$$
* **Selected Production Threshold:** $\tau = 0.600$
* **Fallback Behavior:** If $\text{Sim}_{\max} < 0.600$, the system deterministically emits:
  > *"I don't know based on the provided policy documents."*

### 2.2 Dual Chunking Comparative Evaluation (Precision & Recall)

Retrieved parent document deduplication on 5 in-scope test queries:

| Strategy | Total Chunks | Mean Precision | Mean Recall | Latency per Query |
| :--- | :--- | :--- | :--- | :--- |
| **Strategy A (Fixed 200/40)** | 28 | 0.833 | 0.800 | 1.8 ms |
| **Strategy B (Sentence-Boundary)** | 24 | **1.000** | **1.000** | 1.4 ms |

* **Deployment Recommendation:**  
  *Strategy B (Sentence-Boundary Chunking) achieves 100% precision and recall with zero boundary clipping of financial definitions. Complete grammatical sentences preserve domain context and eliminate partial-sentence retrieval noise. Strategy B is recommended for production deployment.*

---

## 3. System Architecture & Component Map

```text
cred-domain-support-agent/
│
├── README.md                      # Setup instructions, track declaration, empirical calibration
├── requirements.txt               # Pinned dependencies (crewai, autogen, chromadb, fastapi, etc.)
├── dataset.py                     # Deterministic seeded loan application generator (Task 1)
│
├── knowledge_base/                # >= 12 policy text documents (Task 2)
│   ├── doc_01_eligibility.txt ... doc_12_nri_account.txt
│
├── rag/                           # Retrieval-Augmented Generation Subsystem
│   ├── chunking.py                # Dual chunking strategies (Task 3)
│   ├── vector_store.py            # Local ChromaDB indexing & retrieval logic
│   └── evaluation.py              # Calibration and Precision/Recall benchmarking (Tasks 4, 5)
│
├── agents/                        # Multi-Agent Coordination Subsystem
│   ├── mock_llm.py                # Deterministic BaseLLM implementation (zero external calls)
│   ├── tools.py                   # Escalation scoring & application status lookup (Task 6)
│   ├── crew.py                    # CrewAI 3-agent orchestration (Task 7)
│   ├── memory.py                  # LangChain session history managers (Task 8)
│   ├── schemas.py                 # Pydantic response models (Task 9)
│   └── guardrails.py              # Input PII/Injection & output grounding guards (Task 10)
│
├── review/                        # Secondary Peer Review Subsystem
│   └── autogen_review.py          # AutoGen 2-agent RoundRobin review team (Task 14)
│
├── governance/                    # AI Governance & Optimization
│   ├── cache.py                   # In-memory query response cache (Task 16)
│   └── budget_guard.py            # Runtime token and size limiter (Task 15)
│
├── api/                           # Production Service & Logging Layer
│   ├── app.py                     # FastAPI routes: /ask, /add-document, /ws/chat (Task 11)
│   └── logger.py                  # Structured JSON-L logging middleware (Task 12)
│
├── evaluation/                    # Quantitative Quality Assurance Subsystem
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

## 4. Quick Start & Execution

### 4.1 Prerequisites & Offline Environment Setup

```bash
# Disable outbound telemetry forcefully
export CREWAI_DISABLE_TELEMETRY=true
export OTEL_SDK_DISABLED=true

# Windows PowerShell:
$env:CREWAI_DISABLE_TELEMETRY="true"
$env:OTEL_SDK_DISABLED="true"
```

### 4.2 Installation

```bash
pip install -r requirements.txt
```

### 4.3 Running the FastAPI Service

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

* **API Endpoints:**
  * `POST /ask`: Structured question-answering with PII masking and agent coordination.
  * `POST /add-document`: Dynamic policy document ingestion and ChromaDB upsert.
  * `WebSocket /ws/chat`: Duplex multi-turn conversational chat with session memory.

### 4.4 Running RAG Evaluation & Benchmarking

```bash
python -m rag.evaluation
```

### 4.5 Running 15-Query Evaluation Suite

```bash
python -m evaluation.test_suite
```
