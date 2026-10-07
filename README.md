# Cred Domain Support Agent (Banking & FinTech)

**Track:** Banking & FinTech (Cred)  
**System:** Multi-Agent Lending Operations Support Platform  
**Target Architecture:** CrewAI, AutoGen, ChromaDB, SentenceTransformers, FastAPI, LangChain, Pydantic  
**Execution Profile:** 100% Offline, Deterministic, Zero Telemetry, Zero Cloud Dependencies  

---

## 1. Project Overview

The **Cred Domain Support Agent** is an enterprise-grade AI lending operations assistant designed to automate customer support inquiries for Cred's retail lending operations. It handles loan eligibility policies, EMI schedules, credit card fee waivers, KYC procedures, loan application status lookups, and continuous escalation risk calculations.

---

## 2. Repository Structure

```text
.
├── app/                           # Unified application package
│   ├── __init__.py                # Package exports & public API interface
│   ├── db.py                      # Seeded dataset, O(1) lookup & ChromaDB store
│   ├── main.py                    # Production FastAPI gateway & WebSocket
│   ├── memory.py                  # Session memory manager & LangChain buffers
│   ├── models.py                  # Pydantic v2 contracts (AgentResponseSchema, VerdictModel)
│   ├── pipeline.py                # CrewAI orchestration & AutoGen compliance review
│   └── tools.py                   # Application status lookup & continuous escalation engine
├── dataset.py                     # Seeded synthetic loan applications dataset generator
├── knowledge_base/                # 12 authoritative banking policy text documents
│   ├── doc_01_eligibility.txt
│   ├── doc_02_emi_rules.txt
│   ├── doc_03_card_fees.txt
│   ├── doc_04_kyc_docs.txt
│   ├── doc_05_fraud_dispute.txt
│   ├── doc_06_account_closure.txt
│   ├── doc_07_interest_slabs.txt
│   ├── doc_08_prepayment.txt
│   ├── doc_09_min_balance.txt
│   ├── doc_10_credit_score.txt
│   ├── doc_11_joint_account.txt
│   └── doc_12_nri_account.txt
├── rag/                           # RAG Subsystem
│   ├── chunking.py                # Fixed-size (200/40) & sentence-boundary chunking
│   ├── vector_store.py            # Local ChromaDB indexer, embedder, and cosine retriever
│   └── evaluation.py              # Empirical threshold calibration (tau = 0.25) & P/R benchmark
├── agents/                        # Multi-Agent Coordination Subsystem
│   ├── mock_llm.py                # Deterministic BaseLLM subclass with ReAct parser
│   ├── tools.py                   # check_loan_application_status & escalation engine
│   ├── crew.py                    # CrewAI 3-agent team definition & kickoff runner
│   ├── memory.py                  # LangChain InMemoryChatMessageHistory session manager
│   ├── schemas.py                 # Pydantic v2 AgentResponseSchema models
│   └── guardrails.py              # Regex PII masking, injection detector, grounding gate
├── review/                        # Secondary Peer Review Subsystem
│   └── autogen_review.py          # AutoGen 2-agent RoundRobinGroupChat & VerdictModel
├── governance/                    # AI Governance & Optimization
│   ├── budget_guard.py            # Runtime token and character budget limiter (HTTP 413)
│   └── cache.py                   # In-memory normalized query response cache
├── api/                           # Production Service & Logging Layer
│   ├── app.py                     # FastAPI application: /ask, /add-document, /ws/chat
│   └── logger.py                  # Single-line ELK-compatible JSON-L audit logger
├── evaluation/                    # Quantitative Quality Assurance Subsystem
│   ├── judge.py                   # Offline LLM-as-a-judge scoring engine
│   └── test_suite.py              # 15-query evaluation benchmark suite runner
├── transcripts/                   # Verifiable Operational Execution Artifacts
│   ├── memory_continuity.txt      # Multi-turn conversational memory demonstration
│   ├── memory_reset.txt           # Session isolation & clean state reset demonstration
│   ├── autogen_approved.txt       # AutoGen peer review approval transcript
│   ├── autogen_revised.txt        # AutoGen peer review revision transcript
│   └── evaluation_report.md       # Quantitative 15-query evaluation scorecard (0.980 Composite)
├── doc/                           # Comprehensive Engineering Documentation
│   ├── ARCHITECTURE.md            # System architecture & technical specification
│   ├── app.md                     # Consolidated code document for all .py files
│   ├── prd.md                     # Product Requirements Document
│   ├── design.md                  # Detailed System Design Document
│   ├── implementation-plan.md     # Phased Implementation Roadmap
│   ├── task.md                    # Work Breakdown Structure (Tasks 1-16)
│   ├── memory.md                  # Conversational Memory Architecture Specification
│   ├── rules.md                   # System Operating Rules & Governance Directives
│   └── problem_statement.md       # Capstone problem statement & track declaration
├── requirements.txt               # Pinned dependencies
└── .gitignore                     # Git ignore configuration
```

---

## 3. Quick Start & Execution

### 3.1 Setup Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 Run Automated Test Suite
```bash
pytest evaluation/test_suite.py
```

### 3.3 Run API Server
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
