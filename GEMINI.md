# GRACE Agent Project Context & Rules

## ⚠️ CRITICAL RULES (MUST ADHERE)

1.  **NO UNAUTHORIZED CHANGES**:
    *   Do **NOT** modify any code (`replace`, `write_file`, etc.) unless the user explicitly commands "Apply fix", "Modify", or similar corrective actions.
    *   During "Investigation", "Root Cause Analysis", or "Code Review" phases, use **READ-ONLY** tools only (`read_file`, `search_file_content`, `list_directory`, etc.).
    *   Never attempt to "try out" a logic change or "revert" a previous change on your own initiative.

2.  **EXPLICIT APPROVAL MANDATORY**:
    *   Before executing any file-modifying tool, you **MUST** describe the exact changes, the reason for the change, and obtain a "Go ahead" or "Approve" from the user.
    *   This applies even when correcting your own mistakes or returning to a previous state.

3.  **STRICT SEPARATION OF CONCERNS**:
    *   Separate the "Analysis" turn from the "Implementation" turn. Complete the analysis first, present the findings, and wait for implementation orders.

---

# GRACE Agent (Guided Reasoning with Adaptive Confidence Execution) - Project Context

## 1. Project Overview
This project, formerly "Gemini3 Hybrid RAG", is evolving into **GRACE (Guided Reasoning with Adaptive Confidence Execution)**. It is a next-generation AI agent that combines **Plan-and-Execute**, **ReAct**, and **Reflection** patterns with a unique **Confidence-aware** mechanism.

**Core Vision:**
*   **G**uided: Guided by initial planning and Human-In-The-Loop (HITL) intervention.
*   **R**easoning: Inherits ReAct for step-by-step reasoning.
*   **A**daptive: Dynamically replans when confidence is low or errors occur.
*   **C**onfidence: Calculates a confidence score (0.0-1.0) for every action and answer.
*   **E**xecution: Robust execution with specialized agents (Planner, Executor).

**Current Status:**
*   **Legacy Base:** Functional ReAct Agent with RAG pipeline (Qdrant + Gemini/OpenAI).
*   **Active Development:** Implementing the `grace/` core module (Planner, Executor, Confidence Calculator).

## 2. Architecture & Tech Stack

### Hybrid Agent Architecture
GRACE uses a hybrid approach integrating multiple agentic patterns:
1.  **Plan-and-Execute:** Generates a multi-step plan before execution.
2.  **ReAct (Reason + Act):** Executes each step with reasoning (inherited from existing `agent_main.py`).
3.  **Reflection:** Self-evaluates results (inherited).
4.  **HITL (Human-In-The-Loop):** Requests user confirmation or clarification based on Confidence Score.

### Tech Stack
*   **LLM:** **Gemini 2.0 Flash** (Primary for all reasoning/planning).
*   **Embedding:** `gemini-embedding-001` (Unified for new GRACE features).
*   **Vector Database:** Qdrant (Local via Docker).
*   **Task Queue:** Celery with Redis (for async heavy lifting).
*   **UI:** Streamlit (Management Dashboard & Planned Chat Interface).

## 3. Key Files & Directories

### Current Implementation (Legacy/Foundation)
| File/Directory | Description |
| :--- | :--- |
| **`agent_main.py`** | **Legacy CLI:** The original interactive ReAct agent loop. |
| `agent_rag.py` | **Management UI:** Streamlit app for RAG data ingestion/QA generation. |
| `helper_llm.py` | **Core:** Unified wrapper for LLM API calls. |
| `helper_rag.py` | **Core:** RAG logic, chunking, and text processing. |
| `qdrant_client_wrapper.py` | **Core:** Qdrant database interactions. |

### GRACE Architecture (Planned/In-Progress)
| File/Directory | Description |
| :--- | :--- |
| **`grace/`** | **[NEW]** Core package for the GRACE engine. |
| `grace/planner.py` | Generates execution plans (Pydantic schemas). |
| `grace/executor.py` | Executes plans step-by-step using ReAct. |
| `grace/confidence.py` | Calculates confidence scores based on RAG results & LLM self-eval. |
| `grace/intervention.py` | Determines when to ask the user (HITL) based on thresholds. |
| `ui/pages/grace_chat_page.py` | **[NEW]** Specialized UI for the GRACE agent interaction. |

## 4. Confidence & Intervention Logic

The unique feature of GRACE is **Confidence-aware Execution**.
*   **Confidence Score:** Calculated from RAG search quality, Source Agreement, and LLM Self-Evaluation.
*   **Intervention Levels:**
    *   **Silent (>0.9):** Auto-proceed.
    *   **Notify (0.7-0.9):** Show progress.
    *   **Confirm (0.4-0.7):** Ask "Is this plan okay?".
    *   **Escalate (<0.4):** Ask "I need more info about...".

## 5. Setup & Usage

### Prerequisites
*   Python 3.10+
*   Docker & Docker Compose (for Qdrant/Redis)
*   Gemini API Key in `.env`

### Starting Services
```bash
# Start Qdrant and Redis
docker-compose -f docker-compose/docker-compose.yml up -d
```

### Running the Application
**Option 1: Legacy CLI (Current)**
```bash
python agent_main.py
```

**Option 2: Management UI (Streamlit)**
```bash
streamlit run rag_qa_pair_qdrant.py
```

## 6. Development Guidelines

*   **Code Style:** PEP 8. Use `ruff`.
*   **Testing:** `pytest tests/` (Targeting 80% coverage for new `grace/` modules).
*   **Convention:**
    *   New logic goes into `grace/`.
    *   Reuse `services/qdrant_service.py` and `helper_llm.py` where possible.
    *   **Avoid** `old_code/` and legacy OpenAI-specific paths.
*   **Mermaid Diagrams:**
    *   Use **Simple Syntax** (v9 compatible) for compatibility with PyCharm Markdown viewer.
    *   Avoid complex styling (`:::`) or new features (`@{}`).
    *   Example:
        ```mermaid
        graph TD
            A[Start] --> B{Decision}
            B -->|Yes| C[End]
        ```
