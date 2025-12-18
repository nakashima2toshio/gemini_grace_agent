# Gemini3 Hybrid RAG Agent - Project Context

## 1. Project Overview
This project implements a **Hybrid RAG (Retrieval-Augmented Generation) Agent** capable of intelligently switching between general conversation and specialized knowledge retrieval. It is designed to process Japanese and English documents, generate Q/A pairs, store them in a **Qdrant** vector database, and use **Google Gemini 2.0 Flash** as the reasoning engine.

**Key Capabilities:**
*   **ReAct Agent:** Uses Gemini 2.0 to reason (`[🧠 Thought]`) and decide when to call tools (`[🛠️ Tool Call]`).
*   **RAG Pipeline:** Automated chunking (SemanticCoverage), Q/A pair generation (LLM-based), and vector embedding.
*   **Hybrid Architecture:** Supports both OpenAI (legacy/alternative) and Gemini (current focus) models for embeddings and generation.
*   **Scalability:** Uses **Celery + Redis** for parallel processing of large datasets.

## 2. Architecture & Tech Stack

### Core Components
*   **LLM:** Google Gemini 2.0 Flash (Reasoning/Chat), GPT-4o (Optional/Legacy).
*   **Vector Database:** Qdrant (Local via Docker).
*   **Embedding:** `text-embedding-004` (Gemini) or `text-embedding-3-small` (OpenAI).
*   **Task Queue:** Celery with Redis broker (for async Q/A generation).
*   **UI/Interface:**
    *   **CLI:** `agent_main.py` (Interactive Agent Terminal).
    *   **Web UI:** `rag_qa_pair_qdrant.py` (Streamlit Dashboard).
        *Note: Mermaid diagrams in the Streamlit UI's explanation page (`ui/pages/explanation_page.py`) are styled with a black background, white text, and white border for nodes.*

### Data Flow
1.  **Ingestion:** Documents (cc_news, livedoor, wikipedia) are loaded and preprocessed.
2.  **Chunking:** Text is split using `SemanticCoverage` (paragraph-aware + MeCab for Japanese).
3.  **Q/A Generation:** LLMs generate Q/A pairs from chunks (handled asynchronously via Celery).
4.  **Embedding & Storage:** Q/A pairs are embedded and stored in Qdrant collections.
5.  **Retrieval (Agent):** The Agent receives a user query, decides if RAG is needed, searches Qdrant, and synthesizes an answer.

## 3. Key Files & Directories

| File/Directory | Description |
| :--- | :--- |
| **`agent_main.py`** | **Entry Point (CLI):** The main interactive loop for the ReAct agent. |
| `agent_rag.py` | **Entry Point (GUI):** Streamlit app for managing data, generation, and search. |
| `agent_tools.py` | Defines the tools (functions) available to the agent (e.g., `search_rag_knowledge_base`). |
| `celery_tasks.py` | Definitions for asynchronous Celery tasks (Q/A generation). |
| `helper_rag.py` | Core RAG logic, including chunking and text processing. |
| `helper_llm.py` | Unified wrapper for LLM API calls (Gemini/OpenAI). |
| `qdrant_client_wrapper.py` | Abstraction layer for Qdrant database interactions. |
| `config.py` / `config.yml` | Configuration settings (paths, model names, API keys). |
| `doc/` | Extensive documentation (Installation, Spec, Architecture). |
| `docker-compose/` | Setup for Qdrant and Redis services. |

## 4. Setup & Usage

### Prerequisites
*   Python 3.10+
*   Docker & Docker Compose (for Qdrant/Redis)
*   API Keys (Gemini, OpenAI) in `.env`

### Starting Services
```bash
# Start Qdrant and Redis
docker-compose -f docker-compose/docker-compose.yml up -d

# (Optional) Start Celery Workers for background processing
./start_celery.sh start -w 8
```

### Running the Application
**Option 1: CLI Agent (Interactive)**
```bash
python agent_main.py
```
*Use this for testing the agent's reasoning and tool usage.*

**Option 2: Streamlit Dashboard (Management)**
```bash
streamlit run rag_qa_pair_qdrant.py
```
*Use this for data ingestion, Q/A generation, and inspecting the vector DB.*

## 5. Development Guidelines

*   **Code Style:** Follow PEP 8. Use `ruff` for linting if available.
*   **Type Hinting:** Strongly encouraged for all new functions, especially in `services/` and `helper_*.py`.
*   **Logging:** Use the standard `logging` module. The agent logs detailed traces to `logs/`.
*   **Testing:** Run tests using `pytest`.
    ```bash
    pytest tests/
    ```
*   **Convention:** When modifying the RAG pipeline, ensure changes are compatible with both the synchronous (local) and asynchronous (Celery) execution modes.
*   **Mermaid Diagrams:** When creating Mermaid diagrams for documentation, use **simple, version 9-compatible syntax**.
    *   **Problem:** The Markdown viewer in PyCharm Professional uses an older version of Mermaid (v9) and frequently throws syntax errors with newer features, even if they render correctly in other tools (e.g., Typora).
    *   **Requirement:** Always use simple graph structures and basic syntax to ensure correct rendering within the IDE.
    *   **DO:**
        - Use basic `graph TD`, `graph LR`, `sequenceDiagram`, `flowchart` structures
        - Keep node labels simple (avoid special characters)
        - Use standard arrow syntax: `-->`, `---`, `-.->`, `==>`
    *   **DON'T:**
        - Use `:::` class assignments or inline styles
        - Use `subgraph` with complex nesting
        - Use `%%` comments inside diagram blocks
        - Use newer features like `&` for parallel paths or `@{...}` annotations
    *   **Example (GOOD):**
        ```mermaid
        graph TD
            A[Start] --> B[Process]
            B --> C{Decision}
            C -->|Yes| D[End]
            C -->|No| B
        ```
