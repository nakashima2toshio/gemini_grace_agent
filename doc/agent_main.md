# agent_main.py 詳細設計書

## 1. 概要
`agent_main.py` は、Gemini 2.0 Flash を使用した **CLI (Command Line Interface) 専用の Hybrid RAG エージェント** のエントリーポイントです。
Streamlit UI (`agent_rag.py` / `services/agent_service.py`) とは独立した実装となっており、サーバーサイドでの単独動作やデバッグ用途に使用されます。

**主な特徴:**
*   **ReAct (Reasoning + Acting) ループ:** ユーザーの質問に対し、「思考 (Thought) → ツール実行 (Action) → 結果観察 (Observation)」のサイクルを回し、情報を検索・収集してから回答します。
*   **動的システムプロンプト:** 起動時にQdrantからコレクション一覧を取得し、利用可能なナレッジベースに応じてシステム指示（Router Guidelines）を動的に生成します。
*   **IPOログ:** 入力、LLM応答、ツール結果などの中間状態を詳細にログ出力します。

## 2. モジュール構成と依存関係

| インポート元 | 使用する機能 |
| :--- | :--- |
| `google.generativeai` | Gemini API クライアント (`GenerativeModel`, `ChatSession`) |
| `config` | 設定値 (`AgentConfig`, `PathConfig`, `QdrantConfig`) |
| `agent_tools` | RAG検索ツール (`search_rag_knowledge_base`) |
| `services.qdrant_service` | コレクション一覧取得 (`get_all_collections`) |

## 3. 関数詳細

### 3.1 `get_system_instruction`

動的にシステムプロンプトを生成する関数です。

*   **Process:**
    1.  Qdrantクライアントを初期化し、`get_all_collections` で全コレクション名を取得。
    2.  コレクション名に基づいて、使用ヒント（Wikipediaなら一般知識、Livedoorならニュース等）を生成。
    3.  これらを組み込んだ「Router Guidelines」を含むプロンプト文字列を構築して返す。

### 3.2 `setup_agent`

エージェントの初期化を行います。

*   **Process:**
    1.  環境変数 (`GEMINI_API_KEY`) の検証。
    2.  ツールリストの登録 (`search_rag_knowledge_base`, `list_rag_collections`)。
    3.  `get_system_instruction()` を呼び出し、最新のシステムプロンプトを取得。
    4.  `genai.GenerativeModel` を初期化し、`chat_session` を開始。

### 3.3 `run_agent_turn`

1ターンの対話（ユーザー入力 → 最終回答）を実行するコアロジックです。
`services.agent_service.ReActAgent` とは異なり、**関数内で独自にReActループを実装**しています。

*   **Input:**
    *   `chat_session`: Gemini ChatSession
    *   `user_input`: ユーザーの質問
    *   `return_tool_info`: ツール使用情報を返すかどうかのフラグ
*   **Process (ReAct Loop):**
    1.  `chat_session.send_message(user_input)` でLLMに問い合わせ。
    2.  **ループ開始:**
        *   レスポンス解析: `text` (思考/回答) と `function_call` (ツール呼び出し) をチェック。
        *   **Thought**: 思考プロセスがあればログ出力。
        *   **Function Call**:
            *   ツール名と引数を抽出。
            *   `tools_map` から対応する関数を実行 (`agent_tools.py` 内のロジック)。
            *   結果を `function_response` として `chat_session.send_message` で返送。
            *   ループ継続。
        *   **Final Answer**: ツール呼び出しがない場合、テキストを最終回答としてループ終了。
*   **Output:**
    *   最終回答テキスト
    *   (Optional) ツール使用情報辞書

    ```mermaid
    graph TD
        Start[User Input] --> Send[Send to LLM]
        Send --> Check[Check Function Call]
        
        Check -->|Yes| Extract[Extract Tool & Args]
        Extract --> Exec[Execute Tool via agent_tools]
        Exec --> Log[Log IPO]
        Log --> Return[Return Result to LLM]
        Return --> Check
        
        Check -->|No| Answer[Extract Final Text]
        Answer --> End[Return Response]
    ```

### 3.4 `main`

CLIのメインループです。

*   **Process:**
    1.  ロギング設定 (`setup_logging`)。
    2.  エージェント初期化 (`setup_agent`)。
    3.  **対話ループ (While True):**
        *   標準入力 (`input()`) 待機。
        *   終了コマンド (`exit`, `quit`) チェック。
        *   `run_agent_turn` 実行。
        *   回答を表示。
        *   エラー発生時はキャッチしてログ出力し、ループ継続。

## 4. データ構造

### ログ出力 (IPO Log)
`run_agent_turn` 内では、以下の形式で詳細な処理ログが出力されます。デバッグやトレーサビリティに使用されます。

```text
==================== [LEGACY AGENT IPO: INITIAL INPUT] ====================
(ユーザー入力)
============================================================

==================== [LEGACY AGENT IPO: LLM OUTPUT] ====================
[Text]: Thought: ...
[Function Call]: search_rag_knowledge_base(...)
============================================================

==================== [LEGACY AGENT IPO: TOOL RESULT INPUT] ====================
(検索結果のJSON文字列)
============================================================
```