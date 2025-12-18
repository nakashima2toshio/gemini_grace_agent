# agent_chat_page.py 詳細設計図

## 1. モジュール概要

| 項目 | 内容 |
|------|------|
| ファイル名 | `ui/pages/agent_chat_page.py` |
| 目的 | Gemini 2.0 Flash を使用した ReAct 型エージェントとの対話インターフェース |
| 主要機能 | Qdrant 上のナレッジベースを動的に選択し、RAG 検索を行いながら回答 |
| 使用技術 | Streamlit, Google Generative AI, Qdrant, ReAct パターン |

---

## 2. 重要な定数一覧

| 定数名 | 型 | 役割・概要 |
|--------|-----|-----------|
| `SYSTEM_INSTRUCTION_TEMPLATE` | `str` | エージェントのシステムプロンプトテンプレート。ReActプロセス（Thought/Action/Observation）の出力フォーマット、行動指針（Router Guidelines）、コレクション選択ヒント、再試行戦略を定義 |
| `REFLECTION_INSTRUCTION` | `str` | Reflection（自己評価・修正）フェーズ用のプロンプト。正確性・適切性・スタイルのチェックリストと出力フォーマットを定義 |
| `TOOLS_MAP` | `Dict[str, Callable]` | ツール名と実際の関数のマッピング辞書。`search_rag_knowledge_base` と `list_rag_collections` を登録 |

### 2.1 SYSTEM_INSTRUCTION_TEMPLATE の構成

```mermaid
flowchart TB
    subgraph HEADER["SYSTEM_INSTRUCTION_TEMPLATE"]
        direction TB
        subgraph REACT["ReAct プロセスと出力フォーマット"]
            B["ツール使用時: Thought → Action → Observation"]
            C["最終回答時: Thought → Answer"]
        end
        subgraph GUIDELINES["行動指針 Router Guidelines"]
            D["1. 専門知識の検索条件"]
            E["2. コレクション選択のヒント 言語マッチング"]
            F["3. 再試行戦略 Multi-turn Strategy"]
            G["4. 一般的な会話の処理"]
            H["5. 正直さと不足情報の処理"]
            I["6. 回答のスタイル"]
        end
        subgraph PLACEHOLDER["プレースホルダー"]
            J["available_collections: 利用可能なコレクション名"]
        end
    end

    REACT --> GUIDELINES --> PLACEHOLDER
```

### 2.2 コレクション選択ガイド（定数内定義）

| コレクション名 | 対象言語 | 用途 |
|---------------|----------|------|
| `cc_news` | 英語 | 英語ニュース記事 |
| `wikipedia_ja` | 日本語 | 百科事典、一般知識 |
| `livedoor` | 日本語 | ニュース、エンタメ、映画 |
| `japanese_text` | 日本語 | Webテキスト（予備用） |

---

## 3. 関数一覧と IPO 分析

### 3.1 get_available_collections_from_qdrant()

| 項目 | 内容 |
|------|------|
| **行番号** | 118-127 |
| **目的** | Qdrantから利用可能なコレクション名を取得 |

#### IPO

```mermaid
flowchart TB
    subgraph INPUT
        I1["環境変数 QDRANT_URL<br/>(デフォルト: http://localhost:6333)"]
    end

    subgraph PROCESS
        P1["1. QdrantClient を URL で初期化"]
        P2["2. client.get_collections() でコレクション一覧を取得"]
        P3["3. コレクション名のリストを生成"]
        P4["4. 例外発生時は空リストを返却"]
        P1 --> P2 --> P3 --> P4
    end

    subgraph OUTPUT
        O1["List[str]: コレクション名のリスト"]
        O2["失敗時: 空リスト []"]
    end

    INPUT --> PROCESS --> OUTPUT
```

---

### 3.2 setup_agent()

| 項目 | 内容 |
|------|------|
| **行番号** | 129-152 |
| **目的** | Geminiエージェント（ChatSession）のセットアップ |

#### 引数

| 引数名 | 型 | 説明 |
|--------|-----|------|
| `selected_collections` | `List[str]` | 検索対象として選択されたコレクション名のリスト |
| `model_name` | `str` | 使用するGeminiモデル名 |

#### IPO

```mermaid
flowchart TB
    subgraph INPUT
        I1["selected_collections: 選択されたコレクション名リスト"]
        I2["model_name: Geminiモデル名"]
        I3["環境変数: GEMINI_API_KEY or GOOGLE_API_KEY"]
    end

    subgraph PROCESS
        P1["1. API キーの取得・検証"]
        P2["2. genai.configure() で API 設定"]
        P3["3. ツールリストの準備<br/>- search_rag_knowledge_base<br/>- list_rag_collections"]
        P4["4. SYSTEM_INSTRUCTION_TEMPLATE にコレクション名を埋め込み"]
        P5["5. GenerativeModel の生成<br/>- model_name, tools, system_instruction を設定"]
        P6["6. model.start_chat() で ChatSession を開始<br/>- enable_automatic_function_calling=False"]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end

    subgraph OUTPUT
        O1["ChatSession: 初期化されたチャットセッション"]
        O2["エラー時: ValueError 例外をスロー"]
    end

    INPUT --> PROCESS --> OUTPUT
```

---

### 3.3 run_agent_turn()

| 項目 | 内容 |
|------|------|
| **行番号** | 154-306 |
| **目的** | エージェントの1ターンを実行（ReActループ + Reflection） |

#### 引数

| 引数名 | 型 | 説明 |
|--------|-----|------|
| `chat_session` | `ChatSession` | Gemini の ChatSession インスタンス |
| `user_input` | `str` | ユーザーからの入力メッセージ |

#### IPO

```mermaid
flowchart TB
    subgraph INPUT
        I1["chat_session: 初期化済みの ChatSession"]
        I2["user_input: ユーザーの質問文"]
    end

    subgraph PROCESS
        subgraph Phase1["Phase 1: ReAct ループ (max 10 turns)"]
            R1["response.parts をイテレート"]
            R2["part.text があれば思考ログに追加"]
            R3["part.function_call があれば:<br/>a. ツール名・引数を抽出<br/>b. TOOLS_MAP から関数を取得<br/>c. ツールを実行 (with st.spinner)<br/>d. 結果をログに記録<br/>e. 検索失敗時は log_unanswered_question<br/>f. function_response でモデルに結果を返却"]
            R4["function_call がなければループ終了"]
        end

        subgraph Phase2["Phase 2: Reflection (自己洗練)"]
            RF1["REFLECTION_INSTRUCTION + 回答案を送信"]
            RF2["レスポンスから Thought と Final Answer を分離"]
            RF3["Final Answer で最終回答を更新"]
            RF4["エラー時は Draft をそのまま使用"]
        end

        subgraph PostProcess["後処理"]
            PP1["思考ログを st.expander で表示"]
            PP2["'Answer:' タグがあれば抽出、なければ整形"]
        end
    end

    subgraph OUTPUT
        O1["str: 最終的な回答テキスト"]
        O2["副作用: st.expander で思考プロセスを表示"]
    end

    INPUT --> Phase1 --> Phase2 --> PostProcess --> OUTPUT
```

---

### 3.4 show_agent_chat_page()

| 項目 | 内容 |
|------|------|
| **行番号** | 312-519 |
| **目的** | メイン画面の表示とユーザーインタラクションの処理 |

#### IPO

```mermaid
flowchart TB
    subgraph INPUT
        I1["st.session_state: Streamlit セッション状態"]
        I2["ユーザー入力: サイドバー設定、チャット入力"]
    end

    subgraph PROCESS
        P1["詳細は「4. 処理フロー図」参照"]
    end

    subgraph OUTPUT
        O1["Streamlit UI の描画"]
        O2["st.session_state の更新"]
    end

    INPUT --> PROCESS --> OUTPUT
```

---

## 4. 処理フロー図

### 4.1 全体の概要処理フロー図

```mermaid
flowchart TB
    START([START])
    TITLE["タイトル表示<br/>(st.title)"]
    BLOCK_A["[Block A]<br/>元ドキュメント表示エリア<br/>(319-361行)"]
    BLOCK_B["[Block B]<br/>Q&A参照エリア<br/>(366-420行)"]
    BLOCK_C["[Block C]<br/>サイドバー設定<br/>(423-458行)"]
    BLOCK_D["[Block D]<br/>セッション初期化<br/>(460-492行)"]
    BLOCK_E["[Block E]<br/>チャット履歴表示<br/>(494-497行)"]
    BLOCK_F["[Block F]<br/>ユーザー入力処理<br/>(500-519行)"]
    END_NODE([END])

    START --> TITLE
    TITLE --> BLOCK_A
    BLOCK_A --> BLOCK_B
    BLOCK_B --> BLOCK_C
    BLOCK_C --> BLOCK_D
    BLOCK_D --> BLOCK_E
    BLOCK_E --> BLOCK_F
    BLOCK_F --> END_NODE

    style START fill:#90EE90
    style END_NODE fill:#FFB6C1
    style BLOCK_A fill:#E6E6FA
    style BLOCK_B fill:#E6E6FA
    style BLOCK_C fill:#FFEFD5
    style BLOCK_D fill:#E0FFFF
    style BLOCK_E fill:#F0FFF0
    style BLOCK_F fill:#FFF0F5
```

---

### 4.2 各処理ブロックの詳細フロー図

#### Block A: 元ドキュメント表示エリア (319-361行)

```mermaid
flowchart TB
    A_START["st.expander 開始<br/>'📄 元ドキュメント'"]
    A_PATTERNS["target_patterns定義<br/>- cc_news*.txt<br/>- japanese_text*.txt<br/>- livedoor*.txt<br/>- wikipedia_ja*.txt"]
    A_CHECK{"OUTPUT<br/>ディレクトリ<br/>存在?"}
    A_GLOB["glob パターンで<br/>ファイル検索"]
    A_NO_FILE["st.info<br/>'ファイルなし'"]
    A_LATEST["各パターンごとに<br/>最新ファイルを取得<br/>(max by ctime)"]
    A_SELECT["st.selectbox で<br/>ドキュメント選択"]
    A_READ["選択ファイルを<br/>先頭100行読み込み"]
    A_DISPLAY["st.text_area で表示"]

    A_START --> A_PATTERNS
    A_PATTERNS --> A_CHECK
    A_CHECK -->|Yes| A_GLOB
    A_CHECK -->|No| A_NO_FILE
    A_GLOB --> A_LATEST
    A_LATEST --> A_SELECT
    A_SELECT --> A_READ
    A_READ --> A_DISPLAY
```

---

#### Block B: Q&A参照エリア (366-420行)

```mermaid
flowchart TB
    B_START["st.expander 開始<br/>'📚 登録済みQ&A'"]
    B_GET["get_available_collections_from_qdrant()<br/>呼び出し"]
    B_CHECK{"コレクション<br/>存在?"}
    B_WARNING["st.warning<br/>'コレクションなし'"]
    B_SELECT["st.selectbox で<br/>コレクション選択"]
    B_SCROLL["QdrantClient で<br/>scroll() 実行<br/>(limit=100)"]
    B_EXTRACT["payload から<br/>question/answer抽出"]
    B_DISPLAY["pd.DataFrame 作成<br/>st.dataframe 表示"]

    B_START --> B_GET
    B_GET --> B_CHECK
    B_CHECK -->|Yes| B_SELECT
    B_CHECK -->|No| B_WARNING
    B_SELECT --> B_SCROLL
    B_SCROLL --> B_EXTRACT
    B_EXTRACT --> B_DISPLAY
```

---

#### Block C: サイドバー設定 (423-458行)

```mermaid
flowchart TB
    C_START["with st.sidebar:"]
    C_HEADER["st.header<br/>'⚙️ エージェント設定'"]
    C_MODEL["st.selectbox<br/>'使用モデル'<br/>(GeminiConfig.AVAILABLE_MODELS)"]
    C_GET["get_available_collections_from_qdrant()"]
    C_CHECK{"コレクション<br/>空?"}
    C_WARN["st.warning<br/>+ ['(None)']"]
    C_MULTI["st.multiselect<br/>'検索対象コレクション'<br/>(default=全選択)"]
    C_BUTTON["st.button<br/>'🗑️ 会話履歴クリア'"]
    C_PRESS{"ボタン押下?"}
    C_CLEAR["履歴クリア<br/>セッションクリア<br/>st.rerun"]
    C_NOTHING["何もしない"]

    C_START --> C_HEADER
    C_HEADER --> C_MODEL
    C_MODEL --> C_GET
    C_GET --> C_CHECK
    C_CHECK -->|Yes| C_WARN
    C_CHECK -->|No| C_MULTI
    C_WARN --> C_MULTI
    C_MULTI --> C_BUTTON
    C_BUTTON --> C_PRESS
    C_PRESS -->|Yes| C_CLEAR
    C_PRESS -->|No| C_NOTHING
```

---

#### Block D: セッション初期化 (460-492行)

```mermaid
flowchart TB
    D_INIT["chat_history<br/>初期化チェック<br/>(なければ [])"]
    D_FLAG["should_reinitialize = False"]
    D_COL_CHECK["コレクション変更チェック<br/>- current_collections が session_state にない?<br/>- 前回と今回のコレクションが異なる?"]
    D_COL_TRUE["should_reinitialize = True<br/>st.toast('コレクション変更...')"]
    D_MODEL_CHECK["モデル変更チェック<br/>- current_model が session_state にない?<br/>- 前回と今回のモデルが異なる?"]
    D_MODEL_TRUE["should_reinitialize = True<br/>st.toast('モデル変更...')"]
    D_REINIT{"再初期化<br/>必要?"}
    D_SETUP["setup_agent() 呼び出し<br/>↓<br/>session_state更新<br/>- chat_session<br/>- current_collections<br/>- current_model<br/>↓<br/>st.toast('準備完了')"]
    D_NEXT["次へ"]

    D_INIT --> D_FLAG
    D_FLAG --> D_COL_CHECK
    D_COL_CHECK --> D_COL_TRUE
    D_COL_TRUE --> D_MODEL_CHECK
    D_MODEL_CHECK --> D_MODEL_TRUE
    D_MODEL_TRUE --> D_REINIT
    D_REINIT -->|Yes| D_SETUP
    D_REINIT -->|No| D_NEXT
```

---

#### Block E: チャット履歴表示 (494-497行)

```mermaid
flowchart TB
    E_LOOP["for message in chat_history:"]
    E_DISPLAY["st.chat_message(role)<br/>└→ st.markdown(content)"]

    E_LOOP --> E_DISPLAY
```

---

#### Block F: ユーザー入力処理 (500-519行)

```mermaid
flowchart TB
    F_INPUT["st.chat_input<br/>'質問を入力...'"]
    F_CHECK{"入力あり?"}
    F_END_NO["終了"]
    F_USER["st.chat_message('user')<br/>└→ st.markdown(prompt)"]
    F_HISTORY["chat_history に追加<br/>{role: 'user', content: prompt}"]
    F_ASSIST["st.chat_message('assistant')"]
    F_TRY["try:"]
    F_AGENT["run_agent_turn(chat_session, prompt)<br/>↓<br/>response_text"]
    F_RESP_CHECK{"response_text<br/>存在?"}
    F_DISPLAY["markdown表示<br/>履歴追加"]
    F_WARNING["st.warning<br/>'応答なし'"]
    F_EXCEPT["except:<br/>st.error('エラー発生')<br/>logger.error(...)"]

    F_INPUT --> F_CHECK
    F_CHECK -->|No| F_END_NO
    F_CHECK -->|Yes| F_USER
    F_USER --> F_HISTORY
    F_HISTORY --> F_ASSIST
    F_ASSIST --> F_TRY
    F_TRY --> F_AGENT
    F_AGENT --> F_RESP_CHECK
    F_RESP_CHECK -->|Yes| F_DISPLAY
    F_RESP_CHECK -->|No| F_WARNING
    F_TRY -.-> F_EXCEPT
```

---

### 4.3 run_agent_turn() 詳細フロー図

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#1a1a1a', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#666666', 'lineColor': '#888888', 'secondaryColor': '#2a2a2a', 'tertiaryColor': '#1a1a1a'}}}%%
flowchart TB
    START([START])
    SEND["chat_session.send_message(user_input)"]
    INIT["max_turns = 10<br/>turn_count = 0<br/>thought_log = []"]

    subgraph Phase1["Phase 1: ReAct ループ"]
        LOOP_START{"while<br/>turn_count < max_turns"}
        INCREMENT["turn_count += 1<br/>function_call_found = False"]

        subgraph PartLoop["for part in response.parts:"]
            TEXT_CHECK{"part.text?"}
            TEXT_LOG["思考ログに追加<br/>'Thought:' 検出"]
            FC_CHECK{"part.function_call?"}
            FC_FOUND["function_call_found = True"]
            FC_EXTRACT["ツール名・引数抽出"]
            FC_EXEC["TOOLS_MAP から実行<br/>(with st.spinner)"]
            FC_LOG["結果をログに記録"]
            FC_NO_RESULT{"[[NO_RAG_RESULT]]?"}
            FC_UNANSWERED["log_unanswered_question"]
            FC_RESPONSE["function_response 送信"]
        end

        NO_FC{"function_call_found?"}
        SET_FINAL["final_response_text = current_text<br/>break"]
    end

    subgraph Phase2["Phase 2: Reflection 自己洗練"]
        FINAL_CHECK{"final_text<br/>存在?"}
        SKIP["後処理へ"]
        SPINNER["with st.spinner:<br/>'回答を推敲中...'"]
        REFLECT_SEND["REFLECTION_INSTRUCTION +<br/>回答案を送信"]
        SPLIT["'Final Answer:'で分割<br/>- reflection_thought<br/>- reflection_answer"]
        UPDATE["final_response_text<br/>= reflection_answer"]
    end

    subgraph PostProcess["後処理"]
        EXPANDER["thought_log を<br/>st.expander で表示"]
        TAG_PROCESS["'Answer:' タグ処理<br/>- あれば抽出<br/>- なければ整形"]
    end

    RETURN["return final_response_text"]
    END_NODE([END])

    START --> SEND
    SEND --> INIT
    INIT --> LOOP_START
    LOOP_START -->|Yes| INCREMENT
    INCREMENT --> TEXT_CHECK
    TEXT_CHECK -->|Yes| TEXT_LOG
    TEXT_CHECK -->|No| FC_CHECK
    TEXT_LOG --> FC_CHECK
    FC_CHECK -->|Yes| FC_FOUND
    FC_CHECK -->|No| NO_FC
    FC_FOUND --> FC_EXTRACT
    FC_EXTRACT --> FC_EXEC
    FC_EXEC --> FC_LOG
    FC_LOG --> FC_NO_RESULT
    FC_NO_RESULT -->|Yes| FC_UNANSWERED
    FC_NO_RESULT -->|No| FC_RESPONSE
    FC_UNANSWERED --> FC_RESPONSE
    FC_RESPONSE --> LOOP_START
    NO_FC -->|No| SET_FINAL
    NO_FC -->|Yes| LOOP_START
    LOOP_START -->|No| FINAL_CHECK
    SET_FINAL --> FINAL_CHECK

    FINAL_CHECK -->|Yes| SPINNER
    FINAL_CHECK -->|No| SKIP
    SPINNER --> REFLECT_SEND
    REFLECT_SEND --> SPLIT
    SPLIT --> UPDATE
    UPDATE --> EXPANDER
    SKIP --> EXPANDER
    EXPANDER --> TAG_PROCESS
    TAG_PROCESS --> RETURN
    RETURN --> END_NODE

    style START fill:#1a1a1a,stroke:#666,color:#fff
    style END_NODE fill:#1a1a1a,stroke:#666,color:#fff
    style SEND fill:#1a1a1a,stroke:#666,color:#fff
    style INIT fill:#1a1a1a,stroke:#666,color:#fff
    style LOOP_START fill:#1a1a1a,stroke:#666,color:#fff
    style INCREMENT fill:#1a1a1a,stroke:#666,color:#fff
    style TEXT_CHECK fill:#1a1a1a,stroke:#666,color:#fff
    style TEXT_LOG fill:#1a1a1a,stroke:#666,color:#fff
    style FC_CHECK fill:#1a1a1a,stroke:#666,color:#fff
    style FC_FOUND fill:#1a1a1a,stroke:#666,color:#fff
    style FC_EXTRACT fill:#1a1a1a,stroke:#666,color:#fff
    style FC_EXEC fill:#1a1a1a,stroke:#666,color:#fff
    style FC_LOG fill:#1a1a1a,stroke:#666,color:#fff
    style FC_NO_RESULT fill:#1a1a1a,stroke:#666,color:#fff
    style FC_UNANSWERED fill:#1a1a1a,stroke:#666,color:#fff
    style FC_RESPONSE fill:#1a1a1a,stroke:#666,color:#fff
    style NO_FC fill:#1a1a1a,stroke:#666,color:#fff
    style SET_FINAL fill:#1a1a1a,stroke:#666,color:#fff
    style FINAL_CHECK fill:#1a1a1a,stroke:#666,color:#fff
    style SKIP fill:#1a1a1a,stroke:#666,color:#fff
    style SPINNER fill:#1a1a1a,stroke:#666,color:#fff
    style REFLECT_SEND fill:#1a1a1a,stroke:#666,color:#fff
    style SPLIT fill:#1a1a1a,stroke:#666,color:#fff
    style UPDATE fill:#1a1a1a,stroke:#666,color:#fff
    style EXPANDER fill:#1a1a1a,stroke:#666,color:#fff
    style TAG_PROCESS fill:#1a1a1a,stroke:#666,color:#fff
    style RETURN fill:#1a1a1a,stroke:#666,color:#fff
    style Phase1 fill:#2a2a2a,stroke:#666,color:#fff
    style Phase2 fill:#2a2a2a,stroke:#666,color:#fff
    style PostProcess fill:#2a2a2a,stroke:#666,color:#fff
    style PartLoop fill:#333333,stroke:#666,color:#fff
```

---

## 5. 状態管理（st.session_state）

| キー | 型 | 説明 |
|------|-----|------|
| `chat_history` | `List[Dict]` | チャット履歴。各要素は `{"role": str, "content": str}` |
| `chat_session` | `ChatSession` | Gemini の ChatSession インスタンス |
| `current_collections` | `List[str]` | 現在選択されているコレクション名リスト |
| `current_model` | `str` | 現在選択されているモデル名 |

---

## 6. 外部依存関係

### 6.1 インポートモジュール

| モジュール | 用途 |
|-----------|------|
| `streamlit` | Web UI フレームワーク |
| `google.generativeai` | Gemini API クライアント |
| `qdrant_client` | Qdrant ベクトル DB クライアント |
| `pandas` | データフレーム処理 |
| `config.AgentConfig` | エージェント設定 |
| `config.GeminiConfig` | Gemini モデル設定 |
| `agent_tools` | RAG 検索ツール |
| `services.qdrant_service` | Qdrant サービス |
| `services.log_service` | ログ記録サービス |

### 6.2 環境変数

| 変数名 | 必須 | 説明 |
|--------|------|------|
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Yes | Gemini API キー |
| `QDRANT_URL` | No | Qdrant サーバー URL (デフォルト: `http://localhost:6333`) |

---

## 7. エラーハンドリング

| 箇所 | エラー種別 | 処理 |
|------|-----------|------|
| `get_available_collections_from_qdrant` | Qdrant 接続エラー | 空リストを返却、ログ出力 |
| `setup_agent` | API キー未設定 | `st.error` + `ValueError` 送出 |
| `run_agent_turn` | ツール実行エラー | エラーメッセージを `tool_result` に設定 |
| `run_agent_turn` | Reflection エラー | Draft をそのまま使用、ログ出力 |
| `show_agent_chat_page` | エージェント初期化失敗 | `st.error` + 早期リターン |
| `show_agent_chat_page` | チャット処理エラー | `st.error` + ログ出力 |

---

## 8. ログ出力

| ログレベル | 出力内容 |
|-----------|---------|
| `INFO` | Agent Thought, Agent Response, Agent Tool Call, Tool Result, Reflection |
| `ERROR` | Qdrant 接続失敗, Reflection フェーズエラー, チャット処理エラー |

---

*Generated: 2024*