# Services モジュール詳細設計書

## 1. 概要
`services/` ディレクトリは、アプリケーションのビジネスロジックを集約・分離したモジュール群です。
UI層（Streamlit）やCLIツールから呼び出され、外部API（Gemini/OpenAI）、データベース（Qdrant）、ファイルシステムとの対話を抽象化して提供します。

## 2. モジュール構成

| ファイル名 | 役割・責務 |
| :--- | :--- |
| **`agent_service.py`** | **[Core]** ReAct + Reflection パターンを用いた自律型エージェントの対話ロジック。 |
| **`qdrant_service.py`** | **[Core]** ベクトルDB (Qdrant) の操作全般（検索、登録、統合、管理）。 |
| **`qa_service.py`** | Q/Aペア生成ロジック。Gemini API または サブプロセスによるバッチ生成。 |
| **`dataset_service.py`** | データセット（HuggingFace, Livedoor, ファイルアップロード）のダウンロードと前処理。 |
| **`file_service.py`** | ファイルI/O、履歴管理、CSV/JSON操作のヘルパー。 |
| **`config_service.py`** | 設定 (`config.yml` + 環境変数) の一元管理。シングルトン。 |
| **`token_service.py`** | トークン数のカウント（tiktoken）とコスト試算。 |
| **`cache_service.py`** | メモリキャッシュ機能（TTL対応）。 |
| **`json_service.py`** | JSONの安全なシリアライズ・デシリアライズ。 |
| **`log_service.py`** | 未回答質問などのログ記録・管理。 |
| **`prompts.py`** | エージェントやシステムで使用する共通プロンプト定義。 |

---

## 3. IPO設計詳細

### 3.1 Agent Service (`agent_service.py`)

**ReActAgent クラス**

*   **Input**:
    *   `user_input` (str): ユーザーからの自然言語の質問
    *   `selected_collections` (List[str]): 検索対象のQdrantコレクション名リスト
    *   `model_name` (str): 使用するLLMモデル名 (例: `gemini-2.0-flash`)
*   **Process**:
    1.  **キーワード抽出**: 入力から重要キーワードを抽出（MeCab/Regex）。
    2.  **ReActループ**:
        *   LLMに思考と行動（ツール呼び出し）を促す。
        *   ツール（RAG検索）を実行し、結果（コンテキスト）をLLMに戻す。
        *   情報を統合してドラフト回答を生成する。
    3.  **Reflectionフェーズ**:
        *   ドラフト回答を自己評価（正確性、適切性、スタイル）。
        *   必要に応じて修正し、最終回答を生成する。
    
    ```mermaid
    graph TD
        User[User Input] -->|text| KeywordExtract[Keyword Extraction]
        KeywordExtract -->|Augmented Input| ReActLoop{ReAct Loop}
        
        subgraph "Phase 1: ReAct"
            ReActLoop -->|Prompt| LLM[LLM API]
            LLM -->|Response| Check{Action?}
            Check -->|Tool Call| Tool[RAG Search]
            Tool -->|Result| LLM
            Check -->|Final Text| Draft[Draft Answer]
        end
        
        Draft -->|Prompt| Reflection{Reflection}
        
        subgraph "Phase 2: Reflection"
            Reflection -->|Self-Critique| Refined[Refined Answer]
        end
        
        Refined -->|Yield| Output[Final Response]
    ```
*   **Output**:
    *   `Generator`: 進捗イベント（ログ、ツール実行、最終回答）を順次生成。
    *   最終的にはユーザーへの回答テキスト。

### 3.2 Qdrant Service (`qdrant_service.py`)

**Qdrant操作全般**

*   **Input**:
    *   **検索**: `query` (str), `collection_name` (str)
    *   **登録**: `df` (DataFrame), `collection_name` (str)
    *   **統合**: `source_collections` (List[str]), `target_collection` (str)
*   **Process**:
    *   **検索**: クエリをEmbedding APIでベクトル化し、Qdrantの `search` APIを叩く。
    *   **登録**: テキストをバッチ分割し、Embedding APIでベクトル化。`PointStruct`を作成して `upsert`。
    *   **統合**: 複数コレクションから `scroll` で全データを取得し、IDを再計算して新コレクションへ登録。
    
    ```mermaid
    graph LR
        Query[Search Query] -->|Embed| Vector[Vector]
        Vector -->|Search| Qdrant[(Qdrant DB)]
        Qdrant -->|Points| Results[Search Results]
        
        Docs[Documents] -->|Batch & Embed| Vectors[Vectors]
        Vectors -->|Upsert| Qdrant
    ```
*   **Output**:
    *   **検索**: 検索結果のリスト（スコア、ペイロード付き）。
    *   **登録/統合**: 処理成功件数、ステータス。

### 3.3 QA Service (`qa_service.py`)

**Q/Aペア生成**

*   **Input**:
    *   `text` (str): ソースとなるテキストチャンク
    *   `chunk_id` (str): チャンクID
    *   `model` (str): LLMモデル名
*   **Process**:
    1.  教育用Q/A生成の専門家としてのプロンプトを構築。
    2.  Gemini APIの構造化出力（`response_schema`）を使用してQ/Aペアを生成。
    3.  `QAPair` オブジェクトのリストに変換。
    
    ```mermaid
    graph LR
        Chunk[Text Chunk] -->|Prompt| LLM[Gemini API]
        LLM -->|Structured Output| QAPairs[Q/A Pairs List]
    ```
*   **Output**:
    *   `List[QAPair]`: 生成された質問、回答、メタデータのリスト。

### 3.4 Dataset Service (`dataset_service.py`)

**データセット読み込み**

*   **Input**:
    *   `dataset_name` / `uploaded_file`
    *   設定（分割、サンプル数など）
*   **Process**:
    1.  HuggingFace API または ファイルシステム/アップロードからデータを取得。
    2.  フォーマット（JSON, CSV, Tarball）に応じてパース。
    3.  テキストクリーニング（`clean_text`）とフィールド結合を行い、統一フォーマットのDataFrameを作成。
    
    ```mermaid
    graph TD
        Source[Source Data] -->|Download/Load| RawDF[Raw DataFrame]
        RawDF -->|Extract & Clean| ProcessedDF[Processed DataFrame]
        ProcessedDF -->|Add Combined_Text| Output[Final DataFrame]
    ```
*   **Output**:
    *   `pd.DataFrame`: `Combined_Text` カラムを持つ標準化されたデータフレーム。

### 3.5 Token Service (`token_service.py`)

**トークン管理**

*   **Input**:
    *   `text` (str)
    *   `model` (str)
*   **Process**:
    1.  モデル名に対応する `tiktoken` エンコーディングを取得（例: `gpt-4o` -> `cl100k_base`）。
    2.  テキストをエンコードしてトークン数を算出。
    3.  （コスト推定の場合）トークン数 × 単価で計算。
    
    ```mermaid
    graph LR
        Text[Text] -->|Get Encoding| Tokenizer[Tiktoken]
        Tokenizer -->|Encode| Count[Token Count]
        Count -->|Multiply Price| Cost[Estimated Cost]
    ```
*   **Output**:
    *   `int`: トークン数
    *   `float`: 推定コスト(USD)

### 3.6 Config Service (`config_service.py`)

**設定管理**

*   **Input**:
    *   `config.yml` (ファイル)
    *   環境変数 (`os.environ`)
*   **Process**:
    1.  `config.yml` をロード。
    2.  環境変数（`OPENAI_API_KEY`, `LOG_LEVEL` 等）で値をオーバーライド。
    3.  メモリキャッシュに保持し、ドット記法（`api.timeout`）でのアクセスを提供。
*   **Output**:
    *   設定値（Any）

---