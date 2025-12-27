# UI モジュール詳細設計書

## 1. 概要
本アプリケーションは **Streamlit** を使用したシングルページアプリケーション（SPA）ライクな構成を採用しています。
`agent_rag.py` (または `ui/app.py`) をエントリポイントとし、サイドバーのメニュー選択に応じて `ui/pages/` 配下の各ページモジュールを動的に読み込み・描画します。

## 2. 画面構成と遷移

### 2.1 メインレイアウト
*   **サイドバー**: ナビゲーションメニュー、グローバル設定（モデル選択、APIキー等）
*   **メインエリア**: 選択された機能ページの描画エリア

### 2.2 ページ一覧
| ページID | 表示名 | 対応モジュール | 概要 |
| :--- | :--- | :--- | :--- |
| `explanation` | 📖 説明 | `explanation_page.py` | システム概要、README表示 |
| `agent_chat` | 🤖 エージェント対話 | `agent_chat_page.py` | Legacy ReActエージェントとの対話 |
| `grace_chat` | 🧠 GRACE エージェント | `grace_chat_page.py` | 新アーキテクチャ(GRACE)との対話 |
| `log_viewer` | 📊 未回答ログ | `log_viewer_page.py` | 失敗したクエリの分析・管理 |
| `rag_download` | 📥 RAGデータダウンロード | `download_page.py` | データセット取得・前処理 |
| `qa_generation` | 🤖 Q/A生成 | `qa_generation_page.py` | テキストからのQ/Aペア生成 |
| `qdrant_registration` | 📥 CSVデータ登録 | `qdrant_registration_page.py` | Qdrantへのベクトル登録 |
| `show_qdrant` | 🗄️ Qdrantデータ管理 | `qdrant_show_page.py` | コレクション閲覧・統合・削除 |
| `qdrant_search` | 🔎 Qdrant検索 | `qdrant_search_page.py` | ベクトル検索デバッグ |

---

## 3. モジュール詳細とIPO

### 3.1 エントリポイント (`ui/app.py` / `agent_rag.py`)

*   **役割**: アプリケーションの初期化、ページルーティング。
*   **Process**:
    1.  `st.set_page_config` で基本設定（タイトル、アイコン）。
    2.  `st.sidebar` でメニューを表示し、ユーザーの選択を取得。
    3.  選択されたページに対応する関数（例: `show_agent_chat_page()`）を実行。

### 3.2 Agent Chat Page (`ui/pages/agent_chat_page.py`)

*   **概要**: Legacy ReAct Agent とのチャットインターフェース。
*   **Input**:
    *   ユーザーメッセージ (`st.chat_input`)
    *   選択されたコレクション (`st.multiselect`)
    *   使用モデル (`st.selectbox`)
*   **Process**:
    1.  **初期化**: `st.session_state` でチャット履歴とAgentインスタンスを保持。設定変更時に再初期化。
    2.  **対話ループ**:
        *   ユーザー入力を受け取り、履歴に追加。
        *   `agent.execute_turn(prompt)` ジェネレータを呼び出し。
        *   イベント（Thought, Tool Call, Tool Result）を逐次受信し、Expander内にストリーミング表示。
    3.  **回答表示**: 最終回答 (`final_answer` イベント) をチャットエリアに表示。
    
    ```mermaid
    graph TD
        User[User Input] -->|Prompt| Handler{Session State Agent?}
        Handler -->|No| Init[Init ReActAgent]
        Init --> Execute
        Handler -->|Yes| Execute[Agent.execute_turn]
        
        Execute -->|Yield Event| Stream[Streamlit Display]
        Stream -->|Log/Thought| Expander[Thought Process Expander]
        Stream -->|Final Answer| Chat[Chat Message Area]
        
        Chat --> History[Update Session History]
    ```
*   **Output**:
    *   チャットUI（メッセージ、思考プロセスログ）。

### 3.3 GRACE Chat Page (`ui/pages/grace_chat_page.py`)

*   **概要**: 新しいGRACEアーキテクチャを用いたチャット画面。信頼度表示や介入UIを備える。
*   **Input**:
    *   ユーザーメッセージ
    *   介入レスポンス（確認ボタン、テキスト入力）
*   **Process**:
    1.  **計画生成**: `Planner` を呼び出し、実行計画を作成・表示。
    2.  **実行**: `Executor` をジェネレータモードで実行。
    3.  **イベントハンドリング**:
        *   `step_start`/`step_complete`: タイムライン表示更新。
        *   `confidence_update`: サイドバーの信頼度メーター更新。
        *   `intervention_required`: 実行を一時停止し、介入UI (`display_intervention_request`) を表示。
    4.  **再開**: ユーザー介入後、`execution_state` を更新して再実行 (`st.rerun`)。
    
    ```mermaid
    graph TD
        Input[User Input] --> Plan[Planner: Create Plan]
        Plan --> ExecState[Init ExecutionState]
        ExecState --> ExecLoop{Executor Loop}
        
        ExecLoop -->|Step Event| UI_Step[Update Plan UI]
        ExecLoop -->|Confidence| UI_Conf[Update Sidebar Metric]
        ExecLoop -->|Intervention| Pause[Pause & Show UI]
        
        Pause -->|User Action| Resume[Update State & Rerun]
        Resume --> ExecLoop
        
        ExecLoop -->|Finish| Result[Show Final Answer]
    ```
*   **Output**:
    *   実行計画（ステップ進行状況）。
    *   信頼度メーター（ゲージ、レーダーチャート）。
    *   介入ダイアログ。

### 3.4 Qdrant Search Page (`ui/pages/qdrant_search_page.py`)

*   **概要**: ベクトル検索のデバッグ用画面。
*   **Input**:
    *   検索クエリ
    *   対象コレクション
    *   Top-K
    *   ハイブリッド検索ON/OFF
*   **Process**:
    1.  `get_collection_embedding_params` でコレクションの次元数・モデルを特定。
    2.  クエリを `embed_query_for_search` でベクトル化（必要ならSparseベクトルも生成）。
    3.  `qdrant_client_wrapper.search_collection` を呼び出し。
    4.  結果をDataFrame化して表示。
    5.  最上位結果に対してGeminiで回答生成（RAGシミュレーション）。
    
    ```mermaid
    graph LR
        Query[Query] -->|Embed| Vector
        Vector -->|Search| DB[(Qdrant)]
        DB -->|Hits| Display[Result Table]
        Display -->|Top Hit| GenAI[Gemini Answer Gen]
    ```
*   **Output**:
    *   検索結果テーブル（スコア、ペイロード）。
    *   AI生成回答。

### 3.5 Qdrant Registration Page (`ui/pages/qdrant_registration_page.py`)

*   **概要**: CSVデータをQdrantに登録する画面。
*   **Input**:
    *   CSVファイル（`qa_output/` から選択）
    *   コレクション名
    *   ハイブリッド検索オプション
*   **Process**:
    1.  CSV読み込み (`load_csv_for_qdrant`)。
    2.  コレクション作成 (`create_or_recreate_collection`)。
    3.  **Dense Embedding**: Gemini API等でベクトル化。
    4.  **Sparse Embedding** (Option): FastEmbed等でスパースベクトル化。
    5.  `PointStruct` 構築 (`build_points_for_qdrant`)。
    6.  アップサート (`upsert_points_to_qdrant`)。
    
    ```mermaid
    graph TD
        CSV[Select CSV] --> Load[Load DataFrame]
        Load --> EmbedDense[Dense Embedding]
        Load --> EmbedSparse[Sparse Embedding Opt]
        EmbedDense --> Build[Build Points]
        EmbedSparse --> Build
        Build --> Upsert[Qdrant Upsert]
    ```
*   **Output**:
    *   処理ログ。
    *   登録完了メッセージと件数。

### 3.6 Download Page (`ui/pages/download_page.py`)

*   **概要**: データセットのダウンロードと前処理。
*   **Input**:
    *   データソース（HuggingFace Dataset名 / ローカルファイル）
    *   サンプル数、文字数フィルタ設定
*   **Process**:
    1.  `dataset_service` を使用してデータ取得。
    2.  テキスト抽出・結合・クリーニング (`extract_text_content`)。
    3.  `OUTPUT/` フォルダに CSV/TXT/JSON として保存。
*   **Output**:
    *   前処理済みファイル。
    *   データプレビュー。

### 3.7 QA Generation Page (`ui/pages/qa_generation_page.py`)

*   **概要**: ドキュメントからQ/Aペアを生成。
*   **Input**:
    *   入力ソース（前処理済みデータ）
    *   Celeryワーカー数、バッチサイズ設定
*   **Process**:
    1.  `run_advanced_qa_generation` を呼び出し。
    2.  （内部で `make_qa.py` / `qa_generator_runner.py` が動作）
    3.  ドキュメントをチャンク分割し、LLMにQ/A生成を依頼。
    4.  進捗バーを更新。
    5.  `qa_output/` に保存。
*   **Output**:
    *   Q/AペアCSV/JSON。
    *   カバレッジレポート。

### 3.8 Log Viewer Page (`ui/pages/log_viewer_page.py`)

*   **概要**: 未回答ログの確認。
*   **Input**:
    *   フィルタリング条件（テキスト検索）
*   **Process**:
    1.  `logs/unanswered_questions.csv` を読み込み。
    2.  フィルタリング適用。
    3.  DataFrameとして表示。
*   **Output**:
    *   ログテーブル。
    *   CSVダウンロードボタン。

---

## 4. 共通コンポーネント (`ui/components/`)

### `rag_components.py`
*   **`select_model`**: サイドバーでのモデル選択ドロップダウン。
*   **`show_model_info`**: 選択されたモデルのスペック・価格表示。
*   **`estimate_token_usage`**: データフレームからトークン数とコストを試算。

### `grace_components.py`
*   **`display_execution_plan`**: GRACEの実行計画を視覚的に表示（ステップごとのステータス色分け）。
*   **`display_confidence_metric`**: 信頼度スコアと内訳グラフ（Plotly）の描画。
*   **`display_intervention_request`**: ユーザー介入用UI（ボタン、テキスト入力）のレンダリング。
