コマンド:                                                                                                                            │
 /about - バージョン情報を表示
 /auth - 認証方法を変更
 /bug - バグ報告を送信
 /chat - 会話履歴を管理
   list - 保存された会話チェックポイントを一覧表示
   save - 現在の会話をチェックポイントとして保存。使用法: /chat save <タグ> 
   resume - チェックポイントから会話を再開します。使用法: /chat resume <タグ>
   delete - 会話チェックポイントを削除します。使用法: /chat delete <タグ>  
   share - 現在の会話をマークダウンまたはjsonファイルに共有。使用法: /chat share <ファイル名>  
 /clear - 画面と会話履歴をクリア
 /compress - コンテキストを要約に置き換えて圧縮
 /copy - 最後の結果またはコードスニペットをクリップボードにコピー
 /docs - ブラウザでGemini CLIの完全なドキュメントを開く
 /directory - ワークスペースディレクトリを管理 
   add - ワークスペースにディレクトリを追加。複数パスはカンマ区切り
   show - ワークスペース内の全ディレクトリを表示
 /editor - 外部エディタの設定を指定
 /extensions - 拡張機能を管理
   list - アクティブな拡張機能を一覧表示 
   update - 拡張機能を更新します。使用法: update <拡張機能名>--all  
   explore - ブラウザで拡張機能ページを開きます  
   restart - すべての拡張機能を再起動します
 /help - gemini-cli のヘルプを表示  
 /ide - IDE 統合を管理 
 /init - プロジェクトを分析し、カスタマイズされた GEMINI.md ファイルを作成 
 /mcp - 設定済みの Model Context Protocol (MCP) サーバーを管理  
   list - 設定済みの MCP サーバーとツールを一覧表示 
   desc - 設定済みMCPサーバーとツールを説明付きで一覧表示 
   schema - 設定済みMCPサーバーとツールを説明とスキーマ付きで一覧表示 
   auth - OAuth対応MCPサーバーで認証 
   refresh - MCPサーバーを再起動 
 /memory - メモリ操作コマンド
   show - 現在のメモリ内容を表示  
   add - メモリにコンテンツを追加  
   refresh - ソースからメモリを更新  
   list - 使用中の GEMINI.md ファイルのパス一覧を表示
 /model - モデル設定ダイアログを開く  
 /privacy - プライバシー通知を表示   
 /quit - CLIを終了  
 /resume - 自動保存された会話を閲覧・再開  
 /stats - セッション統計を確認。使用法: /stats [sessionmodeltools] 
   session - セッション固有の使用統計を表示  
   model - モデル固有の使用統計を表示  
   tools - ツール固有の使用統計を表示  
 /theme - テーマを変更  
 /tools - 利用可能なGemini CLIツールを一覧表示。使用法: /tools [desc]
 /settings - Gemini CLI設定の表示と編集  
 /vim - vimモードのオン/オフ切り替え  
 /setup-github - GitHub Actionsの設定   
 /terminal-setup - 複数行入力用ターミナルキーバインディングの設定（VS Code、Cursor、Windsurf）  
 ! - シェルコマンド  
[MCP] - モデルコンテキストプロトコルコマンド（外部サーバーから） 


# Gemini3 Hybrid RAG Agent - プロジェクトコンテキスト

## 1. プロジェクト概要
本プロジェクトは、一般的な会話と専門的な知識検索をインテリジェントに切り替えることができる **Hybrid RAG (Retrieval-Augmented Generation) エージェント** を実装するものです。日本語および英語のドキュメントを処理してQ/Aペアを生成し、**Qdrant** ベクトルデータベースに保存して、**Google Gemini 2.0 Flash** を推論エンジンとして使用するように設計されています。

**主な機能:**
*   **ReAct エージェント:** Gemini 2.0 を使用して推論 (`[🧠 Thought]`) を行い、いつツールを呼び出すか (`[🛠️ Tool Call]`) を決定します。
*   **RAG パイプライン:** 自動チャンク分割 (SemanticCoverage)、Q/Aペア生成 (LLMベース)、およびベクトル埋め込み。
*   **ハイブリッドアーキテクチャ:** OpenAI (レガシー/代替) と Gemini (現在の主軸) の両方のモデルを、埋め込みと生成でサポートします。
*   **スケーラビリティ:** **Celery + Redis** を使用して、大規模データセットの並列処理を実現します。

## 2. アーキテクチャと技術スタック

### コアコンポーネント
*   **LLM:** Google Gemini 2.0 Flash (推論/チャット), GPT-4o (オプション/レガシー)。
*   **ベクトルデータベース:** Qdrant (Docker経由のローカル実行)。
*   **Embedding:** `text-embedding-004` (Gemini) または `text-embedding-3-small` (OpenAI)。
*   **タスクキュー:** Celery と Redis ブローカー (非同期Q/A生成用)。
*   **UI/インターフェース:**
    *   **CLI:** `agent_main.py` (対話型エージェントターミナル)。
    *   **Web UI:** `rag_qa_pair_qdrant.py` (Streamlit ダッシュボード)。
        *注: Streamlit UIの説明ページ (`ui/pages/explanation_page.py`) のMermaidダイアグラムは、黒背景・白テキスト・白ボーダーでスタイリングされています。*

### データフロー
1.  **取り込み:** ドキュメント (cc_news, livedoor, wikipedia) をロードし、前処理を行います。
2.  **チャンク分割:** テキストを `SemanticCoverage` (段落認識 + 日本語用MeCab) を使用して分割します。
3.  **Q/A生成:** LLMがチャンクからQ/Aペアを生成します (Celeryによる非同期処理)。
4.  **埋め込みと保存:** Q/Aペアを埋め込み化し、Qdrantコレクションに保存します。
5.  **検索 (エージェント):** エージェントはユーザークエリを受け取り、RAGが必要かを判断し、Qdrantを検索して回答を合成します。

## 3. 主要ファイルとディレクトリ

| ファイル/ディレクトリ | 説明 |
| :--- | :--- |
| **`agent_main.py`** | **エントリーポイント (CLI):** ReActエージェントのメイン対話ループ。 |
| **`rag_qa_pair_qdrant.py`** | **エントリーポイント (GUI):** データ管理、生成、検索のためのStreamlitアプリ。 |
| `agent_tools.py` | エージェントが利用可能なツール (関数) を定義 (例: `search_rag_knowledge_base`)。 |
| `celery_tasks.py` | 非同期Celeryタスク (Q/A生成) の定義。 |
| `helper_rag.py` | チャンク分割やテキスト処理を含む、RAGのコアロジック。 |
| `helper_llm.py` | LLM API呼び出し (Gemini/OpenAI) の統一ラッパー。 |
| `qdrant_client_wrapper.py` | Qdrantデータベース操作の抽象化レイヤー。 |
| `config.py` / `config.yml` | 設定 (パス、モデル名、APIキー)。 |
| `doc/` | 詳細なドキュメント (インストール、仕様、アーキテクチャ)。 |
| `docker-compose/` | QdrantおよびRedisサービスのセットアップ。 |

## 4. セットアップと使用方法

### 前提条件
*   Python 3.10以上
*   Docker & Docker Compose (Qdrant/Redis用)
*   APIキー (Gemini, OpenAI) を `.env` に設定

### サービスの起動
```bash
# Qdrant と Redis を起動
docker-compose -f docker-compose/docker-compose.yml up -d

# (オプション) バックグラウンド処理用のCeleryワーカーを起動
./start_celery.sh start -w 8
```

### アプリケーションの実行
**オプション 1: CLI エージェント (対話型)**
```bash
python agent_main.py
```
*エージェントの推論やツール使用のテストに使用します。*

**オプション 2: Streamlit ダッシュボード (管理用)**
```bash
streamlit run rag_qa_pair_qdrant.py
```
*データの取り込み、Q/A生成、ベクトルDBの確認に使用します。*

## 5. 開発ガイドライン

*   **コードスタイル:** PEP 8 に準拠してください。可能であれば `ruff` でリントを行ってください。
*   **型ヒント:** 新しい関数、特に `services/` や `helper_*.py` 内のものには、型ヒントの記述を強く推奨します。
*   **ロギング:** 標準の `logging` モジュールを使用してください。エージェントは詳細なトレースを `logs/` に記録します。
*   **テスト:** `pytest` を使用してテストを実行してください。
    ```bash
    pytest tests/
    ```
*   **規約:** RAGパイプラインを変更する場合、同期 (ローカル) モードと非同期 (Celery) 実行モードの両方と互換性があることを確認してください。
*   **Mermaidダイアグラム:** ドキュメント用のMermaidダイアグラムを作成する際は、**シンプルでバージョン9互換の構文**を使用してください。
    *   **問題:** PyCharm ProfessionalのMarkdownビューアーは、古いバージョンのMermaid (v9) を使用しており、他のツール (例: Typora) では正常にレンダリングされる新しい機能でも、頻繁に構文エラーが発生します。
    *   **要件:** IDE内での正しいレンダリングを確保するために、常にシンプルなグラフ構造と基本的な構文を使用してください。
    *   **推奨:**
        - 基本的な `graph TD`, `graph LR`, `sequenceDiagram`, `flowchart` 構造を使用
        - ノードラベルはシンプルに（特殊文字を避ける）
        - 標準的な矢印構文を使用: `-->`, `---`, `-.->`, `==>`
    *   **禁止:**
        - `:::` クラス割り当てやインラインスタイル
        - 複雑なネストを持つ `subgraph`
        - ダイアグラムブロック内の `%%` コメント
        - `&` による並列パスや `@{...}` アノテーションなどの新機能
    *   **良い例:**
        ```mermaid
        graph TD
            A[Start] --> B[Process]
            B --> C{Decision}
            C -->|Yes| D[End]
            C -->|No| B
        ```
