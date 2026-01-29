# tools.py - ツール定義モジュール ドキュメント

**Version 1.1** | 最終更新: 2025-01-29

---

## 目次

1. [概要](#概要)
2. [アーキテクチャ構成図](#1-アーキテクチャ構成図)
   - [システム全体構成](#11-システム全体構成)
   - [データフロー](#12-データフロー)
3. [モジュール構成図](#2-モジュール構成図)
   - [内部モジュール構成](#21-内部モジュール構成)
   - [外部依存関係](#22-外部依存関係)
   - [内部依存モジュール](#23-内部依存モジュール)
4. [クラス・関数一覧表](#3-クラス関数一覧表)
   - [データクラス一覧](#31-データクラス一覧)
   - [クラス一覧](#32-クラス一覧)
   - [ファクトリ関数一覧](#33-ファクトリ関数一覧)
5. [クラス・関数 IPO詳細](#4-クラス関数-ipo詳細)
   - [ToolResult データクラス](#41-toolresult-データクラス)
   - [BaseTool クラス（抽象基底）](#42-basetool-クラス抽象基底)
   - [RAGSearchTool クラス](#43-ragsearchtool-クラス)
   - [ReasoningTool クラス](#44-reasoningtool-クラス)
   - [AskUserTool クラス](#45-askusertool-クラス)
   - [ToolRegistry クラス](#46-toolregistry-クラス)
   - [ファクトリ関数](#47-ファクトリ関数)
6. [外部カスタムモジュール IPO詳細](#5-外部カスタムモジュール-ipo詳細) ⭐ NEW
   - [qdrant_client_wrapper](#51-qdrant_client_wrapper)
   - [services.qdrant_service](#52-servicesqdrant_service)
   - [agent_tools](#53-agent_tools)
   - [regex_mecab](#54-regex_mecab)
7. [設定・定数](#6-設定定数)
8. [使用例](#7-使用例)
   - [ToolRegistryを使用した基本ワークフロー](#71-toolregistryを使用した基本ワークフロー)
   - [RAG検索の直接実行](#72-rag検索の直接実行)
   - [推論ツールの使用](#73-推論ツールの使用)
   - [AskUserToolの使用](#74-askusertoolの使用)
9. [エクスポート](#8-エクスポート)
10. [変更履歴](#9-変更履歴)
11. [付録: 依存関係図](#付録-依存関係図)
12. [関連ドキュメント](#関連ドキュメント)

---

## 概要

`tools.py`は、GRACEエージェントが使用するツール（RAG検索、推論、ask_user等）を定義するモジュールです。各ツールは統一されたインターフェース（`BaseTool`）を実装し、`ToolRegistry`を通じて管理・実行されます。

### 主な責務

- ツールの統一インターフェース定義（BaseTool抽象基底クラス）
- RAG検索ツールによるQdrantベクトルDB検索
- LLM推論ツールによる回答生成
- ユーザー質問ツールによるHITL（Human-in-the-Loop）サポート
- ツールレジストリによるツールの一元管理

### 主要機能一覧

| 機能 | 説明 |
|------|------|
| `ToolResult` | ツール実行結果を保持するデータクラス |
| `BaseTool` | すべてのツールの抽象基底クラス |
| `RAGSearchTool` | Qdrantベクトルデータベースからの検索 |
| `RAGSearchTool.execute()` | コレクション自動フォールバック付きRAG検索 |
| `ReasoningTool` | 検索結果を元にした回答生成 |
| `ReasoningTool.execute()` | LLMによる推論・回答生成 |
| `AskUserTool` | ユーザーへの質問・確認要求 |
| `AskUserTool.execute()` | HITL用の質問情報生成 |
| `ToolRegistry` | ツールの登録・取得・実行を一元管理 |
| `ToolRegistry.execute()` | ツール名を指定した実行 |
| `create_tool_registry()` | ToolRegistryのファクトリ関数 |

---

## 1. アーキテクチャ構成図

### 1.1 システム全体構成

```
┌─────────────────────────────────────────────────────────────────┐
│                        Executor 層                              │
│  ┌──────────────────┐                                          │
│  │     Executor     │                                          │
│  │   (計画実行)     │                                          │
│  └────────┬─────────┘                                          │
└───────────┼────────────────────────────────────────────────────┘
            │ execute(tool_name, **kwargs)
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        tools.py                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  ToolRegistry                                              │ │
│  │    ├── register(tool)                                      │ │
│  │    ├── get(name) → BaseTool                                │ │
│  │    ├── list_tools() → List[str]                            │ │
│  │    └── execute(name, **kwargs) → ToolResult                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│  ┌──────────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │  RAGSearchTool   │ │ ReasoningTool│ │  AskUserTool │        │
│  │  (rag_search)    │ │ (reasoning)  │ │  (ask_user)  │        │
│  └────────┬─────────┘ └──────┬───────┘ └──────────────┘        │
└───────────┼──────────────────┼──────────────────────────────────┘
            │                  │
            ▼                  ▼
┌─────────────────────┐  ┌─────────────────────┐
│     Qdrant          │  │    Gemini API       │
│  (ベクトルDB)       │  │    (LLM)            │
└─────────────────────┘  └─────────────────────┘
```

### 1.2 データフロー

**RAG検索フロー**:
1. Executor が `ToolRegistry.execute("rag_search", query=...)` を呼び出し
2. RAGSearchTool がコレクション候補を決定
3. 各コレクションを順次検索（フォールバック付き）
4. 検索結果を `ToolResult` として返却

**推論フロー**:
1. Executor が `ToolRegistry.execute("reasoning", query=..., sources=...)` を呼び出し
2. ReasoningTool がプロンプトを構築
3. Gemini API に送信し回答を生成
4. 生成結果を `ToolResult` として返却

---

## 2. モジュール構成図

### 2.1 内部モジュール構成

```
tools.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[データクラス]
  └── ToolResult                - ツール実行結果

[抽象基底クラス]
  └── BaseTool                  - ツールの共通インターフェース
        ├── name: str
        ├── description: str
        └── execute(**kwargs) → ToolResult [abstract]

[具象ツールクラス]
  ├── RAGSearchTool             - RAG検索ツール
  │     ├── __init__(config, qdrant_url)
  │     ├── client (property)
  │     ├── execute(query, collection, limit, score_threshold)
  │     ├── _get_all_collections_dynamic()
  │     └── _calculate_confidence_factors(scores)
  │
  ├── ReasoningTool             - LLM推論ツール
  │     ├── __init__(config, model_name)
  │     ├── execute(query, context, sources)
  │     └── _build_prompt(query, context, sources)
  │
  └── AskUserTool               - ユーザー質問ツール
        ├── FUNCTION_DECLARATION (class attr)
        └── execute(question, reason, urgency, options)

[レジストリ]
  └── ToolRegistry              - ツール管理
        ├── __init__(config)
        ├── _register_default_tools()
        ├── register(tool)
        ├── get(name)
        ├── list_tools()
        └── execute(name, **kwargs)

[ファクトリ関数]
  └── create_tool_registry(config) → ToolRegistry
```

### 2.2 外部依存関係

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| `qdrant_client` | - | Qdrantベクトルデータベースクライアント |
| `google-genai` | - | Gemini API クライアント |
| `dataclasses` | 標準 | データクラス定義 |
| `abc` | 標準 | 抽象基底クラス |
| `typing` | 標準 | 型ヒント |
| `logging` | 標準 | ログ出力 |

### 2.3 内部依存モジュール

| モジュール | インポート | 用途 |
|-----------|-----------|------|
| `.config` | `get_config`, `GraceConfig` | 設定管理 |

### 2.4 外部カスタムモジュール依存

| モジュール | インポート | 用途 |
|-----------|-----------|------|
| `qdrant_client_wrapper` | `search_collection`, `embed_query_unified`, `embed_sparse_query_unified` | Qdrant検索ラッパー |
| `services.qdrant_service` | `get_collection_embedding_params` | コレクション情報取得 |
| `agent_tools` | `search_rag_knowledge_base_structured` | Legacy Agent検索ロジック |
| `regex_mecab` | `KeywordExtractor` | キーワード抽出 |

**GraceConfigから使用するサブ設定**:

| サブ設定 | 説明 |
|---------|------|
| `config.qdrant.url` | Qdrant接続URL（デフォルト: http://localhost:6333） |
| `config.qdrant.search_priority` | コレクション検索優先順位リスト |
| `config.llm.model` | 使用するLLMモデル（デフォルト: gemini-2.5-flash） |
| `config.llm.temperature` | LLM生成時の温度 |
| `config.llm.max_tokens` | 最大出力トークン数 |
| `config.tools.enabled` | 有効なツールリスト |

---

## 3. クラス・関数一覧表

### 3.1 データクラス一覧

#### ToolResult

| フィールド | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `success` | bool | - | 実行成功フラグ |
| `output` | Any | - | 出力内容 |
| `confidence_factors` | Dict[str, Any] | {} | 信頼度計算用の要素 |
| `error` | Optional[str] | None | エラーメッセージ |
| `execution_time_ms` | Optional[int] | None | 実行時間（ミリ秒） |

### 3.2 クラス一覧

#### BaseTool（抽象基底）

| 属性/メソッド | 型/シグネチャ | 説明 |
|-------------|--------------|------|
| `name` | str | ツール名 |
| `description` | str | ツールの説明 |
| `execute(**kwargs)` | → ToolResult | ツール実行（抽象メソッド） |

#### RAGSearchTool

| メソッド | 概要 |
|---------|------|
| `__init__(config, qdrant_url)` | コンストラクタ |
| `client` (property) | Qdrantクライアント取得（遅延初期化） |
| `execute(query, collection, limit, score_threshold)` | RAG検索実行 |
| `_get_all_collections_dynamic()` | 動的コレクション一覧取得 |
| `_calculate_confidence_factors(scores)` | 信頼度要素計算 |

#### ReasoningTool

| メソッド | 概要 |
|---------|------|
| `__init__(config, model_name)` | コンストラクタ |
| `execute(query, context, sources)` | LLM推論実行 |
| `_build_prompt(query, context, sources)` | プロンプト構築 |

#### AskUserTool

| メソッド | 概要 |
|---------|------|
| `FUNCTION_DECLARATION` (class attr) | Gemini Function Calling用定義 |
| `execute(question, reason, urgency, options)` | ユーザー質問実行 |

#### ToolRegistry

| メソッド | 概要 |
|---------|------|
| `__init__(config)` | コンストラクタ |
| `_register_default_tools()` | デフォルトツール登録 |
| `register(tool)` | ツール登録 |
| `get(name)` | ツール取得 |
| `list_tools()` | 登録ツール名リスト |
| `execute(name, **kwargs)` | ツール実行 |

### 3.3 ファクトリ関数一覧

| 関数名 | 概要 |
|-------|------|
| `create_tool_registry(config)` | ToolRegistryインスタンス作成 |

---

## 4. クラス・関数 IPO詳細

### 4.1 ToolResult データクラス

**概要**: ツール実行結果を保持するデータクラス。すべてのツールがこの形式で結果を返します。

```python
@dataclass
class ToolResult:
    success: bool
    output: Any
    confidence_factors: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
```

**フィールド詳細**:

| フィールド | 型 | 説明 |
|-----------|------|------|
| `success` | bool | ツール実行が成功したかどうか |
| `output` | Any | 出力内容（ツールにより異なる） |
| `confidence_factors` | Dict[str, Any] | 信頼度計算に使用する要素 |
| `error` | Optional[str] | 失敗時のエラーメッセージ |
| `execution_time_ms` | Optional[int] | 実行時間（ミリ秒） |

**ツール別output形式**:

| ツール | output型 | 内容 |
|--------|----------|------|
| `RAGSearchTool` | List[Dict] | 検索結果のリスト |
| `ReasoningTool` | str | 生成された回答文 |
| `AskUserTool` | Dict | 質問情報（question, reason, urgency, options, awaiting_response） |

**戻り値例**:
```python
# RAG検索成功時
ToolResult(
    success=True,
    output=[
        {"score": 0.92, "payload": {"question": "...", "answer": "..."}, "collection": "wikipedia_ja"},
        {"score": 0.85, "payload": {...}, "collection": "wikipedia_ja"}
    ],
    confidence_factors={
        "result_count": 2,
        "avg_score": 0.885,
        "max_score": 0.92,
        "min_score": 0.85,
        "score_variance": 0.00122,
        "used_collection": "wikipedia_ja"
    },
    execution_time_ms=150
)

# 推論成功時
ToolResult(
    success=True,
    output="東京の人口は約1400万人です。",
    confidence_factors={
        "has_sources": True,
        "source_count": 3,
        "answer_length": 120,
        "token_usage": {"input_tokens": 500, "output_tokens": 50}
    },
    execution_time_ms=1200
)

# 失敗時
ToolResult(
    success=False,
    output=None,
    error="No relevant results found in any collection.",
    confidence_factors={"result_count": 0, "avg_score": 0.0},
    execution_time_ms=500
)
```

---

### 4.2 BaseTool クラス（抽象基底）

**概要**: すべてのツールが継承する抽象基底クラス。統一されたインターフェースを提供します。

```python
class BaseTool(ABC):
    name: str = "base_tool"
    description: str = "Base tool"

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass
```

**クラス属性**:

| 属性 | 型 | 説明 |
|------|------|------|
| `name` | str | ツール識別名（ToolRegistryで使用） |
| `description` | str | ツールの説明 |

**抽象メソッド**:

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `execute(**kwargs)` | ToolResult | ツールを実行し結果を返す |

---

### 4.3 RAGSearchTool クラス

Qdrantベクトルデータベースから関連情報を検索するツール。

#### コンストラクタ: `__init__`

**概要**: RAGSearchToolを初期化し、Qdrant接続情報とキーワード抽出器を設定します。

```python
def __init__(
    self,
    config: Optional[GraceConfig] = None,
    qdrant_url: Optional[str] = None
)
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |
| `qdrant_url` | Optional[str] | None | Qdrant接続URL |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]`, `qdrant_url: Optional[str]` |
| **Process** | 1. 設定を取得<br>2. Qdrant URLを設定<br>3. KeywordExtractorを初期化（オプション） |
| **Output** | RAGSearchToolインスタンス |

---

#### プロパティ: `client`

**概要**: Qdrantクライアントを遅延初期化で取得します。

```python
@property
def client(self) -> QdrantClient
```

| 項目 | 内容 |
|------|------|
| **Input** | なし |
| **Process** | クライアントが未初期化なら`QdrantClient(url)`で作成 |
| **Output** | `QdrantClient`: Qdrantクライアント |

---

#### メソッド: `execute`

**概要**: RAG検索を実行します。コレクション自動フォールバック機能付き。

```python
def execute(
    self,
    query: str,
    collection: Optional[str] = None,
    limit: Optional[int] = None,
    score_threshold: Optional[float] = None,
    **kwargs
) -> ToolResult
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `query` | str | - | 検索クエリ |
| `collection` | Optional[str] | None | 検索対象コレクション |
| `limit` | Optional[int] | None | 取得件数上限 |
| `score_threshold` | Optional[float] | None | スコア閾値 |

| 項目 | 内容 |
|------|------|
| **Input** | `query: str`, `collection: Optional[str]`, `limit: Optional[int]`, `score_threshold: Optional[float]` |
| **Process** | 1. 検索対象コレクション候補を決定<br>2. 各コレクションを順次検索<br>3. 結果が見つかったら採用してループ終了<br>4. Dynamic Thresholding（Top1が0.98以上なら他を除外）<br>5. confidence_factorsを計算 |
| **Output** | `ToolResult`: 検索結果 |

**検索フロー**:

```
1. collection指定あり → そのコレクションを最初に検索
2. collection指定なし or 結果なし → 動的コレクション一覧を取得
3. 優先順位に従って順次検索
4. 結果が見つかったらループ終了
5. Dynamic Thresholding適用
6. ToolResultを返却
```

**Dynamic Thresholding**:

| 条件 | 動作 |
|------|------|
| Top1スコア >= 0.98 | 2位以下を除外（ノイズ除去） |
| それ以外 | すべての結果を保持 |

**confidence_factors**:

| キー | 型 | 説明 |
|-----|------|------|
| `result_count` | int | 検索結果数 |
| `avg_score` | float | 平均スコア |
| `max_score` | float | 最大スコア |
| `min_score` | float | 最小スコア |
| `score_variance` | float | スコアの分散 |
| `used_collection` | str | 使用されたコレクション名 |

**戻り値例**:
```python
ToolResult(
    success=True,
    output=[
        {
            "score": 0.92,
            "payload": {
                "question": "東京の人口は？",
                "answer": "東京都の人口は約1400万人です。",
                "source": "統計局データ.pdf"
            },
            "collection": "wikipedia_ja"
        }
    ],
    confidence_factors={
        "result_count": 1,
        "avg_score": 0.92,
        "max_score": 0.92,
        "min_score": 0.92,
        "score_variance": 0.0,
        "used_collection": "wikipedia_ja"
    },
    execution_time_ms=150
)
```

```python
# 使用例
tool = RAGSearchTool()
result = tool.execute(
    query="東京の人口を教えてください",
    collection=None  # 自動フォールバック
)

if result.success:
    for item in result.output:
        print(f"スコア: {item['score']}, 回答: {item['payload'].get('answer')}")
```

---

#### メソッド: `_get_all_collections_dynamic`

**概要**: Qdrantから全コレクション一覧を動的に取得し、優先順位付けして返します。

```python
def _get_all_collections_dynamic(self) -> List[str]
```

| 項目 | 内容 |
|------|------|
| **Input** | なし |
| **Process** | 1. Qdrantから全コレクション取得<br>2. 設定の優先順位リストでソート<br>3. 優先順位リストにないものを後ろに追加 |
| **Output** | `List[str]`: ソートされたコレクション名リスト |

**ソート順序**:
1. `config.qdrant.search_priority` にあるコレクション（順序維持）
2. それ以外のコレクション

**戻り値例**:
```python
["wikipedia_ja", "livedoor", "cc_news", "japanese_text", "custom_collection"]
```

---

#### メソッド: `_calculate_confidence_factors`

**概要**: 検索結果スコアから信頼度計算用の統計情報を算出します。

```python
def _calculate_confidence_factors(self, scores: List[float]) -> Dict[str, Any]
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `scores` | List[float] | - | 検索結果のスコアリスト |

| 項目 | 内容 |
|------|------|
| **Input** | `scores: List[float]` |
| **Process** | 平均、分散、最大、最小を計算 |
| **Output** | `Dict[str, Any]`: 統計情報 |

**戻り値例**:
```python
{
    "result_count": 5,
    "avg_score": 0.82,
    "score_variance": 0.015,
    "max_score": 0.95,
    "min_score": 0.70
}
```

---

### 4.4 ReasoningTool クラス

収集した情報を分析・統合して回答を生成するLLM推論ツール。

#### コンストラクタ: `__init__`

**概要**: ReasoningToolを初期化し、Gemini APIクライアントを設定します。

```python
def __init__(
    self,
    config: Optional[GraceConfig] = None,
    model_name: Optional[str] = None
)
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |
| `model_name` | Optional[str] | None | 使用するモデル名 |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]`, `model_name: Optional[str]` |
| **Process** | 1. 設定を取得<br>2. モデル名を設定<br>3. Gemini Clientを初期化 |
| **Output** | ReasoningToolインスタンス |

---

#### メソッド: `execute`

**概要**: LLM推論を実行し、回答を生成します。

```python
def execute(
    self,
    query: str,
    context: Optional[str] = None,
    sources: Optional[List[Dict]] = None,
    **kwargs
) -> ToolResult
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `query` | str | - | ユーザーの質問 |
| `context` | Optional[str] | None | 追加コンテキスト |
| `sources` | Optional[List[Dict]] | None | 参照ソース（RAG検索結果） |

| 項目 | 内容 |
|------|------|
| **Input** | `query: str`, `context: Optional[str]`, `sources: Optional[List[Dict]]` |
| **Process** | 1. プロンプトを構築<br>2. Gemini APIに送信<br>3. 回答を取得<br>4. トークン使用量を記録 |
| **Output** | `ToolResult`: 生成された回答 |

**confidence_factors**:

| キー | 型 | 説明 |
|-----|------|------|
| `has_sources` | bool | ソースが提供されたか |
| `source_count` | int | ソース数 |
| `answer_length` | int | 回答の文字数 |
| `token_usage` | Dict | トークン使用量 |

**戻り値例**:
```python
ToolResult(
    success=True,
    output="東京都の人口は約1400万人です。総務省統計局のデータによると...",
    confidence_factors={
        "has_sources": True,
        "source_count": 3,
        "answer_length": 250,
        "token_usage": {"input_tokens": 800, "output_tokens": 100}
    },
    execution_time_ms=1500
)
```

```python
# 使用例
tool = ReasoningTool()

# RAG検索結果を使用
sources = [
    {"score": 0.92, "payload": {"question": "...", "answer": "..."}}
]

result = tool.execute(
    query="東京の人口を教えてください",
    sources=sources
)

print(result.output)
```

---

#### メソッド: `_build_prompt`

**概要**: 推論用のプロンプトを構築します。

```python
def _build_prompt(
    self,
    query: str,
    context: Optional[str],
    sources: Optional[List[Dict]]
) -> str
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `query` | str | - | ユーザーの質問 |
| `context` | Optional[str] | - | 追加コンテキスト |
| `sources` | Optional[List[Dict]] | - | 参照ソース |

| 項目 | 内容 |
|------|------|
| **Input** | `query: str`, `context: Optional[str]`, `sources: Optional[List[Dict]]` |
| **Process** | システム指示、参照情報、補足コンテキスト、質問、回答ルールを結合 |
| **Output** | `str`: 構築されたプロンプト |

**プロンプト構造**:

```
1. システム指示
   - ハイブリッド・ナレッジ・エージェントとしての役割
2. 【参照情報】
   - 各ソースの情報（スコア、コレクション、Q&A、出典）
3. 【補足コンテキスト】（任意）
   - 他ステップの結果など
4. 【ユーザーの質問】
5. 【回答の構成ルール】
   - 正確性と誠実さ
   - 判明した事実を優先
   - 出典の明示
   - 丁寧な日本語
   - 捏造禁止
```

---

### 4.5 AskUserTool クラス

ユーザーに追加情報や確認を求めるHITL用ツール。

#### クラス属性: `FUNCTION_DECLARATION`

**概要**: Gemini Function Calling用のツール定義。

```python
FUNCTION_DECLARATION = {
    "name": "ask_user_for_clarification",
    "description": "ユーザーに追加情報を求めるツール...",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "ユーザーへの質問文"},
            "reason": {"type": "string", "description": "なぜこの質問が必要か"},
            "options": {"type": "array", "items": {"type": "string"}, "description": "選択肢"},
            "urgency": {"type": "string", "enum": ["blocking", "optional"], "description": "緊急度"}
        },
        "required": ["question", "reason", "urgency"]
    }
}
```

**使用条件**:
- 質問の意図が曖昧で、複数の解釈が可能
- 必要な情報が検索で見つからない
- 矛盾する情報があり、どちらを優先すべきか不明

---

#### メソッド: `execute`

**概要**: ユーザーへの質問情報を生成します。

```python
def execute(
    self,
    question: str,
    reason: str,
    urgency: str = "blocking",
    options: Optional[List[str]] = None,
    **kwargs
) -> ToolResult
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `question` | str | - | ユーザーへの質問 |
| `reason` | str | - | 質問の理由 |
| `urgency` | str | "blocking" | 緊急度（blocking/optional） |
| `options` | Optional[List[str]] | None | 選択肢リスト |

| 項目 | 内容 |
|------|------|
| **Input** | `question: str`, `reason: str`, `urgency: str`, `options: Optional[List[str]]` |
| **Process** | 質問情報を構造化 |
| **Output** | `ToolResult`: 質問情報（回答待ち状態） |

**urgency値**:

| 値 | 説明 |
|------|------|
| `blocking` | 回答がないと進めない |
| `optional` | 推測で進めることも可能 |

**戻り値例**:
```python
ToolResult(
    success=True,
    output={
        "question": "東京のどの地域について知りたいですか？",
        "reason": "検索結果に複数の地域情報があるため",
        "urgency": "blocking",
        "options": ["23区", "多摩地域", "島嶼部"],
        "awaiting_response": True
    },
    confidence_factors={
        "requires_user_input": True,
        "urgency": "blocking"
    }
)
```

```python
# 使用例
tool = AskUserTool()
result = tool.execute(
    question="どの年度のデータをお探しですか？",
    reason="複数年度のデータが見つかりました",
    urgency="blocking",
    options=["2023年", "2024年", "最新"]
)

# Executorが実際のUI連携を行う
```

---

### 4.6 ToolRegistry クラス

ツールの登録・取得・実行を一元管理するレジストリ。

#### コンストラクタ: `__init__`

**概要**: ToolRegistryを初期化し、デフォルトツールを登録します。

```python
def __init__(self, config: Optional[GraceConfig] = None)
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]` |
| **Process** | 1. 設定を取得<br>2. ツール辞書を初期化<br>3. デフォルトツールを登録 |
| **Output** | ToolRegistryインスタンス |

---

#### メソッド: `_register_default_tools`

**概要**: 設定に基づいてデフォルトツールを登録します。

```python
def _register_default_tools(self)
```

| 項目 | 内容 |
|------|------|
| **Input** | なし |
| **Process** | `config.tools.enabled` に含まれるツールを登録 |
| **Output** | なし |

**登録されるツール**:

| 設定値 | ツールクラス |
|--------|-------------|
| `"rag_search"` | RAGSearchTool |
| `"reasoning"` | ReasoningTool |
| `"ask_user"` | AskUserTool |

---

#### メソッド: `register`

**概要**: ツールをレジストリに登録します。

```python
def register(self, tool: BaseTool)
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `tool` | BaseTool | - | 登録するツール |

| 項目 | 内容 |
|------|------|
| **Input** | `tool: BaseTool` |
| **Process** | `tool.name` をキーとして辞書に登録 |
| **Output** | なし |

```python
# カスタムツールの登録例
class CustomTool(BaseTool):
    name = "custom_tool"
    description = "カスタムツール"

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="OK")

registry = ToolRegistry()
registry.register(CustomTool())
```

---

#### メソッド: `get`

**概要**: 名前でツールを取得します。

```python
def get(self, name: str) -> Optional[BaseTool]
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `name` | str | - | ツール名 |

| 項目 | 内容 |
|------|------|
| **Input** | `name: str` |
| **Process** | 辞書からツールを取得 |
| **Output** | `Optional[BaseTool]`: ツールまたはNone |

---

#### メソッド: `list_tools`

**概要**: 登録されているツール名のリストを取得します。

```python
def list_tools(self) -> List[str]
```

| 項目 | 内容 |
|------|------|
| **Input** | なし |
| **Process** | 登録されたツール名を取得 |
| **Output** | `List[str]`: ツール名リスト |

**戻り値例**:
```python
["rag_search", "reasoning", "ask_user"]
```

---

#### メソッド: `execute`

**概要**: ツール名を指定して実行します。

```python
def execute(self, name: str, **kwargs) -> ToolResult
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `name` | str | - | ツール名 |
| `**kwargs` | Any | - | ツールへの引数 |

| 項目 | 内容 |
|------|------|
| **Input** | `name: str`, `**kwargs` |
| **Process** | 1. ツールを取得<br>2. 見つからなければエラー結果を返す<br>3. ツールの`execute()`を呼び出す |
| **Output** | `ToolResult`: 実行結果 |

**未登録ツールの場合**:
```python
ToolResult(
    success=False,
    output=None,
    error="Unknown tool: invalid_tool"
)
```

```python
# 使用例
registry = ToolRegistry()

# RAG検索
result = registry.execute("rag_search", query="東京の人口")

# 推論
result = registry.execute("reasoning", query="...", sources=[...])

# ユーザー質問
result = registry.execute("ask_user", question="...", reason="...", urgency="blocking")
```

---

### 4.7 ファクトリ関数

#### `create_tool_registry`

**概要**: ToolRegistryインスタンスを作成するファクトリ関数。

```python
def create_tool_registry(config: Optional[GraceConfig] = None) -> ToolRegistry
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]` |
| **Process** | ToolRegistryをインスタンス化 |
| **Output** | `ToolRegistry`: インスタンス |

```python
# 使用例
from grace.tools import create_tool_registry

registry = create_tool_registry()
print(registry.list_tools())
# 出力: ['rag_search', 'reasoning', 'ask_user']
```

---

## 5. 外部カスタムモジュール IPO詳細

tools.py が依存する外部カスタムモジュールの詳細仕様を記述します。

### 5.1 qdrant_client_wrapper

**ファイル**: `qdrant_client_wrapper.py`（1184行）
**概要**: Qdrantベクトルデータベースとの操作を一元管理するユーティリティモジュール

#### 5.1.1 search_collection()

**概要**: コレクションを検索（Dense または Hybrid）。Sparse Vectorエラー時は自動的にDense Vectorのみで再試行。

```python
def search_collection(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    sparse_vector: Optional[models.SparseVector] = None,
    limit: int = 5,
    hybrid_alpha: float = 0.5
) -> List[Dict[str, Any]]
```

| 項目 | 内容 |
|------|------|
| **Input** | `client`: QdrantClientインスタンス |
| | `collection_name`: 検索対象コレクション名 |
| | `query_vector`: クエリの埋め込みベクトル（Dense） |
| | `sparse_vector`: スパースベクトル（Hybrid Search用、省略可） |
| | `limit`: 取得件数上限（デフォルト: 5） |
| | `hybrid_alpha`: ハイブリッド検索のアルファ値（未使用） |
| **Process** | 1. コレクション情報を取得し、名前付きベクトルかどうかを判定 |
| | 2. sparse_vectorがある場合、Hybrid Search（RRF Fusion）を試行 |
| | 3. Sparse Vectorエラー時は自動的にDense Vectorのみで再試行 |
| | 4. Dense Searchを実行（名前付きベクトル対応） |
| | 5. 最終フォールバック: 最もシンプルなquery_points形式 |
| **Output** | `List[Dict[str, Any]]`: 検索結果リスト |

**Output形式**:
```python
[
    {
        "score": 0.92,       # 類似度スコア
        "id": 12345,         # ポイントID
        "payload": {         # メタデータ
            "question": "...",
            "answer": "...",
            "source": "..."
        }
    },
    ...
]
```

**Hybrid Search フロー**:
```
┌─────────────────────────────────────────────────────────┐
│ Hybrid Search (RRF Fusion)                              │
│                                                         │
│  ┌─────────────────┐    ┌─────────────────┐            │
│  │ Dense Prefetch  │    │ Sparse Prefetch │            │
│  │ (query_vector)  │    │ (text-sparse)   │            │
│  │   limit×2       │    │   limit×2       │            │
│  └────────┬────────┘    └────────┬────────┘            │
│           │                      │                      │
│           └──────────┬───────────┘                      │
│                      ▼                                  │
│              ┌─────────────┐                            │
│              │ RRF Fusion  │                            │
│              │  (Ranking)  │                            │
│              └──────┬──────┘                            │
│                     ▼                                   │
│              ┌─────────────┐                            │
│              │   Results   │                            │
│              │   (limit)   │                            │
│              └─────────────┘                            │
└─────────────────────────────────────────────────────────┘
```

#### 5.1.2 embed_query_unified()

**概要**: クエリテキストを埋め込みベクトルに変換（プロバイダー抽象化版）。OpenAIとGeminiの両方に対応。

```python
def embed_query_unified(
    text: str,
    provider: str = None
) -> List[float]
```

| 項目 | 内容 |
|------|------|
| **Input** | `text`: 埋め込むテキスト |
| | `provider`: "gemini" or "openai"（Noneの場合は環境変数デフォルト） |
| **Process** | 1. プロバイダーを決定（デフォルト: 環境変数 `EMBEDDING_PROVIDER`） |
| | 2. `create_embedding_client(provider)` でクライアント作成 |
| | 3. `embed_text(text, task_type="retrieval_query")` を実行 |
| **Output** | `List[float]`: 埋め込みベクトル |

**プロバイダー別次元数**:

| プロバイダー | モデル | 次元数 |
|------------|--------|-------|
| `gemini` | gemini-embedding-001 | 3072 |
| `openai` | text-embedding-3-small | 1536 |
| `fastembed` | BAAI/bge-small-en-v1.5 | 384 |

#### 5.1.3 embed_sparse_query_unified()

**概要**: クエリテキストをSparse Embeddingに変換（キーワードベクトル）

```python
def embed_sparse_query_unified(
    text: str,
    model_name: str = None
) -> models.SparseVector
```

| 項目 | 内容 |
|------|------|
| **Input** | `text`: クエリテキスト |
| | `model_name`: 使用するSparseモデル（省略可） |
| **Process** | 1. `get_sparse_embedding_client(model_name)` でクライアント取得 |
| | 2. `embed_text(text)` でスパースベクトル生成 |
| | 3. Qdrant用 `SparseVector` に変換 |
| **Output** | `models.SparseVector`: スパースベクトル |

**SparseVector構造**:
```python
models.SparseVector(
    indices=[1, 5, 23, 156, ...],  # 非ゼロ要素のインデックス
    values=[0.8, 0.5, 0.3, 0.2, ...]  # 対応する値
)
```

---

### 5.2 services.qdrant_service

**ファイル**: `services/qdrant_service.py`（1066行）
**概要**: Qdrantベクトルデータベースの操作を担当するサービス層

#### 5.2.1 get_collection_embedding_params()

**概要**: コレクションの設定（ベクトル次元数）から埋め込みモデル設定を推論

```python
def get_collection_embedding_params(
    client: QdrantClient,
    collection_name: str
) -> Dict[str, Any]
```

| 項目 | 内容 |
|------|------|
| **Input** | `client`: QdrantClientインスタンス |
| | `collection_name`: コレクション名 |
| **Process** | 1. `client.get_collection()` でコレクション情報取得 |
| | 2. `vectors_config` からベクトルサイズを取得 |
| | 3. マルチベクトルの場合は最初のものを採用 |
| | 4. サイズに基づいてモデルを推論 |
| **Output** | `Dict[str, Any]`: モデル設定 `{"model": str, "dims": int}` |

**次元数→モデルマッピング**:

| 次元数 | 推論モデル |
|-------|-----------|
| 1536 | text-embedding-3-small |
| 3072 | gemini-embedding-001 |
| 768 | gemini-embedding-001 |
| その他 | unknown-embedding-model |
| 取得失敗 | gemini-embedding-001（デフォルト） |

---

### 5.3 agent_tools

**ファイル**: `agent_tools.py`（645行）
**概要**: Legacy Agent の検索ロジックを提供。RAGSearchTool が内部で使用。

#### 5.3.1 カスタム例外クラス

```python
class RAGToolError(Exception):
    """RAGツール固有のエラー基底クラス"""

class QdrantConnectionError(RAGToolError):
    """Qdrant接続エラー"""

class CollectionNotFoundError(RAGToolError):
    """コレクション未存在エラー"""

class EmbeddingError(RAGToolError):
    """埋め込み生成エラー"""
```

#### 5.3.2 SearchMetrics データクラス

**概要**: 検索結果のメトリクス（評価用）

```python
@dataclass
class SearchMetrics:
    query: str                              # 検索クエリ
    collection_name: str                    # コレクション名
    latency_ms: float                       # 検索遅延（ミリ秒）
    total_results: int                      # 総結果数
    filtered_results: int                   # フィルタ後結果数
    top_score: float                        # 最高スコア
    scores: List[float] = field(default_factory=list)  # 全スコアリスト
    error: Optional[str] = None             # エラーメッセージ
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
```

#### 5.3.3 search_rag_knowledge_base_structured()

**概要**: Qdrantデータベースから専門的な知識を検索（構造化データ版）。RAGSearchTool が内部で使用する主要関数。

```python
def search_rag_knowledge_base_structured(
    query: str,
    collection_name: Optional[str] = None,
    use_hybrid_search: bool = True
) -> Union[List[Dict[str, Any]], str]
```

| 項目 | 内容 |
|------|------|
| **Input** | `query`: 検索クエリ |
| | `collection_name`: 検索対象コレクション（省略時はデフォルト） |
| | `use_hybrid_search`: ハイブリッド検索を使用するか（デフォルト: True） |
| **Process** | 1. Qdrantヘルスチェック |
| | 2. コレクション存在確認 |
| | 3. `embed_query()` でクエリベクトル生成 |
| | 4. `use_hybrid_search` が True なら `embed_sparse_query_unified()` でスパースベクトル生成 |
| | 5. `search_collection()` で検索実行（候補20件） |
| | 6. `rerank_results()` でCohere Re-ranking（オプション） |
| | 7. メトリクス記録 |
| **Output** | 成功: `List[Dict[str, Any]]` 検索結果リスト |
| | 失敗: `str` エラーメッセージ |

**検索フロー図**:
```
┌────────────────────────────────────────────────────────────┐
│ search_rag_knowledge_base_structured                        │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                           │
│  │ Health Check│──No──► QdrantConnectionError              │
│  └──────┬──────┘                                           │
│         │ Yes                                               │
│         ▼                                                   │
│  ┌─────────────────┐                                       │
│  │Collection Exists?│──No──► CollectionNotFoundError       │
│  └──────┬──────────┘                                       │
│         │ Yes                                               │
│         ▼                                                   │
│  ┌─────────────────┐                                       │
│  │ embed_query()   │──► query_vector                       │
│  └──────┬──────────┘                                       │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────┐                               │
│  │ use_hybrid_search?      │                               │
│  │   True → sparse_vector  │                               │
│  │   False → None          │                               │
│  └──────┬──────────────────┘                               │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────┐                               │
│  │ search_collection()     │──► candidates (20件)          │
│  └──────┬──────────────────┘                               │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────┐                               │
│  │ rerank_results()        │──► reranked_results           │
│  │ (Cohere Re-ranking)     │    (AgentConfig.RAG_SEARCH_LIMIT件) │
│  └──────┬──────────────────┘                               │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                       │
│  │ Return Results  │                                       │
│  └─────────────────┘                                       │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**エラーメッセージ形式**:

| パターン | 意味 |
|---------|------|
| `[[NO_RAG_RESULT]]` | 検索結果が見つからなかった |
| `[[NO_RAG_RESULT_LOW_SCORE]]` | スコア閾値未満の結果のみ |
| `[[RAG_TOOL_ERROR]]` | 検索中にエラー発生 |

#### 5.3.4 rerank_results()

**概要**: 検索結果をCohere Rerank APIで再評価し、スコアを更新してソート

```python
def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 3,
    threshold: float = 0.5
) -> List[Dict[str, Any]]
```

| 項目 | 内容 |
|------|------|
| **Input** | `query`: ユーザーの検索クエリ |
| | `results`: Qdrantからの検索結果リスト |
| | `top_k`: 最終的に残す件数（デフォルト: 3） |
| | `threshold`: スコアの足切りライン（デフォルト: 0.5） |
| **Process** | 1. Cohere APIキーがない場合 → RRFスコアのままソートして返却 |
| | 2. 各結果から Q&A テキストを作成 |
| | 3. `cohere.rerank()` でリランキング実行 |
| | 4. `original_score`, `rerank_score` を記録 |
| | 5. threshold以上の結果のみ返却 |
| **Output** | `List[Dict[str, Any]]`: リランク済み結果リスト |

**Re-ranking後のスコア構造**:
```python
{
    "score": 0.902,           # Cohereスコア（互換性用）
    "original_score": 0.66,   # 元のRRFスコア
    "rerank_score": 0.902,    # Cohereスコア
    "payload": {...},
    "id": ...
}
```

---

### 5.4 regex_mecab

**ファイル**: `regex_mecab.py`（390行）
**概要**: MeCabと正規表現を統合したキーワード抽出システム

#### 5.4.1 KeywordExtractor クラス

**概要**: MeCabが利用可能な場合は複合名詞抽出を優先し、利用不可の場合は正規表現版に自動フォールバック

```python
class KeywordExtractor:
    def __init__(self, prefer_mecab: bool = True)
    def extract(self, text: str, top_n: int = 5, use_scoring: bool = True) -> List[str]
    def extract_with_details(self, text: str, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]
```

| 項目 | 内容 |
|------|------|
| **属性** | `prefer_mecab`: MeCabを優先的に使用するか |
| | `mecab_available`: MeCabの利用可能フラグ |
| | `stopwords`: ストップワードセット（日本語+英語） |
| | `important_keywords`: 重要キーワードセット（スコアブースト用） |

**extract() メソッド**:

| 項目 | 内容 |
|------|------|
| **Input** | `text`: 分析対象テキスト |
| | `top_n`: 抽出するキーワード数（デフォルト: 5） |
| | `use_scoring`: スコアリングを使用するか（デフォルト: True） |
| **Process** | 1. 言語判定（日本語文字が含まれているか） |
| | 2. MeCab利用可能 & 日本語 → `_extract_with_mecab()` |
| | 3. 上記以外 → `_extract_with_regex()` |
| **Output** | `List[str]`: キーワードリスト |

**抽出フロー**:
```
┌─────────────────────────────────────────────────────────┐
│ KeywordExtractor.extract(text, top_n)                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────┐                │
│  │ 言語判定: 日本語文字が含まれるか？  │                │
│  │ re.search(r'[ぁ-んァ-ヶー一-龠]')  │                │
│  └──────────────┬──────────────────────┘                │
│                 │                                        │
│        ┌───────┴───────┐                                │
│        │               │                                │
│        ▼               ▼                                │
│  ┌──────────┐    ┌──────────┐                          │
│  │ Japanese │    │ English  │                          │
│  └────┬─────┘    └────┬─────┘                          │
│       │               │                                 │
│       ▼               │                                 │
│  ┌────────────────┐   │                                │
│  │ MeCab Available?│  │                                │
│  └───┬────────┬───┘   │                                │
│      │Yes     │No     │                                │
│      ▼        ▼       ▼                                │
│  ┌────────┐ ┌─────────────────────┐                    │
│  │ MeCab  │ │ Regex Extraction    │                    │
│  │複合名詞│ │[ァ-ヴー]{2,}|        │                    │
│  │ 抽出   │ │[一-龥]{2,}|          │                    │
│  └────┬───┘ │[A-Za-z]{2,}[A-Za-z0-9]*│                  │
│       │     └──────────┬──────────┘                    │
│       │                │                                │
│       └───────┬────────┘                                │
│               ▼                                         │
│  ┌─────────────────────────────────────┐               │
│  │ Scoring (use_scoring=True)          │               │
│  │ - 頻度スコア (×0.3)                 │               │
│  │ - 長さスコア (×0.3)                 │               │
│  │ - 重要キーワードブースト (+0.5)     │               │
│  │ - 文字種スコア (カタカナ/英大文字等)│               │
│  └──────────────┬──────────────────────┘               │
│                 ▼                                       │
│  ┌─────────────────────────────────────┐               │
│  │ Return top_n keywords               │               │
│  └─────────────────────────────────────┘               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**スコアリング詳細**:

| スコア種別 | 計算方法 | 重み |
|-----------|----------|------|
| 頻度スコア | `min(freq / 3.0, 1.0)` | ×0.3 |
| 長さスコア | `min(len(word) / 8.0, 1.0)` | ×0.3 |
| 重要キーワードブースト | important_keywordsに部分一致 | +0.5 |
| カタカナ3文字以上 | `^[ァ-ヴー]{3,}$` | +0.2 |
| 英大文字2文字以上（頭字語） | `^[A-Z]{2,}$` | +0.3 |
| 英語固有名詞 | `^[A-Z][a-z]+$` | +0.1 |
| 漢字4文字以上 | `^[一-龥]{4,}$` | +0.2 |

**ストップワード（一部）**:
```python
stopwords = {
    # 日本語
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
    'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる', ...
    # 英語
    'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', ...
}
```

**重要キーワード（スコアブースト用）**:
```python
important_keywords = {
    'AI', 'Artificial Intelligence', 'Machine Learning', 'Deep Learning',
    'NLP', 'Natural Language Processing', 'Transformer', 'BERT', 'GPT',
    'CNN', 'Vision', '医療', 'Diagnosis', 'Autonomous Driving',
    'Ethics', 'Bias', 'Challenges', 'Issues', 'Model', 'Data'
}
```

**使用例**:
```python
from regex_mecab import KeywordExtractor

extractor = KeywordExtractor(prefer_mecab=True)

# 日本語テキスト
text_jp = """
人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
"""

keywords = extractor.extract(text_jp, top_n=5)
# 出力例: ['トランスフォーマー', '自然言語処理', '人工知能', '機械学習', '深層学習']

# 詳細情報付き
details = extractor.extract_with_details(text_jp, top_n=5)
# 出力例: {
#     'MeCab複合名詞': [('自然言語処理', 0.85), ('人工知能', 0.82), ...],
#     '正規表現': [('トランスフォーマー', 0.78), ...],
#     '統合版': [('自然言語処理', 0.85), ...]
# }
```

---

## 6. 設定・定数

### 6.1 QdrantConfig（Qdrant設定）

RAGSearchToolで使用するQdrant関連の設定。

```python
class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection_name: str = "customer_support_faq"
    search_limit: int = 5
    score_threshold: float = 0.35
    search_priority: list = Field(default_factory=lambda: [
        "wikipedia_ja", "livedoor", "cc_news", "japanese_text"
    ])
```

| キー | デフォルト値 | 説明 |
|-----|-------------|------|
| `url` | "http://localhost:6333" | Qdrant接続URL |
| `collection_name` | "customer_support_faq" | デフォルトコレクション名 |
| `search_limit` | 5 | 検索結果の取得件数上限 |
| `score_threshold` | 0.35 | スコア閾値 |
| `search_priority` | ["wikipedia_ja", ...] | コレクション検索優先順位 |

### 6.2 ToolsConfig（ツール設定）

有効なツールを制御する設定。

```python
class ToolsConfig(BaseModel):
    enabled: list = Field(default_factory=lambda: ["rag_search", "reasoning", "ask_user"])
```

| キー | デフォルト値 | 説明 |
|-----|-------------|------|
| `enabled` | ["rag_search", "reasoning", "ask_user"] | 有効なツールリスト |

### 6.3 ReasoningToolのプロンプトルール

回答生成時に適用されるルール。

| ルール | 内容 |
|--------|------|
| 正確性と誠実さ | 参照情報にある事実のみを述べる |
| 判明した事実を優先 | 直接的な回答を最初に述べる |
| 出典の明示 | 「社内ナレッジ（出典）によると...」形式 |
| 丁寧な日本語 | です・ます調で読みやすく構造化 |
| 捏造禁止 | 事前知識での補完・推測を禁止 |

### 6.4 Dynamic Thresholding

RAG検索結果のノイズ除去ルール。

| 条件 | 動作 |
|------|------|
| Top1スコア >= 0.98 | 2位以下の結果を除外 |
| Top1スコア < 0.98 | すべての結果を保持 |

---

## 7. 使用例

### 7.1 ToolRegistryを使用した基本ワークフロー

```python
from grace.tools import create_tool_registry

# 1. レジストリを作成
registry = create_tool_registry()

# 2. 登録されているツールを確認
print(f"利用可能なツール: {registry.list_tools()}")
# 出力: 利用可能なツール: ['rag_search', 'reasoning', 'ask_user']

# 3. RAG検索を実行
search_result = registry.execute(
    "rag_search",
    query="東京の人口を教えてください"
)

if search_result.success:
    print(f"検索結果: {len(search_result.output)}件")

    # 4. 検索結果を使って推論
    reasoning_result = registry.execute(
        "reasoning",
        query="東京の人口を教えてください",
        sources=search_result.output
    )

    if reasoning_result.success:
        print(f"回答: {reasoning_result.output}")
else:
    print(f"検索失敗: {search_result.error}")
```

### 7.2 RAG検索の直接実行

```python
from grace.tools import RAGSearchTool

# ツールを直接作成
tool = RAGSearchTool()

# 特定コレクションを指定して検索
result = tool.execute(
    query="Python の基本文法",
    collection="wikipedia_ja"
)

if result.success:
    for item in result.output:
        print(f"スコア: {item['score']:.2f}")
        print(f"コレクション: {item.get('collection', 'unknown')}")
        print(f"回答: {item['payload'].get('answer', '')[:100]}...")
        print("---")

# 信頼度情報を確認
print(f"信頼度要素: {result.confidence_factors}")
```

### 7.3 推論ツールの使用

```python
from grace.tools import ReasoningTool

# ツールを作成
tool = ReasoningTool()

# ソース情報を準備（RAG検索結果の形式）
sources = [
    {
        "score": 0.92,
        "payload": {
            "question": "東京の人口は？",
            "answer": "東京都の人口は約1400万人です。",
            "source": "統計局データ.pdf"
        },
        "collection": "wikipedia_ja"
    }
]

# 追加コンテキスト（他のステップの結果など）
context = "前のステップで、ユーザーは2024年のデータを希望していることが判明しました。"

# 推論実行
result = tool.execute(
    query="東京の人口について詳しく教えてください",
    context=context,
    sources=sources
)

if result.success:
    print(f"回答:\n{result.output}")
    print(f"\nトークン使用量: {result.confidence_factors.get('token_usage')}")
```

### 7.4 AskUserToolの使用

```python
from grace.tools import AskUserTool

# ツールを作成
tool = AskUserTool()

# 選択肢付きの質問
result = tool.execute(
    question="どの年度のデータをお探しですか？",
    reason="検索結果に複数年度のデータが含まれているため",
    urgency="blocking",
    options=["2022年", "2023年", "2024年", "最新のデータ"]
)

# 結果を確認
output = result.output
print(f"質問: {output['question']}")
print(f"理由: {output['reason']}")
print(f"緊急度: {output['urgency']}")
print(f"選択肢: {output['options']}")
print(f"回答待ち: {output['awaiting_response']}")

# 実際のUIとの連携はExecutorで行う
# Executorが output['awaiting_response'] を見て
# ユーザー入力を待機する
```

---

## 8. エクスポート

`__all__`でエクスポートされる要素：

```python
__all__ = [
    # Data classes
    "ToolResult",

    # Base class
    "BaseTool",

    # Tools
    "RAGSearchTool",
    "ReasoningTool",
    "AskUserTool",

    # Registry
    "ToolRegistry",
    "create_tool_registry",
]
```

---

## 9. 変更履歴

| バージョン | 変更内容 |
|-----------|---------|
| 1.0 | 初版作成（2025-01-29） |
| 1.1 | 外部カスタムモジュール IPO詳細を追加（2025-01-29） |

---

## 付録: 依存関係図

```
tools.py
    │
    ├──► abc
    │        └── ABC
    │        └── abstractmethod
    │
    ├──► dataclasses
    │        └── dataclass
    │        └── field
    │
    ├──► typing
    │        └── Dict, Any, Optional, List
    │
    ├──► logging
    │        └── getLogger
    │
    ├──► qdrant_client
    │        └── QdrantClient
    │
    ├──► google.genai
    │        └── genai.Client
    │        └── types.GenerateContentConfig
    │
    ├──► qdrant_client_wrapper (カスタム)
    │        └── search_collection
    │        └── embed_query_unified
    │        └── embed_sparse_query_unified
    │
    ├──► services.qdrant_service (カスタム)
    │        └── get_collection_embedding_params
    │
    ├──► agent_tools (カスタム)
    │        └── search_rag_knowledge_base_structured
    │
    ├──► regex_mecab (カスタム)
    │        └── KeywordExtractor
    │
    └──► .config (内部)
             └── get_config()
             └── GraceConfig
```

### ツール → 外部サービス連携図

```
ToolRegistry
    │
    ├── RAGSearchTool
    │       │
    │       ├── QdrantClient
    │       │       └── Qdrant Server (localhost:6333)
    │       │
    │       └── search_rag_knowledge_base_structured
    │               └── Legacy Agent検索ロジック
    │
    ├── ReasoningTool
    │       │
    │       └── genai.Client
    │               └── Gemini API
    │
    └── AskUserTool
            │
            └── (UI連携はExecutorが担当)
```

---

## 関連ドキュメント

| ドキュメント | 説明 |
|-------------|------|
| `config.md` | GraceConfig設定管理の詳細ドキュメント |
| `executor.md` | 計画実行エージェント（ToolRegistryの使用元） |
| `confidence.md` | 信頼度計算システム（ToolResult.confidence_factorsの使用先） |
| `schemas.md` | StepResult等のデータ構造 |

---

## 解決済み・残存課題

### Version 1.1 で解決した項目

ドキュメント作成にあたり、以下の情報が **Version 1.1 で追加されました**：

1. ✅ **外部カスタムモジュールの詳細**:
   - `qdrant_client_wrapper` の `search_collection`, `embed_query_unified`, `embed_sparse_query_unified` の仕様 → [セクション 5.1](#51-qdrant_client_wrapper)
   - `services.qdrant_service` の `get_collection_embedding_params` の仕様 → [セクション 5.2](#52-servicesqdrant_service)
   - `agent_tools` の `search_rag_knowledge_base_structured` の詳細な入出力仕様 → [セクション 5.3](#53-agent_tools)
   - `regex_mecab.KeywordExtractor` の詳細仕様 → [セクション 5.4](#54-regex_mecab)

### 残存する課題・注意事項

1. **コメントアウトされた機能**:
   - `RAGSearchTool.execute()` 内のキーワードフィルタリング機能（tools.py 116-125行、166-179行）がコメントアウトされています。
   - この機能は将来の有効化が検討されている可能性があります。有効化時は `required_keywords` によるフィルタリングが追加されます。

2. **limit, score_threshold パラメータの使用**:
   - `RAGSearchTool.execute()` のパラメータ `limit`, `score_threshold` が定義されていますが、現在の実装では `search_rag_knowledge_base_structured` に直接渡されていません。
   - 検索件数は `agent_tools.py` 内で `limit=20`（候補取得）、`top_k=AgentConfig.RAG_SEARCH_LIMIT`（最終結果）としてハードコードされています。

3. **KeywordExtractor の現状**:
   - `RAGSearchTool.__init__` で `KeywordExtractor` が初期化されていますが、キーワードフィルタリング機能がコメントアウトされているため、現在は使用されていません。

