# web_search 実装仕様書 + TODO

## 1. 概要

GRACE Agent の `ActionType.WEB_SEARCH` を実装する。
既存の `rag_search` と同じ `BaseTool` → `ToolRegistry` → `Executor` パターンに準拠し、
`ToolResult` インターフェースで結果を返す。

- **検索バックエンド**: Google Custom Search API
- **本文取得**: snippetのみ（軽量・高速）
- **outputフォーマット**: `rag_search` と下流互換（`reasoning` ステップがそのまま消費可能）

---

## 2. 既存 rag_search のインターフェース分析

### 2.1 データフロー

```
User Query
  → Planner.create_plan()
    → PlanStep(action="rag_search", query=..., collection=...)
      → Executor._execute_step()
        → Executor._prepare_tool_kwargs()  ← rag_search分岐で kwargs 構築
          → RAGSearchTool.execute(**kwargs)
            → ToolResult(success, output, confidence_factors, error, execution_time_ms)
              → Executor が StepResult に変換
                → 後続の reasoning ステップが sources として消費
```

### 2.2 RAGSearchTool.execute() の INPUT

| 引数             | 型                  | 必須 | 供給元                            |
|:----------------|:-------------------|:-----|:--------------------------------|
| `query`         | `str`              | ✅   | `step.query or step.description` |
| `collection`    | `Optional[str]`    | ❌   | `step.collection`                |
| `limit`         | `Optional[int]`    | ❌   | config default (5)               |
| `score_threshold` | `Optional[float]` | ❌   | config default (0.35)            |

### 2.3 RAGSearchTool.execute() の OUTPUT（ToolResult）

```python
ToolResult(
    success=True,
    output=[                           # List[Dict[str, Any]]
        {
            "score": 0.92,             # float: 類似度スコア (0.0-1.0)
            "payload": {
                "question": "...",     # str: Q/Aペアの質問
                "answer": "...",       # str: Q/Aペアの回答
                "content": "...",      # str: チャンク本文
                "source": "file.csv",  # str: 出典ファイル名
            },
            "collection": "wikipedia_ja",  # str: コレクション名
        },
        # ...
    ],
    confidence_factors={
        "result_count": 5,              # int
        "avg_score": 0.85,              # float
        "top_score": 0.92,              # float
        "score_spread": 0.12,           # float
        "used_collection": "wikipedia_ja",  # str
    },
    error=None,                         # Optional[str]
    execution_time_ms=350,              # int
)
```

### 2.4 下流（reasoning ステップ）での消費パターン

`executor.py` の `_prepare_tool_kwargs()` で `reasoning` 分岐処理:

1. `depends_on` で指定された先行ステップの `output` を取得
2. `output` が `List[Dict]` → `sources` としてそのまま `ReasoningTool` に渡す
3. `ReasoningTool._build_prompt()` が `sources` 内の `payload.question`, `payload.answer`, `payload.content`, `payload.source` と `score`, `collection` を参照してプロンプト構築

**→ web_search の output も同じ構造にすれば、ReasoningTool は無変更で利用可能**

---

## 3. web_search 確定仕様

### 3.1 WebSearchTool.execute() の INPUT

| 引数           | 型                | 必須 | デフォルト   | 説明                              |
|:--------------|:-----------------|:-----|:-----------|:--------------------------------|
| `query`       | `str`            | ✅   | —          | 検索クエリ                         |
| `num_results` | `int`            | ❌   | `5`        | 取得件数                           |
| `language`    | `str`            | ❌   | `"ja"`     | 検索言語（`lr` パラメータ）           |

### 3.2 WebSearchTool.execute() の OUTPUT（ToolResult）

`rag_search` と構造互換にする:

```python
ToolResult(
    success=True,
    output=[
        {
            "score": 0.95,                    # 検索順位ベースの正規化スコア
            "payload": {
                "question": "",               # 空文字（web検索にQ/Aペアなし）
                "answer": "snippet text...",  # Google CSE の snippet
                "content": "",                # snippetのみモードでは空
                "source": "https://example.com/article",  # URL
                "title": "記事タイトル",        # ページタイトル（追加フィールド）
            },
            "collection": "web_search",       # 固定識別子
        },
        # ...
    ],
    confidence_factors={
        "result_count": 5,
        "avg_score": 0.80,
        "top_score": 0.95,
        "score_spread": 0.30,
        "search_engine": "google_cse",
    },
    error=None,
    execution_time_ms=1200,
)
```

### 3.3 スコア算出ロジック

Google CSE はスコアを返さないため、検索順位から正規化スコアを生成する:

```python
score = 1.0 - (rank_index / num_results) * 0.5
# rank 0 → 1.0, rank 1 → 0.9, rank 2 → 0.8, ...
```

### 3.4 Google Custom Search API 設定

| 項目                      | 値                                  |
|:-------------------------|:------------------------------------|
| エンドポイント              | `https://www.googleapis.com/customsearch/v1` |
| 認証                      | API Key (`GOOGLE_CSE_API_KEY`)       |
| Search Engine ID          | `GOOGLE_CSE_ENGINE_ID`               |
| 無料枠                    | 100クエリ/日                          |
| 有料                      | $5 / 1,000クエリ                     |

### 3.5 エラーハンドリング

| エラー条件                    | 対応                                    |
|:---------------------------|:----------------------------------------|
| API Key 未設定               | `ToolResult(success=False, error=...)` 即返却 |
| HTTP 429 (レート制限)         | リトライ1回 → 失敗なら error 返却          |
| HTTP 403 (クォータ超過)       | error に残クォータ情報を含めて返却          |
| ネットワークエラー             | リトライ1回 → 失敗なら error 返却          |
| 結果0件                      | `ToolResult(success=False, output=[])` |

---

## 4. 変更対象ファイルと変更内容

### 4.1 新規作成

なし（既存ファイルへの追記のみ）

### 4.2 変更ファイル一覧

| # | ファイル | 変更内容 | 規模 |
|:-:|:--------|:--------|:-----|
| 1 | `grace/config.py` | `WebSearchConfig` モデル追加、`ToolsConfig.enabled` デフォルトに `"web_search"` 追加 | 小 |
| 2 | `config/grace_config.yml` | `web_search` セクション追加 | 小 |
| 3 | `grace/tools.py` | `WebSearchTool` クラス追加、`ToolRegistry._register_default_tools` に登録追加 | 中 |
| 4 | `grace/executor.py` | `_prepare_tool_kwargs` に `web_search` 分岐追加 | 小 |
| 5 | `grace/planner.py` | `PLAN_GENERATION_PROMPT` に `web_search` アクション説明追加 | 小 |
| 6 | `grace/schemas.py` | **変更不要**（`"web_search"` は `PlanStep.action` Literal に既存） | — |
| 7 | `.env` (または環境変数) | `GOOGLE_CSE_API_KEY`, `GOOGLE_CSE_ENGINE_ID` 追加 | 小 |

---

## 5. TODO リスト（実装順序）

### Phase 0: 事前準備

- [ ] **TODO-0.1**: Google Cloud Console で Custom Search API を有効化
- [ ] **TODO-0.2**: API Key を作成（制限設定: Custom Search API のみ）
- [ ] **TODO-0.3**: Programmable Search Engine (CSE) を作成し、Engine ID を取得
  - 検索対象: ウェブ全体
  - 言語: 日本語優先
- [ ] **TODO-0.4**: `.env` に環境変数を追加
  ```
  GOOGLE_CSE_API_KEY=AIza...
  GOOGLE_CSE_ENGINE_ID=a1b2c3...
  ```
- [ ] **TODO-0.5**: `pip install requests`（既存で入っている可能性大、確認のみ）

### Phase 1: Config 変更

- [ ] **TODO-1.1**: `grace/config.py` に `WebSearchConfig` モデルを追加
  ```python
  class WebSearchConfig(BaseModel):
      """Web検索設定"""
      api_key: str = ""           # 環境変数 GOOGLE_CSE_API_KEY から上書き
      engine_id: str = ""         # 環境変数 GOOGLE_CSE_ENGINE_ID から上書き
      default_num_results: int = 5
      language: str = "ja"
      timeout: int = 10
  ```
- [ ] **TODO-1.2**: `GraceConfig` に `web_search: WebSearchConfig` フィールドを追加
- [ ] **TODO-1.3**: `ToolsConfig.enabled` のデフォルトに `"web_search"` を追加
  ```python
  enabled: list = Field(
      default_factory=lambda: ["rag_search", "web_search", "reasoning", "ask_user"]
  )
  ```
- [ ] **TODO-1.4**: `config/grace_config.yml` に `web_search` セクションを追加（任意）

### Phase 2: WebSearchTool 実装（grace/tools.py）

- [ ] **TODO-2.1**: `WebSearchTool` クラスを実装
  ```python
  class WebSearchTool(BaseTool):
      name = "web_search"
      description = "Google検索でWeb上の情報を検索"

      def __init__(self, config=None):
          # API Key, Engine ID を config + 環境変数から取得

      def execute(self, query, num_results=5, language="ja", **kwargs) -> ToolResult:
          # 1. API Key / Engine ID の存在チェック
          # 2. Google CSE API 呼び出し (requests.get)
          # 3. レスポンスパース → rag_search 互換 output 構築
          # 4. confidence_factors 算出
          # 5. ToolResult 返却

      def _call_google_cse(self, query, num_results, language) -> dict:
          # HTTP リクエスト + エラーハンドリング

      def _parse_results(self, raw_results, num_results) -> List[Dict]:
          # items[] → rag_search 互換フォーマット変換
          # score = 1.0 - (rank / num_results) * 0.5

      def _calculate_confidence_factors(self, results) -> Dict:
          # result_count, avg_score, top_score, score_spread
  ```
- [ ] **TODO-2.2**: `ToolRegistry._register_default_tools` に登録を追加
  ```python
  if "web_search" in enabled_tools:
      self.register(WebSearchTool(config=self.config))
  ```
- [ ] **TODO-2.3**: `__all__` に `"WebSearchTool"` を追加

### Phase 3: Executor 対応（grace/executor.py）

- [ ] **TODO-3.1**: `_prepare_tool_kwargs` に `web_search` 分岐を追加
  ```python
  elif step.action == "web_search":
      kwargs["num_results"] = 5  # config から取得も可
      kwargs["language"] = "ja"
  ```
  ※ `query` は全アクション共通で設定済み

### Phase 4: Planner プロンプト更新（grace/planner.py）

- [ ] **TODO-4.1**: `PLAN_GENERATION_PROMPT` の「利用可能なアクション」に追加
  ```
  - web_search: Google検索でWeb上の最新情報を取得（RAGに情報がない場合のフォールバック）
  ```
- [ ] **TODO-4.2**: 計画作成ルールに web_search 使用ガイドラインを追加
  ```
  【web_search の使用条件】
  - RAG検索で十分な情報が得られない可能性がある場合（最新ニュース、一般的なWeb情報）
  - ユーザーが明示的にWeb検索を要求した場合
  - 原則として rag_search → web_search の順で計画する（RAGを優先）
  ```

### Phase 5: テスト

- [ ] **TODO-5.1**: 単体テスト — `WebSearchTool.execute()` の正常系
  - API Key あり → 結果取得 → ToolResult 構造検証
- [ ] **TODO-5.2**: 単体テスト — `WebSearchTool.execute()` の異常系
  - API Key なし → `success=False`
  - 結果0件 → `success=False, output=[]`
- [ ] **TODO-5.3**: 統合テスト — Planner → Executor → WebSearchTool → ReasoningTool
  - 「最新の為替レートは？」のような質問で web_search が計画に含まれることを確認
  - web_search の output が reasoning の sources として正しく消費されることを確認
- [ ] **TODO-5.4**: Streamlit UI での動作確認
  - grace_chat_page から web_search を含む質問を投げて、思考プロセスに表示されることを確認

### Phase 6: ドキュメント

- [ ] **TODO-6.1**: README.md に web_search 機能の説明を追加
- [ ] **TODO-6.2**: 環境変数の設定手順を追加（Google CSE セットアップガイド）

---

## 6. 実装上の注意事項

### 6.1 API Key の管理

```python
# 優先順位: 環境変数 > config.yml > デフォルト(空)
import os
api_key = os.environ.get("GOOGLE_CSE_API_KEY", "") or self.config.web_search.api_key
```

### 6.2 ReasoningTool との互換性

`ReasoningTool._build_prompt()` は以下のフィールドを参照する:

```python
payload.get("question", "")   # web_search: 空文字
payload.get("answer", "")     # web_search: snippet
payload.get("content", "")    # web_search: 空文字（snippetのみモード）
payload.get("source", "")     # web_search: URL
source.get("score", 0)        # web_search: 順位ベーススコア
source.get("collection", "")  # web_search: "web_search"
```

`answer` にsnippetを入れることで、既存プロンプトの「A: {answer}」部分に自然に表示される。
`title` は追加フィールドだが、既存コードでは参照されないため安全。
将来的に `_build_prompt()` で `title` も表示するよう拡張可能。

### 6.3 Planner の判断基準

Planner が `rag_search` vs `web_search` を適切に選択できるよう、
プロンプトに明確な使い分け基準を記載する:

| 条件 | 選択するアクション |
|:-----|:-----------------|
| 社内ドキュメント・FAQ系 | `rag_search` |
| 最新ニュース・一般的なWeb情報 | `web_search` |
| RAGに情報がなさそうな場合 | `rag_search`(fallback="web_search") |
| 両方必要な場合 | `rag_search` → `web_search` → `reasoning` |
