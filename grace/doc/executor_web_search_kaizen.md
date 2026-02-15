---
## web_search 実装 TODO

**作成日**: 2026-02-15
**前提**: replan_kaizen.md の評価結果に基づく改善施策（優先度2）
#### 全体の考え方：「まず壊さない、次に足す、最後に繋ぐ」
- 3フェーズに分けている理由は、現状で動いている rag_search → reasoning のパイプラインを壊さずに進めるためです。
- Phase 0（前準備） — replanを一旦止めて、代わりにステップ単位のリトライを入れる。これだけで一時障害への耐性は上がり、無駄なLLM API呼び出し（Planner再生成）がなくなります。
- Phase A（WebSearchTool実装） — ここが本体。最大の設計判断は 「outputの形式をRAGSearchToolと揃える」 点です。Web検索の生結果を {"payload": {"content": ..., "source": ...}, "score": ...} に正規化することで、executorのreasoning分岐もReasoningToolの_build_promptも一切変更不要になります。影響範囲を最小にする鍵がこの正規化です。
- Phase B（replan統合） — web_searchが動く状態になって初めて、replanの FALLBACK 戦略が「rag_search失敗 → web_searchに切り替え」として機能します。ここでreplanを再有効化します。
- API選定について
- DuckDuckGoを第一段階にした理由は、API Key不要で pip install duckduckgo-search だけで動くためです。学習用プロジェクトとして動作確認のサイクルを速くまわせます。日本語精度が不十分と判断した段階でGoogle Custom Search APIに切り替える想定です。
- 最もリスクの高い箇所
- output正規化の不整合です。
RAGSearchToolは List[Dict] を返し、それが _format_output で文字列化され、reasoningステップで ast.literal_eval で復元されるという複雑な経路を通ります。
WebSearchToolの出力がこの経路を正しく通過するかの検証が、Phase A の成否を決めます。
テスト A-9 がそれに当たります。Web search todoドキュメント

---

## 1. 目的

`rag_search`（Qdrant）が唯一の情報獲得手段である現状を改善し、Web検索による代替情報獲得経路を追加する。これにより以下が実現する:

- `rag_search` 失敗時の FALLBACK が実効性を持つ
- replan の FULL/PARTIAL 戦略で「異なる計画」が生成可能になる
- Qdrant に存在しない情報（最新ニュース等）への回答が可能になる

---

## 2. API選定

### 候補比較

| 候補 | メリット | デメリット | コスト |
|------|---------|-----------|--------|
| **Google Custom Search JSON API** | Geminiエコシステムと親和性高い、日本語検索精度が高い | API Key + Search Engine ID が必要 | 無料枠: 100回/日、超過: $5/1000回 |
| **Gemini Grounding with Google Search** | Gemini API内で完結、実装最小 | レスポンス形式がRAGと異なる、細かい制御が難しい | Gemini API料金に含まれる |
| **DuckDuckGo (duckduckgo-search)** | API Key不要、pip installだけで動く | 日本語検索精度がやや低い、レート制限あり | 無料 |
| **SerpAPI** | 安定、多機能 | 有料前提 | 無料枠: 100回/月 |

### 推奨: 2段階アプローチ

**Phase A（学習・プロトタイプ）**: DuckDuckGo — API Key不要で即座に動作確認可能
**Phase B（本番品質）**: Google Custom Search JSON API — 日本語精度とGeminiとの親和性

---

## 3. TODO一覧

### Phase 0: 前準備（replan無効化）

| # | タスク | 対象ファイル | 内容 |
|---|--------|-------------|------|
| 0-1 | replan一旦無効化 | `executor.py` | `Executor.__init__` の `enable_replan` デフォルトを `False` に変更。web_search実装完了後に `True` に戻す |
| 0-2 | ステップレベルリトライ追加 | `executor.py` | `_execute_step` 内で一時的障害（timeout, 接続エラー）を `ErrorConfig.max_retries` 回リトライするロジックを追加 |

### Phase A: WebSearchTool 実装（DuckDuckGo版）

| # | タスク | 対象ファイル | 内容 |
|---|--------|-------------|------|
| A-1 | パッケージ追加 | `requirements.txt` | `duckduckgo-search` を追加 |
| A-2 | WebSearchTool クラス実装 | `tools.py` | `BaseTool` を継承した `WebSearchTool` クラスを新規作成 |
| A-3 | ToolRegistry 登録 | `tools.py` | `_register_default_tools` に `web_search` の登録を追加 |
| A-4 | ToolsConfig 更新 | `config.py` | `ToolsConfig.enabled` のデフォルトに `"web_search"` を追加 |
| A-5 | Executor kwargs対応 | `executor.py` | `_prepare_tool_kwargs` に `web_search` 用の分岐を追加 |
| A-6 | Planner プロンプト更新 | `planner.py` | `PLAN_GENERATION_PROMPT` の利用可能アクションに `web_search` の使い分けルールを追記 |
| A-7 | Reasoning統合 | `executor.py` | `_prepare_tool_kwargs` の `reasoning` 分岐で、web_search結果もcontextに含める対応 |
| A-8 | 単体テスト | `tests/` | `WebSearchTool.execute()` の正常系・異常系テスト |
| A-9 | 結合テスト | `tests/` | `rag_search失敗 → web_search fallback` のE2Eテスト |

### Phase B: replan有効化と統合

| # | タスク | 対象ファイル | 内容 |
|---|--------|-------------|------|
| B-1 | replan再有効化 | `executor.py` | `enable_replan` デフォルトを `True` に戻す |
| B-2 | Planner fallback設定 | `planner.py` | `PLAN_GENERATION_PROMPT` で `rag_search` の `fallback` に `"web_search"` を推奨する指示を追加 |
| B-3 | ReplanManager戦略改善 | `replan.py` | `determine_strategy` で、rag_search失敗時に FALLBACK(web_search) を優先選択するロジック追加 |
| B-4 | リプラン結合テスト | `tests/` | `rag_search失敗 → replan → web_search` のフローテスト |

### Phase C: 本番品質化（Google Custom Search移行）

| # | タスク | 対象ファイル | 内容 |
|---|--------|-------------|------|
| C-1 | WebSearchConfig追加 | `config.py` | Google API Key、Search Engine ID、max_results等の設定モデル |
| C-2 | WebSearchTool差し替え | `tools.py` | DuckDuckGo → Google Custom Search API に実装を変更 |
| C-3 | コスト管理統合 | `config.py` | `CostConfig` にweb_search APIコスト上限を追加 |

---

## 4. 設計詳細

### 4.1 WebSearchTool クラス設計（Phase A）

```python
class WebSearchTool(BaseTool):
    """Web検索ツール"""

    name = "web_search"
    description = "Webから最新情報や追加情報を検索"

    def __init__(self, config=None, max_results=5):
        self.config = config or get_config()
        self.max_results = max_results

    def execute(self, query, max_results=None, **kwargs) -> ToolResult:
        """
        Web検索を実行

        Args:
            query: 検索クエリ
            max_results: 取得件数

        Returns:
            ToolResult: 検索結果
                output: List[Dict] — rag_searchと同じ構造に正規化
                    各要素: {"payload": {"content": ..., "source": URL}, "score": 関連度}
        """
```

**設計上の重要ポイント**: `output` の形式を `RAGSearchTool` と揃える。これにより `ReasoningTool._build_prompt` と `executor._prepare_tool_kwargs` が既存のsources処理ロジックをそのまま使える。

### 4.2 出力正規化の例

```python
# DuckDuckGo の生の結果
{"title": "...", "body": "...", "href": "https://..."}

# ToolResult.output に格納する正規化形式
{
    "payload": {
        "content": "...",      # body を格納
        "question": "...",     # title を格納
        "answer": "",          # web検索では空
        "source": "https://..."  # URL を格納
    },
    "score": 0.8,              # 検索順位ベースのスコア（1位=0.9, 2位=0.8, ...）
    "collection": "web_search" # 識別用
}
```

### 4.3 修正が必要な箇所の詳細

**executor.py `_prepare_tool_kwargs`**:

```python
# 追加する分岐
elif step.action == "web_search":
    kwargs["max_results"] = 5  # またはconfigから取得
```

**executor.py `_prepare_tool_kwargs` のreasoning分岐**:
変更不要。web_search の output を rag_search と同じ形式に正規化しているため、既存の sources 処理がそのまま動く。

**planner.py `PLAN_GENERATION_PROMPT`**:

```
【利用可能なアクション】
- rag_search: ベクトルDB（Qdrant）から関連情報を検索
- web_search: Webから最新情報や追加情報を検索（Qdrantにない情報が必要な場合）
- reasoning: 収集した情報を分析・統合して回答を生成
- ask_user: ユーザーに追加情報や確認を求める

【rag_search と web_search の使い分け】
- 原則: rag_search を優先。社内ナレッジや登録済みデータへの質問は rag_search。
- web_search を使う場面: 最新ニュース、Qdrantに未登録の一般知識、rag_search のfallback。
- rag_search の fallback には "web_search" を設定してください。
```

### 4.4 replan戦略改善（Phase B）

`determine_strategy` に以下のロジックを追加:

```python
# rag_search失敗 + web_searchが利用可能 → FALLBACK優先
if context.trigger == ReplanTrigger.STEP_FAILED:
    if context.failed_step_id:
        failed_step = self._find_step(current_plan, context.failed_step_id)
        if failed_step and failed_step.action == "rag_search":
            # fallbackにweb_searchがあればFALLBACK
            if failed_step.fallback == "web_search":
                return ReplanStrategy.FALLBACK
            # なくてもweb_searchへの切り替えを試みる
            # （fallbackフィールドを動的に書き換える案もある）
```

---

## 5. 実装順序とマイルストーン

```
Phase 0 (1日)
  ├── 0-1: replan無効化
  └── 0-2: ステップリトライ追加
      ↓
Phase A (2-3日)
  ├── A-1〜A-3: WebSearchTool実装・登録
  ├── A-4〜A-7: 既存モジュール連携
  └── A-8〜A-9: テスト
      ↓
Phase B (1-2日)
  ├── B-1〜B-3: replan再有効化・戦略改善
  └── B-4: 結合テスト
      ↓
Phase C (将来)
  └── Google Custom Search API移行
```

---

## 6. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| DuckDuckGoのレート制限 | 連続テスト時にブロックされる | テストではモック使用、本番は間隔を空ける |
| Web検索結果の品質ばらつき | reasoning の回答品質が低下 | confidence_factors に `source_type: "web"` を追加し、rag結果より低めのスコアを設定 |
| output正規化の不整合 | reasoning が sources を読めない | 正規化形式の単体テストを必須にする |
| API コスト増大 | 予算超過 | Phase Aでは無料API、Phase Cで `CostConfig` にweb検索上限を追加 |
