## replan.py 実効性評価レポート

**作成日**: 2026-02-15
**対象**: GRACE replan.py 動的リプランニングシステム
**評価観点**: 現状のツール構成における replan の実効性・有効性

---

## 1. 評価結論

**replan.py の実効性は「極めて限定的」である。**

現状のツール構成で、各リプラン戦略が実際に何をするかをトレースした結果、情報獲得手段が `rag_search` の1つしかないため、リプランによる改善がほぼ見込めない。

---

## 2. 現状のツール実装状況

| action | 実装クラス | 実質的な処理 |
|--------|-----------|-------------|
| `rag_search` | `RAGSearchTool` | Qdrantベクトル検索（外部データ取得） |
| `reasoning` | `ReasoningTool` | Gemini LLMによる回答生成（加工処理） |
| `ask_user` | `AskUserTool` | ユーザーへの質問（メタデータ返却のみ） |

「情報を取ってくる手段」は `rag_search` の1つだけである。`reasoning` は取得済み情報の加工、`ask_user` は問い合わせであり、**情報獲得のバリエーションがない**のが根本問題。

なお、`schemas.py` の `ActionType` と `PlanStep.action` の Literal には `web_search` と `code_execute` が定義されているが、`tools.py` に実装クラスがなく、`ToolRegistry` にも登録されない。

---

## 3. 各リプラン戦略の実効性分析

### 3.1 FULL（全体再計画）

`_create_full_replan` → `Planner.create_plan(enhanced_query)` を呼ぶ。エラー情報をクエリに付加して再生成するが、Plannerが使える道具は同じ `rag_search → reasoning` の2ステップ構成。同じQdrant・同じコレクションに対して、クエリ表現が微妙に変わるだけ。

Qdrant接続障害やデータ不在が原因の場合、**何度やっても同じ結果**になる。

### 3.2 PARTIAL（部分再計画）

`_create_partial_replan` → 失敗ステップ以降を `Planner.create_plan()` で再生成。典型的なプランは `rag_search(step1) → reasoning(step2)` の2ステップ。step1(rag_search)が失敗した場合、step1以降を再計画しても、**また `rag_search → reasoning` が生成される**だけ。

### 3.3 FALLBACK（代替アクション）

`_apply_fallback` → `step.fallback` のアクションに差し替え。Plannerのフォールバック計画を見ると `fallback="reasoning"` が設定されている（planner.py L283）。つまり rag_search失敗 → reasoning に切り替え。

これは**ソースなしでLLMが回答を生成する**ことを意味し、GRACEの「根拠に基づく回答」という設計意図と矛盾する。ただし、唯一「異なる動作」にはなる。

### 3.4 SKIP（スキップ）

`_skip_failed_step` → rag_searchをスキップすると、reasoningが依存先なしで実行される。FALLBACKと同じく**根拠なしLLM回答**になる。

### 3.5 ABORT（中断）

これは正常に機能する。

---

## 4. 構造的問題の整理

### 問題1：情報獲得手段が1つしかない

リプランの本質は「Plan Aがダメなら Plan Bを試す」こと。しかし情報獲得が `rag_search`（Qdrant検索）のみのため、Plan Bが存在しない。`web_search` と `code_execute` は `schemas.py` の `ActionType` と `PlanStep.action` の Literal に定義されているが、`tools.py` に実装クラスがなく、`ToolRegistry` にも登録されない。

### 問題2：リプラン時の「変化」が乏しい

FULL/PARTIALは `Planner.create_plan()` を再呼び出しするが、Plannerのプロンプト（`PLAN_GENERATION_PROMPT`）が「rag_searchは1ステップにまとめろ」「queryは元の質問をそのままコピーしろ」と指示しているため、LLMが生成する計画はほぼ同一になる。

`_enhance_query_with_context` でエラー情報を付加しても、Plannerが選べるアクションが同じなので、出力される計画に実質的な差異が生じにくい。

### 問題3：リプラントリガーと失敗原因のミスマッチ

現在の `rag_search` 失敗パターンとリプラン効果の対応:

| 失敗原因 | リプランで解決するか |
|---------|-------------------|
| Qdrant接続障害 | しない（同じ接続先に再試行） |
| コレクションにデータがない | しない（同じデータに再検索） |
| クエリとデータのミスマッチ | ほぼしない（クエリ変更指示が弱い） |
| LLM API一時障害 | する可能性あり（時間差で復旧） |
| タイムアウト | する可能性あり（一時的な遅延なら） |

解決できるのは**一時的障害のみ**であり、それはリプランではなく単純リトライ（`config.py` の `ErrorConfig.max_retries`）で十分対応できる。

---

## 5. 総合評価

| 評価項目 | スコア (5段階) | コメント |
|---------|---------------|---------|
| 実効性 | ★☆☆☆☆ | 現状のツール構成ではほぼ機能しない |
| コード品質 | ★★★★☆ | 設計・構造は良い |
| 将来拡張性 | ★★★★☆ | web_search等の追加で実効性が出る |

現状は**過剰設計（over-engineering）の状態**。アーキテクチャとしては拡張を見越した妥当な設計だが、`web_search` が実装されていない今、5つの戦略・オーケストレーター・履歴管理は事実上使われない。

---

## 6. 推奨アクション

短期的に取れる施策を優先度順に記載する。

### 優先度1：`ErrorConfig` のリトライで十分なケースを分離する

一時的障害（API timeout、接続エラー）は `executor.py` のステップ実行レベルでリトライすれば済む。replanに回す必要がない。現状の `executor.py` にはステップレベルのリトライロジックがないので、これを追加する方がreplanより実効性がある。

### 優先度2：`web_search` ツールを実装する

これが入れば `rag_search` 失敗時に `web_search` へ切り替えるFALLBACKが意味を持つ。Plannerのプロンプトにも既に `web_search (必要な場合)` の記載がある（replan.py L506）ので、ツール実装とToolRegistry登録だけで動く。

### 優先度3：replanを一旦無効化し、複雑さを減らす

`Executor.__init__` の `enable_replan=True` を `False` にしておき、web_search実装後に有効化する。現状はLLM APIコスト（Planner再呼び出し）を消費するだけで、回答品質の改善に寄与していない。

---

## 付録：分析対象ファイル一覧

| ファイル | 役割 |
|---------|------|
| `replan.py` | 動的リプランニングシステム（評価対象） |
| `planner.py` | 計画生成エージェント |
| `executor.py` | 計画実行エージェント |
| `tools.py` | ツール定義・実装（RAGSearchTool, ReasoningTool, AskUserTool） |
| `schemas.py` | Pydanticモデル定義（ActionType, PlanStep, ExecutionPlan等） |
| `config.py` | 設定管理（ReplanConfig, ErrorConfig等） |

