# executor.py - GRACE計画実行エージェント ドキュメント

**Version 1.0** | 最終更新: 2025-01-28

---

## 目次

1. [概要](#概要)
   - [主な責務](#主な責務)
   - [主要機能一覧](#主要機能一覧)
2. [アーキテクチャ構成図](#1-アーキテクチャ構成図)
   - [システム全体構成](#11-システム全体構成)
   - [データフロー](#12-データフロー)
3. [モジュール構成図](#2-モジュール構成図)
   - [内部モジュール構成](#21-内部モジュール構成)
   - [外部依存関係](#22-外部依存関係)
   - [内部依存モジュール](#23-内部依存モジュール)
4. [クラス・関数一覧表](#3-クラス関数一覧表)
   - [クラス一覧](#31-クラス一覧)
   - [関数一覧（カテゴリ別）](#32-関数一覧カテゴリ別)
5. [クラス・関数 IPO詳細](#4-クラス関数-ipo詳細)
   - [ExecutionState データクラス](#41-executionstate-データクラス)
   - [Executor クラス](#42-executor-クラス)
   - [ファクトリ関数](#43-ファクトリ関数)
6. [設定・定数](#5-設定定数)
   - [LEGACY_AGENT_AVAILABLE](#51-legacy_agent_available)
   - [GraceConfigから使用される設定](#52-graceconfigから使用される設定)
7. [使用例](#6-使用例)
   - [基本的なワークフロー](#61-基本的なワークフロー)
   - [コールバック付きの使用](#62-コールバック付きの使用)
   - [ジェネレータ版の使用](#63-ジェネレータ版の使用)
8. [エクスポート](#7-エクスポート)
9. [変更履歴](#8-変更履歴)
10. [付録: 依存関係図](#付録-依存関係図)
11. [付録: エラーハンドリング](#付録-エラーハンドリング)
12. [付録: ステータス遷移図](#付録-ステータス遷移図)

---

## 概要

`executor.py`は、GRACE（Guided Reasoning with Adaptive Confidence Execution）エージェントの計画実行コンポーネントです。Plannerが生成した`ExecutionPlan`を受け取り、各ステップを順次実行して結果を管理します。

### 主な責務

- 計画の順次実行（ブロッキング版/ジェネレータ版）
- ステップ間の依存関係管理
- ツールの呼び出しと結果管理
- 信頼度（Confidence）の計算と評価
- Human-in-the-Loop（HITL）介入処理
- 失敗時のリプラン連携
- 実行状態の追跡とコールバック通知

### 主要機能一覧

| 機能 | 説明 |
|------|------|
| `ExecutionState` | 実行状態管理データクラス |
| `ExecutionState.__post_init__()` | 全ステップをPENDINGで初期化 |
| `ExecutionState.get_completed_outputs()` | 成功したステップの出力を取得 |
| `ExecutionState.get_completed_sources()` | 成功したステップのソースを取得 |
| `ExecutionState.can_replan()` | リプラン可能か判定 |
| `ExecutionState.get_execution_time_ms()` | 実行時間（ミリ秒）を取得 |
| `Executor` | 計画実行エージェントクラス |
| `Executor.__init__()` | コンストラクタ（各種コンポーネントの初期化） |
| `Executor.execute_plan()` | 計画を同期実行（ブロッキング版） |
| `Executor.execute_plan_generator()` | 計画をジェネレータで実行（UI連携用） |
| `Executor._execute_step()` | 個別ステップの実行 |
| `Executor._check_dependencies()` | ステップの依存関係を確認 |
| `Executor._calculate_overall_confidence()` | 全体信頼度の計算 |
| `Executor.cancel()` | 実行をキャンセル |
| `Executor.resume()` | 実行を再開 |
| `create_executor()` | Executorインスタンスを作成するファクトリ関数 |

---

## 1. アーキテクチャ構成図

### 1.1 システム全体構成

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           GRACE Executor Architecture                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐                                                               │
│  │ Execution    │                                                               │
│  │    Plan      │ (from Planner)                                                │
│  └──────┬───────┘                                                               │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                          EXECUTOR MODULE                                 │   │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │   │
│  │  │                     ExecutionState (Dataclass)                     │  │   │
│  │  │  • plan              • step_results      • overall_confidence      │  │   │
│  │  │  • current_step_id   • step_statuses     • is_cancelled/paused     │  │   │
│  │  │  • replan_count      • intervention_request                        │  │   │
│  │  └────────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                          │   │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │   │
│  │  │                        Executor Class                              │  │   │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐  │  │   │
│  │  │  │ execute_plan()  │  │execute_plan_    │  │ _execute_step()    │  │  │   │
│  │  │  │  (blocking)     │  │  generator()    │  │                    │  │  │   │
│  │  │  └────────┬────────┘  └────────┬────────┘  └────────┬───────────┘  │  │   │
│  │  │           │                    │                    │              │  │   │
│  │  │           └────────────────────┼────────────────────┘              │  │   │
│  │  │                                │                                   │  │   │
│  │  │                                ▼                                   │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────────┐   │  │   │
│  │  │  │                    ToolRegistry                             │   │  │   │
│  │  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                │   │  │   │
│  │  │  │  │RAGSearch  │  │Reasoning  │  │ AskUser   │  ...           │   │  │   │
│  │  │  │  │   Tool    │  │   Tool    │  │   Tool    │                │   │  │   │
│  │  │  │  └───────────┘  └───────────┘  └───────────┘                │   │  │   │
│  │  │  └─────────────────────────────────────────────────────────────┘   │  │   │
│  │  │                                │                                   │  │   │
│  │  │                                ▼                                   │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────────┐   │  │   │
│  │  │  │              Confidence System (Phase 2)                    │   │  │   │
│  │  │  │  ┌─────────────────┐  ┌─────────────────┐                   │   │  │   │
│  │  │  │  │Confidence       │  │LLMSelfEvaluator │                   │   │  │   │
│  │  │  │  │  Calculator     │  │                 │                   │   │  │   │
│  │  │  │  └─────────────────┘  └─────────────────┘                   │   │  │   │
│  │  │  │  ┌─────────────────┐  ┌─────────────────┐                   │   │  │   │
│  │  │  │  │QueryCoverage    │  │Confidence       │                   │   │  │   │
│  │  │  │  │  Calculator     │  │  Aggregator     │                   │   │  │   │
│  │  │  │  └─────────────────┘  └─────────────────┘                   │   │  │   │
│  │  │  └─────────────────────────────────────────────────────────────┘   │  │   │
│  │  │                                │                                   │  │   │
│  │  │                                ▼                                   │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────────┐   │  │   │
│  │  │  │            Intervention System (Phase 3)                    │   │  │   │
│  │  │  │  ┌─────────────────┐                                        │   │  │   │
│  │  │  │  │Intervention     │ ──▶ NOTIFY / CONFIRM / ESCALATE        │   │  │   │
│  │  │  │  │   Handler       │                                        │   │  │   │
│  │  │  │  └─────────────────┘                                        │   │  │   │
│  │  │  └─────────────────────────────────────────────────────────────┘   │  │   │
│  │  │                                │                                   │  │   │
│  │  │                                ▼                                   │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────────┐   │  │   │
│  │  │  │              Replan System (Phase 4)                        │   │  │   │
│  │  │  │  ┌─────────────────┐                                        │   │  │   │
│  │  │  │  │Replan           │ ──▶ 失敗時の計画再生成                    │   │  │   │
│  │  │  │  │  Orchestrator   │                                        │   │  │   │
│  │  │  │  └─────────────────┘                                        │   │  │   │
│  │  │  └─────────────────────────────────────────────────────────────┘   │  │   │
│  │  └────────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────┐                                                               │
│  │ Execution    │                                                               │
│  │   Result     │───────────────────────────────────────────────────────────────┼──▶ User/UI
│  └──────────────┘                                                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 データフロー

1. Plannerから`ExecutionPlan`を受信
2. `ExecutionState`を初期化し、全ステップをPENDINGに設定
3. 各ステップを順次実行（依存関係を確認）
4. ツールを呼び出し、結果を取得
5. 信頼度を計算し、必要に応じて介入を処理
6. 失敗時はリプランを実行
7. `ExecutionResult`を生成して返却

---

## 2. モジュール構成図

### 2.1 内部モジュール構成

```
executor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[設定・定数]
  • LEGACY_AGENT_AVAILABLE       - Legacy Agentインポート可否フラグ

[データクラス]
  └── ExecutionState             - 実行状態管理
        ├── __post_init__()
        ├── get_completed_outputs()
        ├── get_completed_sources()
        ├── can_replan()
        └── get_execution_time_ms()

[クラス]
  └── Executor                   - 計画実行エージェント
        ├── __init__()
        │
        ├── [実行メソッド]
        │   ├── execute_plan()
        │   └── execute_plan_generator()
        │
        ├── [ステップ実行]
        │   ├── _execute_step()
        │   ├── _execute_legacy_agent_step()
        │   ├── _execute_fallback()
        │   └── _check_dependencies()
        │
        ├── [信頼度計算]
        │   ├── _calculate_step_confidence()
        │   ├── _llm_calculate_step_confidence()
        │   └── _calculate_overall_confidence()
        │
        ├── [介入処理]
        │   ├── _handle_intervention_notify()
        │   ├── _handle_intervention_confirm()
        │   ├── _handle_intervention_escalate()
        │   └── _handle_intervention_if_needed()
        │
        ├── [ユーティリティ]
        │   ├── _prepare_tool_kwargs()
        │   ├── _extract_sources()
        │   ├── _format_output()
        │   └── _create_execution_result()
        │
        └── [制御]
            ├── cancel()
            └── resume()

[ファクトリ関数]
  └── create_executor()          - Executorインスタンス生成
```

### 2.2 外部依存関係

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| `logging` | 標準ライブラリ | ログ出力 |
| `time` | 標準ライブラリ | 実行時間計測 |
| `dataclasses` | 標準ライブラリ | データクラス定義 |

### 2.3 内部依存モジュール

| モジュール | 用途 |
|-----------|------|
| `grace.schemas` | ExecutionPlan, PlanStep, StepResult, ExecutionResult, StepStatus等のデータモデル |
| `grace.tools` | ToolRegistry, ToolResult, create_tool_registry |
| `grace.config` | get_config, GraceConfig設定管理 |
| `grace.confidence` | ConfidenceCalculator, LLMSelfEvaluator, ConfidenceAggregator等の信頼度計算 |
| `grace.intervention` | InterventionHandler, InterventionRequest, InterventionResponse等の介入処理 |
| `grace.replan` | ReplanOrchestrator, create_replan_orchestrator |
| `services.agent_service` | ReActAgent（オプション、Legacy Agent用） |

---

## 3. クラス・関数一覧表

### 3.1 クラス一覧

#### ExecutionState

| メソッド | 概要 |
|---------|------|
| `__post_init__()` | 全ステップをPENDINGで初期化 |
| `get_completed_outputs()` | 成功したステップの出力を取得 |
| `get_completed_sources()` | 成功したステップのソースを取得 |
| `can_replan()` | リプラン可能か判定 |
| `get_execution_time_ms()` | 実行時間（ミリ秒）を取得 |

#### Executor

| メソッド | 概要 |
|---------|------|
| `__init__(config, tool_registry, ...)` | コンストラクタ（各種コンポーネントの初期化） |
| `execute_plan(plan)` | 計画を同期実行（ブロッキング版） |
| `execute_plan_generator(plan, state)` | 計画をジェネレータで実行（UI連携用） |
| `_execute_step(step, state)` | 個別ステップの実行 |
| `_execute_legacy_agent_step(step, state, start_time)` | Legacy ReActAgentを使用したステップ実行 |
| `_execute_fallback(step, state, start_time)` | フォールバックアクションの実行 |
| `_check_dependencies(step, state)` | ステップの依存関係を確認 |
| `_prepare_tool_kwargs(step, state)` | ツール実行引数の準備 |
| `_calculate_step_confidence(tool_result)` | ステップ信頼度の計算（Heuristic版） |
| `_llm_calculate_step_confidence(tool_result, step, state)` | ステップ信頼度の計算（LLM版） |
| `_calculate_overall_confidence(state)` | 全体信頼度の計算 |
| `_extract_sources(tool_result)` | ツール結果からソースを抽出 |
| `_format_output(output)` | 出力を文字列にフォーマット |
| `_create_execution_result(state)` | ExecutionResultを生成 |
| `_handle_intervention_notify(message)` | NOTIFYレベルの介入処理 |
| `_handle_intervention_confirm(request)` | CONFIRMレベルの介入処理 |
| `_handle_intervention_escalate(request)` | ESCALATEレベルの介入処理 |
| `_handle_intervention_if_needed(action_decision, step, state)` | 介入が必要か判定して処理 |
| `cancel(state)` | 実行をキャンセル |
| `resume(state)` | 実行を再開 |

### 3.2 関数一覧（カテゴリ別）

#### ファクトリ関数

| 関数名 | 概要 |
|-------|------|
| `create_executor(config, tool_registry, **kwargs)` | Executorインスタンスを作成 |

---

## 4. クラス・関数 IPO詳細

### 4.1 ExecutionState データクラス

実行状態管理。計画の実行状態、ステップ結果、信頼度、制御フラグなどを保持します。

#### フィールド一覧

| フィールド | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `plan` | ExecutionPlan | - | 実行中の計画 |
| `current_step_id` | int | 0 | 現在実行中のステップID |
| `step_results` | Dict[int, StepResult] | {} | ステップID → 結果のマッピング |
| `step_statuses` | Dict[int, StepStatus] | {} | ステップID → ステータスのマッピング |
| `overall_confidence` | float | 0.0 | 全体の信頼度スコア (0.0-1.0) |
| `is_cancelled` | bool | False | キャンセルフラグ |
| `is_paused` | bool | False | 一時停止フラグ |
| `intervention_request` | Optional[Any] | None | 保留中の介入リクエスト |
| `replan_count` | int | 0 | リプラン実行回数 |
| `max_replans` | int | 3 | 最大リプラン回数 |
| `start_time` | Optional[float] | None | 実行開始時刻 |
| `end_time` | Optional[float] | None | 実行終了時刻 |

---

#### メソッド: `__post_init__`

**概要**: データクラス初期化後の処理。全ステップのステータスをPENDINGで初期化します。

```python
def __post_init__(self) -> None
```

| 項目 | 内容 |
|------|------|
| **Input** | なし（selfのみ） |
| **Process** | 計画内の全ステップのステータスを`StepStatus.PENDING`で初期化 |
| **Output** | なし（`self.step_statuses`が初期化された状態） |

---

#### メソッド: `get_completed_outputs`

**概要**: 成功したステップの出力を取得します。

```python
def get_completed_outputs(self) -> Dict[int, str]
```

| 項目 | 内容 |
|------|------|
| **Input** | なし（selfのみ） |
| **Process** | statusが"success"のステップの出力を抽出 |
| **Output** | `Dict[int, str]`: ステップID → 出力のマッピング |

**戻り値例**:
```python
{
    1: "検索結果: 『金色夜叉』は尾崎紅葉の作品です...",
    2: "尾崎紅葉は明治時代の小説家で..."
}
```

---

#### メソッド: `get_completed_sources`

**概要**: 成功したステップのソースを取得します。

```python
def get_completed_sources(self) -> List[str]
```

| 項目 | 内容 |
|------|------|
| **Input** | なし（selfのみ） |
| **Process** | statusが"success"でsourcesが存在するステップからソースを収集 |
| **Output** | `List[str]`: ソースURLや参照のリスト |

**戻り値例**:
```python
["wikipedia_ja:尾崎紅葉", "wikipedia_ja:金色夜叉"]
```

---

#### メソッド: `can_replan`

**概要**: リプラン可能か判定します。

```python
def can_replan(self) -> bool
```

| 項目 | 内容 |
|------|------|
| **Input** | なし（selfのみ） |
| **Process** | リプラン回数が上限未満かつキャンセルされていないか確認 |
| **Output** | `bool`: リプラン可能ならTrue |

---

#### メソッド: `get_execution_time_ms`

**概要**: 実行時間をミリ秒で取得します。

```python
def get_execution_time_ms(self) -> Optional[int]
```

| 項目 | 内容 |
|------|------|
| **Input** | なし（selfのみ） |
| **Process** | start_timeからend_time（またはcurrent time）までの経過時間を計算 |
| **Output** | `Optional[int]`: 実行時間（ミリ秒）、start_timeがNoneの場合はNone |

**戻り値例**:
```python
1234  # 1.234秒
```

---

### 4.2 Executor クラス

計画実行エージェント（GRACEネイティブ実装）。ToolRegistry、Confidenceシステム、Interventionシステム、Replanシステムを統合して計画を実行します。

#### コンストラクタ: `__init__`

**概要**: Executorインスタンスを初期化します。設定、ToolRegistry、各種Confidenceコンポーネント、コールバック、InterventionHandler、ReplanOrchestratorを設定します。

```python
Executor(
    config: Optional[GraceConfig] = None,
    tool_registry: Optional[ToolRegistry] = None,
    on_step_start: Optional[Callable[[PlanStep], None]] = None,
    on_step_complete: Optional[Callable[[StepResult], None]] = None,
    on_intervention_required: Optional[Callable[[str, Dict], Any]] = None,
    on_confidence_update: Optional[Callable[[ConfidenceScore, ActionDecision], None]] = None,
    on_replan: Optional[Callable[[str, int], None]] = None,
    replan_orchestrator: Optional[ReplanOrchestrator] = None,
    enable_replan: bool = True
)
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定（Noneの場合はデフォルト設定を使用） |
| `tool_registry` | Optional[ToolRegistry] | None | ツールレジストリ（Noneの場合はデフォルト作成） |
| `on_step_start` | Optional[Callable] | None | ステップ開始時コールバック |
| `on_step_complete` | Optional[Callable] | None | ステップ完了時コールバック |
| `on_intervention_required` | Optional[Callable] | None | 介入要求時コールバック |
| `on_confidence_update` | Optional[Callable] | None | 信頼度更新時コールバック |
| `on_replan` | Optional[Callable] | None | リプラン発生時コールバック |
| `replan_orchestrator` | Optional[ReplanOrchestrator] | None | リプランオーケストレーター |
| `enable_replan` | bool | True | リプラン機能の有効/無効 |

| 項目 | 内容 |
|------|------|
| **Input** | 上記パラメータ |
| **Process** | 1. 設定の取得<br>2. ToolRegistryの初期化<br>3. Confidenceコンポーネントの初期化<br>4. コールバックの設定<br>5. InterventionHandlerの初期化<br>6. ReplanOrchestratorの初期化 |
| **Output** | Executorインスタンス |

```python
# 使用例
from grace.executor import Executor
from grace.config import get_config

# デフォルト設定で初期化
executor = Executor()

# カスタム設定で初期化
config = get_config("config/custom.yml")
executor = Executor(config=config, enable_replan=False)
```

---

#### メソッド: `execute_plan`

**概要**: 計画を同期実行します（ブロッキング版）。全ステップを順次実行し、最終結果を返します。

```python
def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `plan` | ExecutionPlan | - | 実行する計画 |

| 項目 | 内容 |
|------|------|
| **Input** | `plan: ExecutionPlan` |
| **Process** | 1. ExecutionStateの初期化<br>2. 各ステップを順次実行<br>3. 依存関係の確認<br>4. ツール実行と結果保存<br>5. 失敗時のリプラン処理<br>6. 全体信頼度の計算<br>7. ExecutionResultの生成 |
| **Output** | `ExecutionResult`: 実行結果 |

**戻り値例**:
```python
ExecutionResult(
    plan_id="plan_20250128_123456_abc123",
    original_query="『金色夜叉』の作者は誰ですか？",
    final_answer="『金色夜叉』の作者は尾崎紅葉です。",
    step_results=[...],
    overall_confidence=0.85,
    overall_status="success",
    replan_count=0,
    total_execution_time_ms=1234
)
```

```python
# 使用例
from grace.executor import create_executor
from grace.planner import create_planner

planner = create_planner()
plan = planner.create_plan("『金色夜叉』の作者は誰ですか？")

executor = create_executor()
result = executor.execute_plan(plan)

print(f"ステータス: {result.overall_status}")
print(f"信頼度: {result.overall_confidence:.2f}")
print(f"回答: {result.final_answer}")
```

---

#### メソッド: `execute_plan_generator`

**概要**: 計画をジェネレータで実行します（UI連携用）。各ステップ完了後に状態をyieldし、リアルタイム表示を可能にします。

```python
def execute_plan_generator(
    self,
    plan: ExecutionPlan,
    state: Optional[ExecutionState] = None
) -> Generator[ExecutionState, None, ExecutionResult]
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `plan` | ExecutionPlan | - | 実行する計画 |
| `state` | Optional[ExecutionState] | None | 既存の状態（再開時に指定） |

| 項目 | 内容 |
|------|------|
| **Input** | `plan: ExecutionPlan`, `state: Optional[ExecutionState] = None` |
| **Process** | execute_planと同様だが、各ステップ完了時に`yield state`で状態を返す |
| **Yields** | `ExecutionState`: 各ステップ完了後の状態 |
| **Returns** | `ExecutionResult`: 最終実行結果 |

```python
# 使用例
from grace.executor import create_executor

executor = create_executor()
generator = executor.execute_plan_generator(plan)

try:
    while True:
        state = next(generator)
        print(f"現在のステップ: {state.current_step_id}")
        print(f"完了ステップ: {list(state.step_results.keys())}")

        if state.is_paused and state.intervention_request:
            # UIで介入処理
            handle_intervention(state.intervention_request)
            state.is_paused = False

except StopIteration as e:
    result = e.value
    print(f"完了: {result.overall_status}")
```

---

#### メソッド: `_execute_step`

**概要**: 個別ステップを実行します。ツールを取得し、引数を準備して実行、信頼度を計算してStepResultを返します。

```python
def _execute_step(self, step: PlanStep, state: ExecutionState) -> Any
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `step` | PlanStep | - | 実行するステップ |
| `state` | ExecutionState | - | 現在の実行状態 |

| 項目 | 内容 |
|------|------|
| **Input** | `step: PlanStep`, `state: ExecutionState` |
| **Process** | 1. ツールをレジストリから取得<br>2. ツール引数を準備<br>3. ツールを実行<br>4. 信頼度を計算<br>5. ソースを抽出<br>6. StepResultを構築<br>7. 失敗時はフォールバック実行 |
| **Output** | `StepResult` または `Generator[Any, None, StepResult]` |

---

#### メソッド: `_check_dependencies`

**概要**: ステップの依存関係を確認します。依存するステップが全て成功しているか確認します。

```python
def _check_dependencies(self, step: PlanStep, state: ExecutionState) -> bool
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `step` | PlanStep | - | 確認するステップ |
| `state` | ExecutionState | - | 現在の実行状態 |

| 項目 | 内容 |
|------|------|
| **Input** | `step: PlanStep`, `state: ExecutionState` |
| **Process** | depends_onの各ステップIDが結果に存在し、failedでないことを確認 |
| **Output** | `bool`: 依存関係が満たされていればTrue |

---

#### メソッド: `_calculate_overall_confidence`

**概要**: 全体の信頼度を計算します。各ステップのConfidenceScore、LLM自己評価、クエリ網羅度を統合します。

```python
def _calculate_overall_confidence(self, state: ExecutionState) -> float
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `state` | ExecutionState | - | 最終実行状態 |

| 項目 | 内容 |
|------|------|
| **Input** | `state: ExecutionState` |
| **Process** | 1. 各ステップのConfidenceScoreを収集<br>2. LLMSelfEvaluatorで最終回答を評価<br>3. QueryCoverageCalculatorでクエリ網羅度を評価<br>4. ConfidenceAggregatorで統合 |
| **Output** | `float`: 全体信頼度スコア (0.0-1.0) |

**戻り値例**:
```python
0.85  # 85%の信頼度
```

---

#### メソッド: `cancel`

**概要**: 実行をキャンセルします。

```python
def cancel(self, state: ExecutionState) -> None
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `state` | ExecutionState | - | キャンセルする実行状態 |

| 項目 | 内容 |
|------|------|
| **Input** | `state: ExecutionState` |
| **Process** | `state.is_cancelled = True`を設定 |
| **Output** | なし |

---

#### メソッド: `resume`

**概要**: 実行を再開します。

```python
def resume(self, state: ExecutionState) -> None
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `state` | ExecutionState | - | 再開する実行状態 |

| 項目 | 内容 |
|------|------|
| **Input** | `state: ExecutionState` |
| **Process** | `state.is_paused = False`を設定 |
| **Output** | なし |

---

### 4.3 ファクトリ関数

#### `create_executor`

**概要**: Executorインスタンスを作成するファクトリ関数です。

```python
def create_executor(
    config: Optional[GraceConfig] = None,
    tool_registry: Optional[ToolRegistry] = None,
    **kwargs
) -> Executor
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |
| `tool_registry` | Optional[ToolRegistry] | None | ツールレジストリ |
| `**kwargs` | Any | - | 各種コールバック等 |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig] = None`, `tool_registry: Optional[ToolRegistry] = None`, `**kwargs` |
| **Process** | Executorコンストラクタを呼び出してインスタンスを生成 |
| **Output** | `Executor`: Executorインスタンス |

```python
# 使用例
from grace.executor import create_executor

# デフォルト設定で作成
executor = create_executor()

# コールバック付きで作成
def on_step_complete(result):
    print(f"ステップ {result.step_id} 完了")

executor = create_executor(on_step_complete=on_step_complete)
```

---

## 5. 設定・定数

### 5.1 LEGACY_AGENT_AVAILABLE

Legacy Agent（ReActAgent）のインポート可否を示すフラグ。

```python
LEGACY_AGENT_AVAILABLE: bool
# services.agent_service のインポートに成功した場合は True
```

### 5.2 GraceConfigから使用される設定

| 設定パス | 型 | デフォルト | 説明 |
|---------|-----|----------|------|
| `llm.model` | str | `gemini-2.5-flash` | LLMモデル名 |
| `qdrant.url` | str | `http://localhost:6333` | QdrantサーバーURL |
| `qdrant.search_priority` | list | `["wikipedia_ja", ...]` | 検索優先順序 |
| `confidence.weights.*` | float | 各種 | 信頼度計算の重み |
| `confidence.thresholds.*` | float | 各種 | 介入レベルの閾値 |
| `replan.max_replans` | int | 3 | 最大リプラン回数 |

---

## 6. 使用例

### 6.1 基本的なワークフロー

```python
from grace.executor import create_executor
from grace.planner import create_planner

# 1. Plannerインスタンスを作成
planner = create_planner()

# 2. 計画を生成
query = "『金色夜叉』の作者は誰ですか？"
plan = planner.create_plan(query)

# 3. Executorインスタンスを作成
executor = create_executor()

# 4. 計画を実行
result = executor.execute_plan(plan)

# 5. 結果を確認
print(f"ステータス: {result.overall_status}")
print(f"信頼度: {result.overall_confidence:.2f}")
print(f"回答: {result.final_answer}")
print(f"実行時間: {result.total_execution_time_ms}ms")

# 出力例:
# ステータス: success
# 信頼度: 0.85
# 回答: 『金色夜叉』の作者は尾崎紅葉です。
# 実行時間: 1234ms
```

### 6.2 コールバック付きの使用

```python
from grace.executor import create_executor

def on_step_start(step):
    print(f"▶ ステップ {step.step_id} 開始: {step.description}")

def on_step_complete(result):
    status = "✓" if result.status == "success" else "✗"
    print(f"{status} ステップ {result.step_id} 完了: 信頼度={result.confidence:.2f}")

def on_intervention(type, data):
    if type == "confirm":
        return input(f"確認: {data['message']} (proceed/cancel): ")
    elif type == "escalate":
        return input(f"入力が必要: {data['message']}: ")
    return None

def on_confidence_update(score, decision):
    print(f"  信頼度更新: {score.score:.2f} -> {decision.level.value}")

executor = create_executor(
    on_step_start=on_step_start,
    on_step_complete=on_step_complete,
    on_intervention_required=on_intervention,
    on_confidence_update=on_confidence_update
)

result = executor.execute_plan(plan)
```

### 6.3 ジェネレータ版の使用

```python
from grace.executor import create_executor, ExecutionState

executor = create_executor()

# ジェネレータで実行（各ステップ後に状態を取得）
generator = executor.execute_plan_generator(plan)

try:
    while True:
        state = next(generator)

        # 進捗表示
        completed = len(state.step_results)
        total = len(state.plan.steps)
        print(f"進捗: {completed}/{total} ステップ完了")

        # 一時停止状態のチェック
        if state.is_paused and state.intervention_request:
            # UIで介入処理
            req = state.intervention_request
            print(f"介入要求: {req.message}")
            user_input = input("応答: ")
            # 応答を処理...
            state.is_paused = False

except StopIteration as e:
    result = e.value
    print(f"\n完了: {result.overall_status}")
    print(f"最終信頼度: {result.overall_confidence:.2f}")
```

---

## 7. エクスポート

`executor.py`でエクスポートされる要素：

```python
__all__ = [
    # データクラス
    "ExecutionState",
    # クラス
    "Executor",
    # ファクトリ関数
    "create_executor",
]
```

---

## 8. 変更履歴

| バージョン | 変更内容 |
|-----------|---------|
| 0.1.0 | 初版作成 |
| 1.0 | ドキュメント改修: a_md_doc_format.md v1.2に準拠、主な責務・主要機能一覧・IPO詳細に「**概要**:」ラベルを追加 |

---

## 付録: 依存関係図

```
executor.py
    │
    ├──► grace.schemas (内部)
    │        └── ExecutionPlan
    │        └── PlanStep
    │        └── StepResult
    │        └── ExecutionResult
    │        └── StepStatus
    │        └── create_plan_id()
    │
    ├──► grace.tools (内部)
    │        └── ToolRegistry
    │        └── ToolResult
    │        └── create_tool_registry()
    │
    ├──► grace.config (内部)
    │        └── get_config()
    │        └── GraceConfig
    │
    ├──► grace.confidence (内部)
    │        └── ConfidenceCalculator
    │        └── ConfidenceFactors
    │        └── ConfidenceScore
    │        └── LLMSelfEvaluator
    │        └── ConfidenceAggregator
    │        └── ActionDecision
    │        └── InterventionLevel
    │        └── create_confidence_calculator()
    │        └── create_llm_evaluator()
    │        └── create_confidence_aggregator()
    │        └── create_query_coverage_calculator()
    │
    ├──► grace.intervention (内部)
    │        └── InterventionHandler
    │        └── InterventionRequest
    │        └── InterventionResponse
    │        └── InterventionAction
    │        └── create_intervention_handler()
    │
    ├──► grace.replan (内部)
    │        └── ReplanOrchestrator
    │        └── create_replan_orchestrator()
    │
    └──► services.agent_service (外部/オプション)
             └── ReActAgent
             └── get_available_collections_from_qdrant_helper()
```

---

## 付録: エラーハンドリング

### ツール実行失敗時

| 状況 | 動作 | ログレベル |
|-----|------|----------|
| ツール実行エラー | `_execute_fallback()`を使用 | ERROR |
| フォールバック成功 | 結果を返却 | INFO |
| フォールバック失敗 | status="failed"で結果を返却 | ERROR |

### 依存関係未達時

| 状況 | 動作 | ログレベル |
|-----|------|----------|
| 依存ステップ未完了 | ステップをスキップ（SKIPPED） | WARNING |
| 依存ステップ失敗 | ステップをスキップ（SKIPPED） | WARNING |

### 全体実行失敗時

| 状況 | 動作 | ログレベル |
|-----|------|----------|
| 例外発生 | overall_status="failed"で結果を返却 | ERROR |
| 回答 | "実行エラー: {error_message}" | - |
| 信頼度 | 0.0 | - |

### リプラン関連

| 状況 | 動作 | ログレベル |
|-----|------|----------|
| リプラン成功 | 新しい計画で再実行 | INFO |
| リプラン上限到達 | 元の結果をそのまま返却 | INFO |
| リプラン失敗 | 元の計画の結果をそのまま返却 | WARNING |

---

## 付録: ステータス遷移図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        StepStatus State Machine                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌───────────┐                                        │
│           初期化 ────▶  │  PENDING  │                                        │
│                        └─────┬─────┘                                        │
│                              │                                              │
│              ┌───────────────┼───────────────┐                              │
│              │               │               │                              │
│              ▼               ▼               ▼                              │
│       ┌──────────┐    ┌──────────┐    ┌──────────┐                          │
│       │ SKIPPED  │    │ RUNNING  │    │（待機）   │                          │
│       └──────────┘    └─────┬────┘    └──────────┘                          │
│       (依存関係NG)            │                                              │
│                   ┌─────────┴─────────┐                                     │
│                   │                   │                                     │
│                   ▼                   ▼                                     │
│            ┌──────────┐        ┌──────────┐                                 │
│            │ SUCCESS  │        │  FAILED  │                                 │
│            └──────────┘        └─────┬────┘                                 │
│                                      │                                      │
│                                      ▼                                      │
│                               ┌──────────────┐                              │
│                               │   Replan?    │                              │
│                               └──────┬───────┘                              │
│                                      │                                      │
│                          ┌───────────┴───────────┐                          │
│                          │ Yes                   │ No                       │
│                          ▼                       ▼                          │
│                   ┌──────────────┐        ┌──────────────┐                  │
│                   │ 新しい計画で   │        │ 結果を返却     │                  │
│                   │ 再実行        │        │              │                  │
│                   └──────────────┘        └──────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

