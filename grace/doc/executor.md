# GRACE Executor モジュール ドキュメント

**Version:** 0.1.0
**Module:** `grace/executor.py`
**Last Updated:** 2025-01-28

---

## 目次

1. [概要](#概要)
2. [アーキテクチャ構成図](#アーキテクチャ構成図)
3. [モジュール構成図](#モジュール構成図)
4. [クラス・関数一覧](#クラス関数一覧)
5. [クラス・関数のIPO（Input/Process/Output）](#クラス関数のipoinputprocessoutput)
6. [実行フロー](#実行フロー)
7. [コールバックシステム](#コールバックシステム)
8. [依存関係](#依存関係)
9. [設定項目](#設定項目)
10. [使用例](#使用例)
11. [エラーハンドリング](#エラーハンドリング)

---

## 概要

`executor.py` は GRACE (Guided Reasoning with Adaptive Confidence Execution) エージェントの**計画実行コンポーネント**です。Planner が生成した `ExecutionPlan` を受け取り、各ステップを順次実行して結果を管理します。

### 主な責務

- 計画の順次実行（ブロッキング/ジェネレータ版）
- ステップ間の依存関係管理
- ツールの呼び出しと結果管理
- 信頼度（Confidence）の計算と評価
- Human-in-the-Loop（HITL）介入処理
- 失敗時のリプラン連携
- 実行状態の追跡とコールバック通知

### GRACE フェーズとの対応

| フェーズ | 機能 | 実装状況 |
|---------|------|---------|
| Phase 1 | 基本計画実行 | ✅ 完了 |
| Phase 2 | 信頼度計算 | ✅ 完了 |
| Phase 3 | HITL介入 | ✅ 完了 |
| Phase 4 | 適応型リプラン | ✅ 完了 |

---

## アーキテクチャ構成図

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
│  │  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐               │   │  │   │
│  │  │  │  │RAGSearch  │  │Reasoning  │  │ AskUser   │  ...          │   │  │   │
│  │  │  │  │   Tool    │  │   Tool    │  │   Tool    │               │   │  │   │
│  │  │  │  └───────────┘  └───────────┘  └───────────┘               │   │  │   │
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
│  │  │  │  │Intervention     │ ──▶ NOTIFY / CONFIRM / ESCALATE       │   │  │   │
│  │  │  │  │   Handler       │                                        │   │  │   │
│  │  │  │  └─────────────────┘                                        │   │  │   │
│  │  │  └─────────────────────────────────────────────────────────────┘   │  │   │
│  │  │                                │                                   │  │   │
│  │  │                                ▼                                   │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────────┐   │  │   │
│  │  │  │              Replan System (Phase 4)                        │   │  │   │
│  │  │  │  ┌─────────────────┐                                        │   │  │   │
│  │  │  │  │Replan           │ ──▶ 失敗時の計画再生成                 │   │  │   │
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

---

## モジュール構成図

```
grace/executor.py
│
├── [定数] LEGACY_AGENT_AVAILABLE
│       └── Legacy Agent (ReActAgent) のインポート可否フラグ
│
├── [データクラス] ExecutionState
│       ├── plan: ExecutionPlan
│       ├── current_step_id: int
│       ├── step_results: Dict[int, StepResult]
│       ├── step_statuses: Dict[int, StepStatus]
│       ├── overall_confidence: float
│       ├── is_cancelled / is_paused: bool
│       ├── intervention_request: Optional[InterventionRequest]
│       ├── replan_count: int
│       ├── start_time / end_time: Optional[float]
│       │
│       └── [メソッド]
│           ├── __post_init__()
│           ├── get_completed_outputs()
│           ├── get_completed_sources()
│           ├── can_replan()
│           └── get_execution_time_ms()
│
├── [クラス] Executor
│       ├── __init__()
│       │
│       ├── [実行メソッド]
│       │   ├── execute_plan()              # ブロッキング版
│       │   └── execute_plan_generator()    # ジェネレータ版
│       │
│       ├── [ステップ実行]
│       │   ├── _execute_step()
│       │   ├── _execute_legacy_agent_step()
│       │   ├── _execute_fallback()
│       │   └── _check_dependencies()
│       │
│       ├── [信頼度計算]
│       │   ├── _calculate_step_confidence()       # Heuristic版
│       │   ├── _llm_calculate_step_confidence()   # LLM版
│       │   └── _calculate_overall_confidence()
│       │
│       ├── [介入処理]
│       │   ├── _handle_intervention_notify()
│       │   ├── _handle_intervention_confirm()
│       │   ├── _handle_intervention_escalate()
│       │   └── _handle_intervention_if_needed()
│       │
│       ├── [ユーティリティ]
│       │   ├── _prepare_tool_kwargs()
│       │   ├── _extract_sources()
│       │   ├── _format_output()
│       │   └── _create_execution_result()
│       │
│       └── [制御]
│           ├── cancel()
│           └── resume()
│
└── [ファクトリ] create_executor()

依存モジュール:
├── grace.schemas (ExecutionPlan, PlanStep, StepResult, ExecutionResult, StepStatus)
├── grace.tools (ToolRegistry, ToolResult, create_tool_registry)
├── grace.config (get_config, GraceConfig)
├── grace.confidence (ConfidenceCalculator, LLMSelfEvaluator, ConfidenceAggregator, ...)
├── grace.intervention (InterventionHandler, InterventionRequest, InterventionResponse, ...)
├── grace.replan (ReplanOrchestrator, create_replan_orchestrator)
└── services.agent_service (ReActAgent) [Optional]
```

---

## クラス・関数一覧

### ExecutionState データクラス

| フィールド/メソッド | 種別 | 概要 |
|-------------------|------|------|
| `plan` | フィールド | 実行中の `ExecutionPlan` |
| `current_step_id` | フィールド | 現在実行中のステップID |
| `step_results` | フィールド | ステップID → `StepResult` のマッピング |
| `step_statuses` | フィールド | ステップID → `StepStatus` のマッピング |
| `overall_confidence` | フィールド | 全体の信頼度スコア (0.0-1.0) |
| `is_cancelled` | フィールド | キャンセルフラグ |
| `is_paused` | フィールド | 一時停止フラグ |
| `intervention_request` | フィールド | 保留中の介入リクエスト |
| `replan_count` | フィールド | リプラン実行回数 |
| `max_replans` | フィールド | 最大リプラン回数（デフォルト: 3） |
| `start_time` / `end_time` | フィールド | 実行開始/終了時刻 |
| `__post_init__()` | メソッド | 全ステップを PENDING で初期化 |
| `get_completed_outputs()` | メソッド | 成功したステップの出力を取得 |
| `get_completed_sources()` | メソッド | 成功したステップのソースを取得 |
| `can_replan()` | メソッド | リプラン可能か判定 |
| `get_execution_time_ms()` | メソッド | 実行時間（ミリ秒）を取得 |

### Executor クラス

| メソッド名 | 種別 | 概要 |
|-----------|------|------|
| `__init__` | コンストラクタ | Executor の初期化（ToolRegistry、Confidence、Intervention、Replan コンポーネントを設定） |
| `execute_plan` | パブリック | 計画を同期実行（ブロッキング版） |
| `execute_plan_generator` | パブリック | 計画をジェネレータで実行（UI連携用） |
| `_execute_step` | プライベート | 個別ステップの実行 |
| `_execute_legacy_agent_step` | プライベート | Legacy ReActAgent を使用したステップ実行 |
| `_execute_fallback` | プライベート | フォールバックアクションの実行 |
| `_check_dependencies` | プライベート | ステップの依存関係を確認 |
| `_prepare_tool_kwargs` | プライベート | ツール実行引数の準備 |
| `_calculate_step_confidence` | プライベート | ステップ信頼度の計算（Heuristic版） |
| `_llm_calculate_step_confidence` | プライベート | ステップ信頼度の計算（LLM版） |
| `_calculate_overall_confidence` | プライベート | 全体信頼度の計算 |
| `_extract_sources` | プライベート | ツール結果からソースを抽出 |
| `_format_output` | プライベート | 出力を文字列にフォーマット |
| `_create_execution_result` | プライベート | `ExecutionResult` を生成 |
| `_handle_intervention_notify` | プライベート | NOTIFY レベルの介入処理 |
| `_handle_intervention_confirm` | プライベート | CONFIRM レベルの介入処理 |
| `_handle_intervention_escalate` | プライベート | ESCALATE レベルの介入処理 |
| `_handle_intervention_if_needed` | プライベート | 介入が必要か判定して処理 |
| `cancel` | パブリック | 実行をキャンセル |
| `resume` | パブリック | 実行を再開 |

### ファクトリ関数

| 関数名 | 概要 |
|--------|------|
| `create_executor` | Executor インスタンスを作成するファクトリ関数 |

### エクスポート一覧

```python
__all__ = [
    "ExecutionState",
    "Executor",
    "create_executor",
]
```

---

## クラス・関数のIPO（Input/Process/Output）

### ExecutionState データクラス

#### `__post_init__()`

| 項目 | 内容 |
|------|------|
| **Input** | `self.plan.steps` - 計画内の全ステップ |
| **Process** | 全ステップのステータスを `StepStatus.PENDING` で初期化 |
| **Output** | `self.step_statuses` が初期化された状態 |

#### `get_completed_outputs()`

| 項目 | 内容 |
|------|------|
| **Input** | `self.step_results` - 実行済みステップの結果 |
| **Process** | status が "success" のステップの出力を抽出 |
| **Output** | `Dict[int, str]` - ステップID → 出力のマッピング |

#### `can_replan()`

| 項目 | 内容 |
|------|------|
| **Input** | `self.replan_count`, `self.max_replans`, `self.is_cancelled` |
| **Process** | リプラン回数が上限未満かつキャンセルされていないか確認 |
| **Output** | `bool` - リプラン可能なら True |

---

### Executor クラス

#### `__init__(config, tool_registry, on_step_start, on_step_complete, on_intervention_required, on_confidence_update, on_replan, replan_orchestrator, enable_replan)`

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]` - 設定<br>`tool_registry: Optional[ToolRegistry]` - ツールレジストリ<br>`on_step_start: Callable` - ステップ開始コールバック<br>`on_step_complete: Callable` - ステップ完了コールバック<br>`on_intervention_required: Callable` - 介入要求コールバック<br>`on_confidence_update: Callable` - 信頼度更新コールバック<br>`on_replan: Callable` - リプランコールバック<br>`replan_orchestrator: Optional[ReplanOrchestrator]`<br>`enable_replan: bool` |
| **Process** | 1. 設定の取得<br>2. ToolRegistry の初期化<br>3. Confidence コンポーネントの初期化<br>4. コールバックの設定<br>5. InterventionHandler の初期化<br>6. ReplanOrchestrator の初期化 |
| **Output** | `Executor` インスタンス |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          __init__ IPO Diagram                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                         PROCESS                         OUTPUT       │
│  ─────                         ───────                         ──────       │
│                                                                             │
│  config ─────────────────▶ ┌──────────────────────────────┐                 │
│  tool_registry ──────────▶ │ 1. get_config()              │                 │
│  on_step_start ──────────▶ │ 2. create_tool_registry()    │                 │
│  on_step_complete ───────▶ │ 3. Confidence components:    │                 │
│  on_intervention_required ▶│    - ConfidenceCalculator    │ ──▶ Executor   │
│  on_confidence_update ───▶ │    - LLMSelfEvaluator        │     Instance    │
│  on_replan ──────────────▶ │    - QueryCoverageCalculator │                 │
│  replan_orchestrator ────▶ │    - ConfidenceAggregator    │                 │
│  enable_replan ──────────▶ │ 4. InterventionHandler       │                 │
│                            │ 5. ReplanOrchestrator        │                 │
│                            └──────────────────────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### `execute_plan(plan)`

| 項目 | 内容 |
|------|------|
| **Input** | `plan: ExecutionPlan` - 実行する計画 |
| **Process** | 1. `ExecutionState` の初期化<br>2. 各ステップを順次実行<br>3. 依存関係の確認<br>4. ツール実行と結果保存<br>5. 失敗時のリプラン処理<br>6. 全体信頼度の計算<br>7. `ExecutionResult` の生成 |
| **Output** | `ExecutionResult` - 実行結果 |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        execute_plan IPO Diagram                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT              PROCESS                                    OUTPUT       │
│  ─────              ───────                                    ──────       │
│                                                                             │
│                     ┌──────────────────────────────────────┐                │
│                     │ 1. ExecutionState(plan=plan)         │                │
│                     │    ↓                                 │                │
│                     │ 2. for step in plan.steps:           │                │
│                     │    ├─ _check_dependencies()          │                │
│  plan ──────────▶   │    ├─ on_step_start callback         │                │
│  (ExecutionPlan)    │    ├─ _execute_step()                │ ──▶ Execution │
│                     │    ├─ Save result to state           │     Result     │
│                     │    ├─ on_step_complete callback      │                │
│                     │    └─ Replan if failed               │                │
│                     │    ↓                                 │                │
│                     │ 3. _calculate_overall_confidence()   │                │
│                     │    ↓                                 │                │
│                     │ 4. _create_execution_result()        │                │
│                     └──────────────────────────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### `execute_plan_generator(plan, state)`

| 項目 | 内容 |
|------|------|
| **Input** | `plan: ExecutionPlan` - 実行する計画<br>`state: Optional[ExecutionState]` - 既存の状態（再開時） |
| **Process** | `execute_plan` と同様だが、各ステップ完了時に `yield state` で状態を返す |
| **Yields** | `ExecutionState` - 各ステップ完了後の状態 |
| **Returns** | `ExecutionResult` - 最終実行結果 |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   execute_plan_generator IPO Diagram                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT              PROCESS                                    OUTPUT       │
│  ─────              ───────                                    ──────       │
│                                                                             │
│  plan ──────────▶   ┌──────────────────────────────────────┐                │
│  (ExecutionPlan)    │ 1. Initialize state                  │                │
│                     │    ↓                                 │                │
│  state ─────────▶   │ 2. for step in steps_to_execute:     │                │
│  (Optional)         │    ├─ _check_dependencies()          │ ──YIELD──▶    │
│                     │    ├─ _execute_step() (may yield)    │  Execution    │
│                     │    ├─ Save result                    │  State        │
│                     │    ├─ Intervention check             │                │
│                     │    │   (may yield & return)          │                │
│                     │    ├─ yield state ◀─────────────────┼────────────    │
│                     │    └─ Replan if needed               │                │
│                     │    ↓                                 │                │
│                     │ 3. _calculate_overall_confidence()   │                │
│                     │    ↓                                 │ ──RETURN──▶   │
│                     │ 4. return ExecutionResult            │  Execution    │
│                     └──────────────────────────────────────┘  Result        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### `_execute_step(step, state)`

| 項目 | 内容 |
|------|------|
| **Input** | `step: PlanStep` - 実行するステップ<br>`state: ExecutionState` - 現在の状態 |
| **Process** | 1. ツールをレジストリから取得<br>2. ツール引数を準備<br>3. ツールを実行<br>4. 信頼度を計算<br>5. ソースを抽出<br>6. `StepResult` を構築<br>7. 失敗時はフォールバック実行 |
| **Output** | `StepResult` または `Generator[Any, None, StepResult]` |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        _execute_step IPO Diagram                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT              PROCESS                                    OUTPUT       │
│  ─────              ───────                                    ──────       │
│                                                                             │
│  step ──────────▶   ┌──────────────────────────────────────┐                │
│  (PlanStep)         │ 1. tool = tool_registry.get(action)  │                │
│                     │    ↓                                 │                │
│  state ─────────▶   │ 2. kwargs = _prepare_tool_kwargs()   │                │
│  (ExecutionState)   │    ↓                                 │ ──▶ StepResult│
│                     │ 3. tool_result = tool.execute()      │     or        │
│                     │    ↓                                 │     Generator │
│                     │ 4. confidence = _llm_calculate_...() │                │
│                     │    ↓                                 │                │
│                     │ 5. sources = _extract_sources()      │                │
│                     │    ↓                                 │                │
│                     │ 6. return StepResult(...)            │                │
│                     │    [Error] → _execute_fallback()     │                │
│                     └──────────────────────────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### `_llm_calculate_step_confidence(tool_result, step, state)`

| 項目 | 内容 |
|------|------|
| **Input** | `tool_result: ToolResult` - ツール実行結果<br>`step: PlanStep` - 実行したステップ<br>`state: ExecutionState` - 現在の状態 |
| **Process** | 1. ツール結果から信頼度要素を抽出<br>2. ソース一致度を計算<br>3. 依存ステップからスコアを継承<br>4. `ConfidenceFactors` を構築<br>5. `ConfidenceCalculator.llm_calculate()` で計算<br>6. `ActionDecision` を取得<br>7. コールバックで通知 |
| **Output** | `float` - 信頼度スコア (0.0-1.0) |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               _llm_calculate_step_confidence IPO Diagram                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT              PROCESS                                    OUTPUT       │
│  ─────              ───────                                    ──────       │
│                                                                             │
│  tool_result ────▶  ┌──────────────────────────────────────┐                │
│  (ToolResult)       │ 1. Extract confidence_factors        │                │
│                     │    ↓                                 │                │
│  step ───────────▶  │ 2. Calculate source_agreement        │                │
│  (PlanStep)         │    (if multiple sources)             │                │
│                     │    ↓                                 │ ──▶ float     │
│  state ──────────▶  │ 3. Inherit scores from dependencies  │    (0.0-1.0)  │
│  (ExecutionState)   │    ↓                                 │                │
│                     │ 4. Build ConfidenceFactors           │                │
│                     │    ↓                                 │                │
│                     │ 5. confidence_calculator.llm_calculate│                │
│                     │    ↓                                 │                │
│                     │ 6. Store in step_confidence_scores   │                │
│                     │    ↓                                 │                │
│                     │ 7. on_confidence_update callback     │                │
│                     └──────────────────────────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### `_calculate_overall_confidence(state)`

| 項目 | 内容 |
|------|------|
| **Input** | `state: ExecutionState` - 実行状態 |
| **Process** | 1. 各ステップの `ConfidenceScore` を収集<br>2. LLMSelfEvaluator で最終回答を評価<br>3. QueryCoverageCalculator でクエリ網羅度を評価<br>4. ConfidenceAggregator で統合 |
| **Output** | `float` - 全体信頼度スコア (0.0-1.0) |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               _calculate_overall_confidence IPO Diagram                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT              PROCESS                                    OUTPUT       │
│  ─────              ───────                                    ──────       │
│                                                                             │
│                     ┌──────────────────────────────────────┐                │
│                     │ 1. Collect step_confidence_scores    │                │
│  state ──────────▶  │    ↓                                 │                │
│  (ExecutionState)   │ 2. Get final_answer from reasoning   │                │
│                     │    step                              │                │
│                     │    ↓                                 │ ──▶ float     │
│                     │ 3. llm_evaluator.evaluate()          │    (0.0-1.0)  │
│                     │    (Accuracy, Style, etc.)           │                │
│                     │    ↓                                 │                │
│                     │ 4. query_coverage_calculator.calc()  │                │
│                     │    ↓                                 │                │
│                     │ 5. confidence_aggregator.aggregate() │                │
│                     │    method="weighted"                 │                │
│                     └──────────────────────────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### `_prepare_tool_kwargs(step, state)`

| 項目 | 内容 |
|------|------|
| **Input** | `step: PlanStep` - 実行するステップ<br>`state: ExecutionState` - 現在の状態 |
| **Process** | アクション種別に応じて引数を準備:<br>- `rag_search`: query, collection<br>- `reasoning`: query, sources, context（依存ステップから）<br>- `ask_user`: question, reason, urgency |
| **Output** | `Dict[str, Any]` - ツール実行引数 |

---

#### `_handle_intervention_if_needed(action_decision, step, state)`

| 項目 | 内容 |
|------|------|
| **Input** | `action_decision: ActionDecision` - 信頼度に基づく判定<br>`step: PlanStep` - 現在のステップ<br>`state: ExecutionState` - 実行状態 |
| **Process** | 1. SILENT/NOTIFY は自動続行（NOTIFY はログ出力）<br>2. CONFIRM/ESCALATE は介入処理を実行<br>3. CANCEL の場合は `state.is_cancelled = True` |
| **Output** | `Optional[InterventionResponse]` - 介入レスポンス |

---

#### `_create_execution_result(state)`

| 項目 | 内容 |
|------|------|
| **Input** | `state: ExecutionState` - 最終実行状態 |
| **Process** | 1. 全体ステータスを判定（success/partial/failed/cancelled）<br>2. 最終回答を取得（最後の reasoning/legacy_agent ステップから）<br>3. `ExecutionResult` を構築 |
| **Output** | `ExecutionResult` - 実行結果オブジェクト |

---

### ファクトリ関数

#### `create_executor(config, tool_registry, **kwargs)`

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]`<br>`tool_registry: Optional[ToolRegistry]`<br>`**kwargs` - 各種コールバック |
| **Process** | `Executor` クラスをインスタンス化 |
| **Output** | `Executor` インスタンス |

---

## 実行フロー

### ステップ実行フロー

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       Step Execution Flow                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────┐                                                           │
│  │ PlanStep    │                                                           │
│  └──────┬──────┘                                                           │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────────────┐     No                                            │
│  │ Check Dependencies  │─────────▶ SKIP (StepStatus.SKIPPED)              │
│  └──────────┬──────────┘                                                   │
│             │ Yes                                                          │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ on_step_start()     │                                                   │
│  │ callback            │                                                   │
│  └──────────┬──────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ _prepare_tool_kwargs│                                                   │
│  └──────────┬──────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ ToolRegistry.get()  │                                                   │
│  │ tool.execute()      │                                                   │
│  └──────────┬──────────┘                                                   │
│             │                                                              │
│             ├──────────────────────────────────────┐                       │
│             │ Success                              │ Failure               │
│             ▼                                      ▼                       │
│  ┌─────────────────────┐              ┌─────────────────────┐              │
│  │ _llm_calculate_     │              │ _execute_fallback() │              │
│  │ step_confidence()   │              └──────────┬──────────┘              │
│  └──────────┬──────────┘                         │                         │
│             │                                    │                         │
│             ▼                                    │                         │
│  ┌─────────────────────┐                         │                         │
│  │ on_confidence_update│◀────────────────────────┘                         │
│  │ callback            │                                                   │
│  └──────────┬──────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐     CONFIRM/ESCALATE                              │
│  │ Intervention Check  │─────────────────────▶ Pause & Yield State        │
│  └──────────┬──────────┘                                                   │
│             │ SILENT/NOTIFY                                                │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ Save StepResult     │                                                   │
│  │ to state            │                                                   │
│  └──────────┬──────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ on_step_complete()  │                                                   │
│  │ callback            │                                                   │
│  └──────────┬──────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐     Yes (Failed)                                  │
│  │ Replan Check        │─────────────────────▶ ReplanOrchestrator         │
│  └──────────┬──────────┘                       → Recursive execute        │
│             │ No                                                           │
│             ▼                                                              │
│        Next Step                                                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## コールバックシステム

Executor は以下のコールバックをサポートしています:

| コールバック | 引数 | 説明 |
|-------------|------|------|
| `on_step_start` | `PlanStep` | ステップ実行開始時に呼び出し |
| `on_step_complete` | `StepResult` | ステップ実行完了時に呼び出し |
| `on_intervention_required` | `type: str`, `data: Dict` | 介入が必要な時に呼び出し（notify/confirm/escalate/ask_user） |
| `on_confidence_update` | `ConfidenceScore`, `ActionDecision` | 信頼度更新時に呼び出し |
| `on_replan` | `reason: str`, `count: int` | リプラン発生時に呼び出し |

### 介入タイプと応答

| タイプ | 信頼度 | UI動作 | 期待される応答 |
|--------|--------|--------|---------------|
| `notify` | 0.7-0.9 | 通知表示 | なし（自動続行） |
| `confirm` | 0.4-0.7 | 確認ダイアログ | "proceed" / "modify" / "cancel" |
| `escalate` | < 0.4 | 入力要求 | ユーザー入力テキスト |
| `ask_user` | - | 質問表示 | ユーザー回答 |

---

## 依存関係

### 内部モジュール

| モジュール | インポート項目 | 用途 |
|-----------|---------------|------|
| `grace.schemas` | `ExecutionPlan`, `PlanStep`, `StepResult`, `ExecutionResult`, `StepStatus`, `create_plan_id` | データ構造 |
| `grace.tools` | `ToolRegistry`, `ToolResult`, `create_tool_registry` | ツール実行 |
| `grace.config` | `get_config`, `GraceConfig` | 設定管理 |
| `grace.confidence` | `ConfidenceCalculator`, `LLMSelfEvaluator`, `ConfidenceAggregator`, `ConfidenceFactors`, `ConfidenceScore`, `ActionDecision`, `InterventionLevel` | 信頼度計算 |
| `grace.intervention` | `InterventionHandler`, `InterventionRequest`, `InterventionResponse`, `InterventionAction` | 介入処理 |
| `grace.replan` | `ReplanOrchestrator`, `create_replan_orchestrator` | リプラン処理 |

### 外部モジュール（オプション）

| モジュール | 用途 |
|-----------|------|
| `services.agent_service` | Legacy ReActAgent（フォールバック用） |

---

## 設定項目

`GraceConfig` から使用される設定:

| 設定パス | 型 | デフォルト | 説明 |
|---------|-----|----------|------|
| `llm.model` | str | `gemini-2.5-flash` | LLM モデル名 |
| `qdrant.url` | str | `http://localhost:6333` | Qdrant URL |
| `qdrant.search_priority` | list | `["wikipedia_ja", ...]` | 検索優先順序 |
| `confidence.weights.*` | float | 各種 | 信頼度計算の重み |
| `confidence.thresholds.*` | float | 各種 | 介入レベルの閾値 |
| `replan.max_replans` | int | 3 | 最大リプラン回数 |

---

## 使用例

### 基本的な使用（ブロッキング版）

```python
from grace.executor import create_executor
from grace.planner import create_planner

# 計画を生成
planner = create_planner()
plan = planner.create_plan("『金色夜叉』の作者は誰ですか？")

# 実行
executor = create_executor()
result = executor.execute_plan(plan)

print(f"ステータス: {result.overall_status}")
print(f"信頼度: {result.overall_confidence:.2f}")
print(f"回答: {result.final_answer}")
```

### コールバック付きの使用

```python
from grace.executor import create_executor

def on_step_start(step):
    print(f"▶ ステップ {step.step_id} 開始: {step.description}")

def on_step_complete(result):
    print(f"✓ ステップ {result.step_id} 完了: 信頼度={result.confidence:.2f}")

def on_intervention(type, data):
    if type == "confirm":
        return input(f"確認: {data['message']} (proceed/cancel): ")
    return None

executor = create_executor(
    on_step_start=on_step_start,
    on_step_complete=on_step_complete,
    on_intervention_required=on_intervention
)
```

### ジェネレータ版の使用（リアルタイム UI 連携）

```python
from grace.executor import create_executor, ExecutionState

executor = create_executor()

# ジェネレータで実行（各ステップ後に状態を取得）
generator = executor.execute_plan_generator(plan)

try:
    while True:
        state = next(generator)
        print(f"現在のステップ: {state.current_step_id}")
        print(f"完了ステップ: {list(state.step_results.keys())}")

        # 一時停止状態のチェック
        if state.is_paused and state.intervention_request:
            # UIで介入処理
            handle_intervention(state.intervention_request)
            state.is_paused = False

except StopIteration as e:
    result = e.value
    print(f"完了: {result.overall_status}")
```

---

## エラーハンドリング

### ツール実行失敗時

- **動作**: フォールバックアクションがあれば実行
- **ログ**: `ERROR` レベルでエラー内容を記録
- **信頼度**: 0.0 に設定

### 依存関係未達時

- **動作**: ステップをスキップ (`StepStatus.SKIPPED`)
- **ログ**: `WARNING` レベルで警告を記録

### 全体実行失敗時

- **動作**: `ExecutionResult` を `overall_status="failed"` で返却
- **回答**: `"実行エラー: {error_message}"`
- **信頼度**: 0.0

### リプラン失敗時

- **動作**: 元の計画の結果をそのまま返却
- **ログ**: `INFO` レベルでリプラン試行を記録

---

## ステータス遷移図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        StepStatus State Machine                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌───────────┐                                        │
│           初期化 ────▶ │  PENDING  │                                        │
│                        └─────┬─────┘                                        │
│                              │                                              │
│              ┌───────────────┼───────────────┐                              │
│              │               │               │                              │
│              ▼               ▼               ▼                              │
│       ┌──────────┐    ┌──────────┐    ┌──────────┐                          │
│       │ SKIPPED  │    │ RUNNING  │    │（待機）  │                          │
│       └──────────┘    └─────┬────┘    └──────────┘                          │
│                             │                                               │
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
│                               └──────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 関連ドキュメント

- [planner.md](./planner.md) - 計画生成エージェント
- [schemas.md](./schemas.md) - データモデル定義
- [tools.md](./tools.md) - ツール定義
- [confidence.md](./confidence.md) - 信頼度計算システム
- [intervention.md](./intervention.md) - 介入処理システム
- [replan.md](./replan.md) - リプラン処理システム

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 0.1.0 | 2025-01-28 | 初版作成 |
