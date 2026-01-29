# replan.py - 動的リプランニングシステム ドキュメント

**Version 1.2** | 最終更新: 2025-01-29

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
   - [Enum一覧](#31-enum一覧)
   - [データクラス一覧](#32-データクラス一覧)
   - [クラス一覧](#33-クラス一覧)
   - [ファクトリ関数一覧](#34-ファクトリ関数一覧)
5. [クラス・関数 IPO詳細](#4-クラス関数-ipo詳細)
   - [ReplanTrigger Enum](#41-replantrigger-enum)
   - [ReplanStrategy Enum](#42-replanstrategy-enum)
   - [ReplanContext データクラス](#43-replancontext-データクラス)
   - [ReplanResult データクラス](#44-replanresult-データクラス)
   - [ReplanManager クラス](#45-replanmanager-クラス)
   - [ReplanOrchestrator クラス](#46-replanorchestrator-クラス)
   - [ファクトリ関数](#47-ファクトリ関数)
6. [設定・定数](#5-設定定数)
7. [使用例](#6-使用例)
   - [基本的なワークフロー](#61-基本的なワークフロー)
   - [ユーザーフィードバックによるリプラン](#62-ユーザーフィードバックによるリプラン)
   - [Orchestratorを使用した統合フロー](#63-orchestratorを使用した統合フロー)
8. [エクスポート](#7-エクスポート)
9. [変更履歴](#8-変更履歴)
10. [付録: 依存関係図](#付録-依存関係図)
11. [関連ドキュメント](#関連ドキュメント)
12. [補足情報](#補足情報)

---

## 概要

`replan.py`は、GRACEシステムにおける動的リプランニング機能を提供するモジュールです。ステップ実行の失敗、低信頼度、ユーザーフィードバックなどのトリガーに応じて、計画を動的に修正・再生成します。

### 主な責務

- リプランのトリガー条件判定（ステップ失敗、低信頼度、ユーザーフィードバック等）
- リプラン戦略の決定（部分再計画、全体再計画、代替アクション、スキップ、中断）
- 失敗情報やフィードバックを考慮した新計画の生成
- リプラン履歴の管理
- Executor との統合によるリプランフロー制御

### 主要機能一覧

| 機能 | 説明 |
|------|------|
| `ReplanTrigger` | リプランのトリガー条件を定義するEnum |
| `ReplanStrategy` | リプラン戦略を定義するEnum |
| `ReplanContext` | リプラン時のコンテキスト情報を保持 |
| `ReplanResult` | リプラン結果を保持 |
| `ReplanManager` | 動的リプランニング管理の主クラス |
| `ReplanManager.should_replan()` | ステップ結果に基づくリプラン要否判定 |
| `ReplanManager.should_replan_from_feedback()` | フィードバックに基づくリプラン要否判定 |
| `ReplanManager.determine_strategy()` | リプラン戦略の決定 |
| `ReplanManager.create_new_plan()` | 新しい計画の生成 |
| `ReplanOrchestrator` | Executorとの統合リプランフロー管理 |
| `ReplanOrchestrator.handle_step_failure()` | ステップ失敗時のリプラン処理 |
| `ReplanOrchestrator.handle_user_feedback()` | ユーザーフィードバックによるリプラン処理 |
| `create_replan_manager()` | ReplanManagerのファクトリ関数 |
| `create_replan_orchestrator()` | ReplanOrchestratorのファクトリ関数 |

---

## 1. アーキテクチャ構成図

### 1.1 システム全体構成

```
┌─────────────────────────────────────────────────────────────────┐
│                        Executor 層                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  StepExecutor    │  │   UserInterface  │  │  Monitoring  │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘  │
└───────────┼─────────────────────┼───────────────────┼──────────┘
            │                     │                   │
            │ StepResult          │ Feedback          │
            └──────────────────┬──┴───────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        replan.py                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  ReplanOrchestrator                                        │ │
│  │    └── ReplanManager                                       │ │
│  │          ├── should_replan()                               │ │
│  │          ├── determine_strategy()                          │ │
│  │          └── create_new_plan()                             │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       依存モジュール層                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │    Planner     │  │    schemas     │  │    config      │    │
│  │  (計画生成)    │  │  (データ構造)  │  │   (設定管理)   │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 データフロー

1. Executor がステップ実行結果（StepResult）を取得
2. ReplanOrchestrator が結果を受け取り、リプラン要否を判定
3. リプランが必要な場合、ReplanManager が戦略を決定
4. 戦略に応じて Planner を使用して新計画を生成
5. 新計画（ExecutionPlan）を Executor に返却

---

## 2. モジュール構成図

### 2.1 内部モジュール構成

```
replan.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Enum]
  ├── ReplanTrigger           - リプランのトリガー条件
  └── ReplanStrategy          - リプラン戦略

[データクラス]
  ├── ReplanContext           - リプラン時のコンテキスト
  └── ReplanResult            - リプラン結果

[クラス]
  ├── ReplanManager           - 動的リプランニング管理
  │     ├── __init__()
  │     ├── _get_planner()
  │     ├── should_replan()
  │     ├── should_replan_from_feedback()
  │     ├── determine_strategy()
  │     ├── create_new_plan()
  │     ├── _create_full_replan()
  │     ├── _create_partial_replan()
  │     ├── _apply_fallback()
  │     ├── _skip_failed_step()
  │     ├── _enhance_query_with_context()
  │     ├── _create_remaining_query()
  │     ├── _adjust_step_ids()
  │     ├── _find_step()
  │     ├── can_replan()
  │     ├── get_history()
  │     └── clear_history()
  │
  └── ReplanOrchestrator      - リプランオーケストレーター
        ├── __init__()
        ├── handle_step_failure()
        └── handle_user_feedback()

[ファクトリ関数]
  ├── create_replan_manager()
  └── create_replan_orchestrator()
```

### 2.2 外部依存関係

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| `dataclasses` | 標準 | データクラス定義 |
| `typing` | 標準 | 型ヒント |
| `enum` | 標準 | 列挙型定義 |
| `datetime` | 標準 | 日時処理 |
| `logging` | 標準 | ログ出力 |

### 2.3 内部依存モジュール

| モジュール | インポート | 用途 |
|-----------|-----------|------|
| `.schemas` | `ExecutionPlan`, `PlanStep`, `StepResult` | 計画・結果のデータ構造 |
| `.planner` | `Planner`, `create_planner` | 計画生成 |
| `.config` | `get_config`, `GraceConfig` | 設定管理 |

**Planner クラスの使用方法**:

ReplanManager は以下の Planner メソッドを使用して新計画を生成します：

| メソッド | 用途 | 呼び出し元 |
|---------|------|-----------|
| `Planner.create_plan(query)` | クエリから新規計画を生成 | `_create_full_replan()`, `_create_partial_replan()` |
| `Planner.refine_plan(plan, feedback)` | フィードバックに基づく計画修正 | （将来の拡張用） |

**GraceConfigから使用するサブ設定**:

| サブ設定 | 説明 |
|---------|------|
| `config.replan.max_replans` | 最大リプラン回数（デフォルト: 3） |
| `config.replan.confidence_threshold` | リプラン発動の信頼度閾値（デフォルト: 0.4） |
| `config.replan.partial_replan_threshold` | 部分リプランの閾値（デフォルト: 0.6） |
| `config.replan.cooldown_seconds` | リプラン間隔（デフォルト: 5秒）※将来の拡張用 |

---

## 3. クラス・関数一覧表

### 3.1 Enum一覧

#### ReplanTrigger

| 値 | 説明 |
|------|------|
| `STEP_FAILED` | ステップ実行失敗 |
| `LOW_CONFIDENCE` | 信頼度が閾値未満 |
| `USER_FEEDBACK` | ユーザーからの修正要求 |
| `NEW_INFORMATION` | 新しい情報の発見 |
| `TIMEOUT` | タイムアウト |

#### ReplanStrategy

| 値 | 説明 |
|------|------|
| `PARTIAL` | 失敗ステップ以降のみ再計画 |
| `FULL` | 全体を再計画 |
| `FALLBACK` | 代替アクションへ切り替え |
| `SKIP` | 失敗ステップをスキップ |
| `ABORT` | 実行中断 |

### 3.2 データクラス一覧

#### ReplanContext

| フィールド/プロパティ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `trigger` | ReplanTrigger | - | トリガー条件 |
| `original_query` | str | - | 元のクエリ |
| `failed_step_id` | Optional[int] | None | 失敗したステップID |
| `error_message` | Optional[str] | None | エラーメッセージ |
| `completed_results` | Dict[int, StepResult] | {} | 完了済み結果 |
| `user_feedback` | Optional[str] | None | ユーザーフィードバック |
| `new_information` | Optional[str] | None | 新情報 |
| `replan_count` | int | 0 | リプラン回数 |
| `created_at` | datetime | now() | 作成日時 |
| `has_completed_steps` (property) | bool | - | 完了済みステップの有無 |
| `completed_step_ids` (property) | List[int] | - | 完了済みステップIDリスト |

#### ReplanResult

| フィールド | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `success` | bool | - | 成功フラグ |
| `strategy` | ReplanStrategy | - | 採用された戦略 |
| `new_plan` | Optional[ExecutionPlan] | None | 新しい計画 |
| `reason` | str | "" | 理由 |
| `replan_count` | int | 0 | リプラン回数 |
| `created_at` | datetime | now() | 作成日時 |

### 3.3 クラス一覧

#### ReplanManager

| メソッド | 概要 |
|---------|------|
| `__init__(config, planner)` | コンストラクタ |
| `should_replan(step_result, replan_count)` | ステップ結果からリプラン要否判定 |
| `should_replan_from_feedback(feedback, replan_count)` | フィードバックからリプラン要否判定 |
| `determine_strategy(context, current_plan)` | リプラン戦略を決定 |
| `create_new_plan(context, strategy, current_plan)` | 新しい計画を生成 |
| `can_replan(replan_count)` | リプラン可能か判定 |
| `get_history()` | リプラン履歴を取得 |
| `clear_history()` | 履歴をクリア |

#### ReplanOrchestrator

| メソッド | 概要 |
|---------|------|
| `__init__(config, replan_manager)` | コンストラクタ |
| `handle_step_failure(step_result, current_plan, completed_results, replan_count)` | ステップ失敗時のリプラン処理 |
| `handle_user_feedback(feedback, current_plan, completed_results, replan_count)` | フィードバックによるリプラン処理 |

### 3.4 ファクトリ関数一覧

| 関数名 | 概要 |
|-------|------|
| `create_replan_manager(config, planner)` | ReplanManagerインスタンス作成 |
| `create_replan_orchestrator(config, replan_manager)` | ReplanOrchestratorインスタンス作成 |

---

## 4. クラス・関数 IPO詳細

### 4.1 ReplanTrigger Enum

**概要**: リプランのトリガー条件を定義するEnum。

```python
class ReplanTrigger(str, Enum):
    STEP_FAILED = "step_failed"
    LOW_CONFIDENCE = "low_confidence"
    USER_FEEDBACK = "user_feedback"
    NEW_INFORMATION = "new_information"
    TIMEOUT = "timeout"
```

| 値 | 説明 | 発生条件 |
|------|------|---------|
| `STEP_FAILED` | ステップ実行失敗 | `step_result.status == "failed"` |
| `LOW_CONFIDENCE` | 低信頼度 | `step_result.confidence < threshold` |
| `USER_FEEDBACK` | ユーザーフィードバック | フィードバックに修正キーワード含む |
| `NEW_INFORMATION` | 新情報発見 | 追加情報により計画変更が必要 |
| `TIMEOUT` | タイムアウト | ステップ実行がタイムアウト |

---

### 4.2 ReplanStrategy Enum

**概要**: リプラン戦略を定義するEnum。

```python
class ReplanStrategy(str, Enum):
    PARTIAL = "partial"
    FULL = "full"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
```

| 値 | 説明 | 適用条件 |
|------|------|---------|
| `PARTIAL` | 部分再計画 | 中盤以降で失敗、フィードバック時 |
| `FULL` | 全体再計画 | 序盤（進捗≤34%）で失敗、タイムアウト |
| `FALLBACK` | 代替アクション | 失敗ステップに`fallback`定義あり |
| `SKIP` | ステップスキップ | 失敗ステップが必須でない場合 |
| `ABORT` | 中断 | 最大リプラン回数超過 |

---

### 4.3 ReplanContext データクラス

**概要**: リプラン時のコンテキスト情報を保持するデータクラス。

```python
@dataclass
class ReplanContext:
    trigger: ReplanTrigger
    original_query: str
    failed_step_id: Optional[int] = None
    error_message: Optional[str] = None
    completed_results: Dict[int, StepResult] = field(default_factory=dict)
    user_feedback: Optional[str] = None
    new_information: Optional[str] = None
    replan_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
```

**プロパティ**:

| プロパティ | 戻り値型 | 説明 |
|-----------|---------|------|
| `has_completed_steps` | bool | 完了済みステップがあるか |
| `completed_step_ids` | List[int] | 完了済みステップIDのソート済みリスト |

**メソッド**:

| メソッド | 戻り値型 | 説明 |
|---------|---------|------|
| `get_completed_outputs()` | Dict[int, str] | 完了済みステップの出力を取得 |

**戻り値例**:
```python
ReplanContext(
    trigger=ReplanTrigger.STEP_FAILED,
    original_query="東京の天気を教えて",
    failed_step_id=2,
    error_message="検索結果が見つかりませんでした",
    completed_results={1: StepResult(...)},
    replan_count=1
)
```

---

### 4.4 ReplanResult データクラス

**概要**: リプラン結果を保持するデータクラス。

```python
@dataclass
class ReplanResult:
    success: bool
    strategy: ReplanStrategy
    new_plan: Optional[ExecutionPlan] = None
    reason: str = ""
    replan_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
```

**戻り値例**:
```python
# 成功時
ReplanResult(
    success=True,
    strategy=ReplanStrategy.PARTIAL,
    new_plan=ExecutionPlan(...),
    reason="部分再計画",
    replan_count=2
)

# 失敗時（中断）
ReplanResult(
    success=False,
    strategy=ReplanStrategy.ABORT,
    new_plan=None,
    reason="最大リプラン回数超過により中断",
    replan_count=3
)
```

---

### 4.5 ReplanManager クラス

動的リプランニング管理の主クラス。

#### コンストラクタ: `__init__`

**概要**: ReplanManagerを初期化し、設定からリプラン制限を読み込みます。

```python
def __init__(
    self,
    config: Optional[GraceConfig] = None,
    planner: Optional[Planner] = None,
)
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |
| `planner` | Optional[Planner] | None | 計画生成用Planner（遅延初期化） |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]`, `planner: Optional[Planner]` |
| **Process** | 1. 設定を取得<br>2. max_replans, confidence_threshold等を設定<br>3. 履歴リストを初期化 |
| **Output** | ReplanManagerインスタンス |

```python
# 使用例
from grace.replan import ReplanManager

manager = ReplanManager()
# または
manager = ReplanManager(config=config, planner=planner)
```

---

#### メソッド: `should_replan`

**概要**: ステップ実行結果に基づいてリプランが必要か判定します。

```python
def should_replan(
    self,
    step_result: StepResult,
    replan_count: int
) -> tuple[bool, Optional[ReplanTrigger]]
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `step_result` | StepResult | - | ステップ実行結果 |
| `replan_count` | int | - | 現在のリプラン回数 |

| 項目 | 内容 |
|------|------|
| **Input** | `step_result: StepResult`, `replan_count: int` |
| **Process** | 1. 最大リプラン回数チェック<br>2. ステップ失敗チェック<br>3. 低信頼度チェック |
| **Output** | `tuple[bool, Optional[ReplanTrigger]]`: (リプラン要否, トリガー) |

**判定ロジック**:

| 条件 | 結果 |
|------|------|
| `replan_count >= max_replans` | (False, None) |
| `step_result.status == "failed"` | (True, STEP_FAILED) |
| `step_result.confidence < confidence_threshold` | (True, LOW_CONFIDENCE) |
| それ以外 | (False, None) |

**戻り値例**:
```python
(True, ReplanTrigger.STEP_FAILED)  # リプラン必要
(False, None)  # リプラン不要
```

```python
# 使用例
manager = ReplanManager()
should, trigger = manager.should_replan(step_result, replan_count=1)

if should:
    print(f"リプラン必要: {trigger.value}")
```

---

#### メソッド: `should_replan_from_feedback`

**概要**: ユーザーフィードバックに基づいてリプラン要否を判定します。

```python
def should_replan_from_feedback(
    self,
    feedback: str,
    replan_count: int
) -> tuple[bool, Optional[ReplanTrigger]]
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `feedback` | str | - | ユーザーフィードバック |
| `replan_count` | int | - | 現在のリプラン回数 |

| 項目 | 内容 |
|------|------|
| **Input** | `feedback: str`, `replan_count: int` |
| **Process** | 1. 最大リプラン回数チェック<br>2. 修正キーワードの有無をチェック |
| **Output** | `tuple[bool, Optional[ReplanTrigger]]`: (リプラン要否, トリガー) |

**修正キーワード**:
- "修正", "変更", "やり直し", "違う", "別の"

**戻り値例**:
```python
(True, ReplanTrigger.USER_FEEDBACK)  # 修正キーワード含む
(False, None)  # 含まない
```

```python
# 使用例
should, trigger = manager.should_replan_from_feedback(
    feedback="別のアプローチで試してください",
    replan_count=1
)
# -> (True, ReplanTrigger.USER_FEEDBACK)
```

---

#### メソッド: `determine_strategy`

**概要**: リプランコンテキストと現在の計画からリプラン戦略を決定します。

```python
def determine_strategy(
    self,
    context: ReplanContext,
    current_plan: ExecutionPlan
) -> ReplanStrategy
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `context` | ReplanContext | - | リプランコンテキスト |
| `current_plan` | ExecutionPlan | - | 現在の計画 |

| 項目 | 内容 |
|------|------|
| **Input** | `context: ReplanContext`, `current_plan: ExecutionPlan` |
| **Process** | トリガー・進捗・代替手段の有無に基づいて戦略を決定 |
| **Output** | `ReplanStrategy`: 選択された戦略 |

**戦略決定ロジック**:

| 条件 | 戦略 |
|------|------|
| `replan_count >= max_replans` | ABORT |
| STEP_FAILED + fallbackあり | FALLBACK |
| TIMEOUT | FULL |
| USER_FEEDBACK + "最初から"含む | FULL |
| USER_FEEDBACK | PARTIAL |
| 序盤の失敗（進捗≤34%） | FULL |
| それ以外 | PARTIAL |

**戻り値例**:
```python
ReplanStrategy.PARTIAL  # 部分再計画
ReplanStrategy.FULL     # 全体再計画
ReplanStrategy.FALLBACK # 代替アクション
```

```python
# 使用例
context = ReplanContext(
    trigger=ReplanTrigger.STEP_FAILED,
    original_query="質問",
    failed_step_id=2,
    replan_count=1
)
strategy = manager.determine_strategy(context, current_plan)
print(f"戦略: {strategy.value}")
```

---

#### メソッド: `create_new_plan`

**概要**: 戦略に基づいて新しい計画を生成します。

```python
def create_new_plan(
    self,
    context: ReplanContext,
    strategy: ReplanStrategy,
    current_plan: ExecutionPlan
) -> ReplanResult
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `context` | ReplanContext | - | リプランコンテキスト |
| `strategy` | ReplanStrategy | - | リプラン戦略 |
| `current_plan` | ExecutionPlan | - | 現在の計画 |

| 項目 | 内容 |
|------|------|
| **Input** | `context: ReplanContext`, `strategy: ReplanStrategy`, `current_plan: ExecutionPlan` |
| **Process** | 戦略に応じた計画生成処理を実行 |
| **Output** | `ReplanResult`: リプラン結果 |

**戦略別処理**:

| 戦略 | 処理 |
|------|------|
| FULL | エラー情報を含めて全体を再生成 |
| PARTIAL | 完了済みステップを保持し、残りを再生成 |
| FALLBACK | 失敗ステップを代替アクションに置換 |
| SKIP | 失敗ステップを除外し、依存関係を更新 |
| ABORT | 失敗結果を返却 |

**戻り値例**:
```python
ReplanResult(
    success=True,
    strategy=ReplanStrategy.PARTIAL,
    new_plan=ExecutionPlan(
        original_query="...",
        steps=[...],
        requires_confirmation=True
    ),
    reason="部分再計画",
    replan_count=2
)
```

```python
# 使用例
result = manager.create_new_plan(context, strategy, current_plan)

if result.success:
    print(f"新計画: {len(result.new_plan.steps)}ステップ")
else:
    print(f"リプラン失敗: {result.reason}")
```

---

#### メソッド: `can_replan`

**概要**: リプラン可能か判定します。

```python
def can_replan(self, replan_count: int) -> bool
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `replan_count` | int | - | 現在のリプラン回数 |

| 項目 | 内容 |
|------|------|
| **Input** | `replan_count: int` |
| **Process** | `replan_count < max_replans` を判定 |
| **Output** | `bool`: リプラン可能か |

```python
# 使用例
if manager.can_replan(replan_count=2):
    # リプラン実行
    pass
```

---

#### メソッド: `get_history`

**概要**: リプラン履歴を取得します。

```python
def get_history(self) -> List[ReplanResult]
```

| 項目 | 内容 |
|------|------|
| **Input** | なし |
| **Process** | 履歴リストのコピーを返却 |
| **Output** | `List[ReplanResult]`: リプラン履歴 |

---

#### メソッド: `clear_history`

**概要**: リプラン履歴をクリアします。

```python
def clear_history(self)
```

| 項目 | 内容 |
|------|------|
| **Input** | なし |
| **Process** | 履歴リストをクリア |
| **Output** | なし |

---

### 4.6 ReplanOrchestrator クラス

Executor と ReplanManager を統合し、リプランフローを管理するオーケストレーター。

#### コンストラクタ: `__init__`

**概要**: ReplanOrchestratorを初期化します。

```python
def __init__(
    self,
    config: Optional[GraceConfig] = None,
    replan_manager: Optional[ReplanManager] = None,
)
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |
| `replan_manager` | Optional[ReplanManager] | None | リプランマネージャー |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]`, `replan_manager: Optional[ReplanManager]` |
| **Process** | 設定を取得し、ReplanManagerを初期化 |
| **Output** | ReplanOrchestratorインスタンス |

```python
# 使用例
orchestrator = ReplanOrchestrator()
```

---

#### メソッド: `handle_step_failure`

**概要**: ステップ失敗時のリプラン処理を統合的に実行します。

```python
def handle_step_failure(
    self,
    step_result: StepResult,
    current_plan: ExecutionPlan,
    completed_results: Dict[int, StepResult],
    replan_count: int
) -> Optional[ReplanResult]
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `step_result` | StepResult | - | 失敗したステップの結果 |
| `current_plan` | ExecutionPlan | - | 現在の計画 |
| `completed_results` | Dict[int, StepResult] | - | 完了済み結果 |
| `replan_count` | int | - | 現在のリプラン回数 |

| 項目 | 内容 |
|------|------|
| **Input** | `step_result`, `current_plan`, `completed_results`, `replan_count` |
| **Process** | 1. リプラン要否判定<br>2. コンテキスト作成<br>3. 戦略決定<br>4. リプラン実行 |
| **Output** | `Optional[ReplanResult]`: リプラン結果（リプランしない場合はNone） |

**戻り値例**:
```python
# リプラン実行時
ReplanResult(
    success=True,
    strategy=ReplanStrategy.PARTIAL,
    new_plan=ExecutionPlan(...),
    reason="部分再計画",
    replan_count=2
)

# リプランしない場合
None
```

```python
# 使用例
orchestrator = ReplanOrchestrator()

result = orchestrator.handle_step_failure(
    step_result=failed_step_result,
    current_plan=current_plan,
    completed_results={1: result1, 2: result2},
    replan_count=1
)

if result and result.success:
    # 新計画で再実行
    new_plan = result.new_plan
```

---

#### メソッド: `handle_user_feedback`

**概要**: ユーザーフィードバックによるリプラン処理を統合的に実行します。

```python
def handle_user_feedback(
    self,
    feedback: str,
    current_plan: ExecutionPlan,
    completed_results: Dict[int, StepResult],
    replan_count: int
) -> Optional[ReplanResult]
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `feedback` | str | - | ユーザーフィードバック |
| `current_plan` | ExecutionPlan | - | 現在の計画 |
| `completed_results` | Dict[int, StepResult] | - | 完了済み結果 |
| `replan_count` | int | - | 現在のリプラン回数 |

| 項目 | 内容 |
|------|------|
| **Input** | `feedback`, `current_plan`, `completed_results`, `replan_count` |
| **Process** | 1. フィードバックからリプラン要否判定<br>2. コンテキスト作成<br>3. 戦略決定<br>4. リプラン実行 |
| **Output** | `Optional[ReplanResult]`: リプラン結果 |

```python
# 使用例
result = orchestrator.handle_user_feedback(
    feedback="違うアプローチで試してください",
    current_plan=current_plan,
    completed_results={1: result1},
    replan_count=0
)

if result:
    print(f"フィードバック反映: {result.strategy.value}")
```

---

### 4.7 ファクトリ関数

#### `create_replan_manager`

**概要**: ReplanManagerインスタンスを作成するファクトリ関数。

```python
def create_replan_manager(
    config: Optional[GraceConfig] = None,
    planner: Optional[Planner] = None,
) -> ReplanManager
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |
| `planner` | Optional[Planner] | None | Plannerインスタンス |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]`, `planner: Optional[Planner]` |
| **Process** | ReplanManagerをインスタンス化 |
| **Output** | `ReplanManager`: インスタンス |

```python
# 使用例
from grace.replan import create_replan_manager

manager = create_replan_manager()
```

---

#### `create_replan_orchestrator`

**概要**: ReplanOrchestratorインスタンスを作成するファクトリ関数。

```python
def create_replan_orchestrator(
    config: Optional[GraceConfig] = None,
    replan_manager: Optional[ReplanManager] = None,
) -> ReplanOrchestrator
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |
| `replan_manager` | Optional[ReplanManager] | None | ReplanManagerインスタンス |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]`, `replan_manager: Optional[ReplanManager]` |
| **Process** | ReplanOrchestratorをインスタンス化 |
| **Output** | `ReplanOrchestrator`: インスタンス |

```python
# 使用例
from grace.replan import create_replan_orchestrator

orchestrator = create_replan_orchestrator()
```

---

## 5. 設定・定数

### 5.1 ReplanConfig（リプラン設定）

`config.py` で定義されるリプラン関連の設定。

```python
class ReplanConfig(BaseModel):
    max_replans: int = 3
    confidence_threshold: float = 0.4
    partial_replan_threshold: float = 0.6
    cooldown_seconds: int = 5
```

| キー | デフォルト値 | 説明 |
|-----|-------------|------|
| `max_replans` | 3 | 最大リプラン回数 |
| `confidence_threshold` | 0.4 | リプラン発動の信頼度閾値 |
| `partial_replan_threshold` | 0.6 | 部分リプランの閾値 |
| `cooldown_seconds` | 5 | リプラン間の待機時間（秒） |

### 5.2 修正キーワード

フィードバックからリプラン要否を判定するキーワード。

```python
modification_keywords = ["修正", "変更", "やり直し", "違う", "別の"]
```

---

## 6. 使用例

### 6.1 基本的なワークフロー

```python
from grace.replan import (
    ReplanManager,
    ReplanContext,
    ReplanTrigger,
    create_replan_manager,
)
from grace.schemas import StepResult

# 1. ReplanManagerを作成
manager = create_replan_manager()

# 2. ステップ実行結果を作成（失敗例）
step_result = StepResult(
    step_id=2,
    status="failed",
    output=None,
    confidence=0.0,
    error="検索結果が見つかりませんでした"
)

# 3. リプラン要否を判定
should, trigger = manager.should_replan(step_result, replan_count=0)

if should:
    print(f"リプラン必要: {trigger.value}")

    # 4. コンテキストを作成
    context = ReplanContext(
        trigger=trigger,
        original_query="東京の天気を教えて",
        failed_step_id=step_result.step_id,
        error_message=step_result.error,
        completed_results={1: previous_result},
        replan_count=0
    )

    # 5. 戦略を決定
    strategy = manager.determine_strategy(context, current_plan)
    print(f"戦略: {strategy.value}")

    # 6. 新計画を生成
    result = manager.create_new_plan(context, strategy, current_plan)

    if result.success:
        print(f"リプラン成功: {len(result.new_plan.steps)}ステップ")
    else:
        print(f"リプラン失敗: {result.reason}")
```

### 6.2 ユーザーフィードバックによるリプラン

```python
from grace.replan import create_replan_manager

manager = create_replan_manager()

# ユーザーフィードバックを受信
feedback = "違うアプローチで、Web検索も使って調べてください"

# リプラン要否を判定
should, trigger = manager.should_replan_from_feedback(feedback, replan_count=0)

if should:
    context = ReplanContext(
        trigger=trigger,
        original_query="東京の観光スポットを教えて",
        user_feedback=feedback,
        completed_results={1: result1, 2: result2},
        replan_count=0
    )

    strategy = manager.determine_strategy(context, current_plan)
    # フィードバックによる場合は通常 PARTIAL が選択される

    result = manager.create_new_plan(context, strategy, current_plan)
    print(f"フィードバック反映完了: {result.strategy.value}")
```

### 6.3 Orchestratorを使用した統合フロー

```python
from grace.replan import create_replan_orchestrator

# 1. Orchestratorを作成
orchestrator = create_replan_orchestrator()

# 2. ステップ実行ループ
replan_count = 0
current_plan = initial_plan
completed_results = {}

for step in current_plan.steps:
    # ステップ実行
    result = execute_step(step)

    if result.status == "success":
        completed_results[step.step_id] = result
    else:
        # 失敗時のリプラン処理
        replan_result = orchestrator.handle_step_failure(
            step_result=result,
            current_plan=current_plan,
            completed_results=completed_results,
            replan_count=replan_count
        )

        if replan_result and replan_result.success:
            # 新計画で再開
            current_plan = replan_result.new_plan
            replan_count = replan_result.replan_count
            print(f"リプラン実行: {replan_result.strategy.value}")
            break  # 新計画で再ループ
        elif replan_result and not replan_result.success:
            # リプラン失敗（中断）
            print(f"実行中断: {replan_result.reason}")
            break

# ユーザーフィードバック処理
user_input = get_user_input()
if user_input:
    feedback_result = orchestrator.handle_user_feedback(
        feedback=user_input,
        current_plan=current_plan,
        completed_results=completed_results,
        replan_count=replan_count
    )

    if feedback_result and feedback_result.success:
        current_plan = feedback_result.new_plan
```

---

## 7. エクスポート

`__all__`でエクスポートされる要素：

```python
__all__ = [
    # Enums
    "ReplanTrigger",
    "ReplanStrategy",

    # Data classes
    "ReplanContext",
    "ReplanResult",

    # Managers
    "ReplanManager",
    "ReplanOrchestrator",

    # Factory functions
    "create_replan_manager",
    "create_replan_orchestrator",
]
```

---

## 8. 変更履歴

| バージョン | 変更内容 |
|-----------|---------|
| 1.0 | 初版作成（2025-01-29） |
| 1.1 | planner.pyの情報を反映：Plannerとの連携詳細、未使用機能の説明を追加 |
| 1.2 | Planner連携詳細を拡充：`_enhance_query_with_context()`、フォールバック動作、プロンプトルールを追加 |

---

## 付録: 依存関係図

```
replan.py
    │
    ├──► dataclasses
    │        └── dataclass
    │        └── field
    │
    ├──► typing
    │        └── Optional, List, Dict, Any
    │
    ├──► enum
    │        └── Enum
    │
    ├──► datetime
    │        └── datetime
    │
    ├──► logging
    │        └── getLogger
    │
    ├──► .schemas (内部)
    │        └── ExecutionPlan
    │        └── PlanStep
    │        └── StepResult
    │
    ├──► .planner (内部)
    │        └── Planner
    │        │     ├── __init__(config, model_name)
    │        │     ├── create_plan(query) → ExecutionPlan
    │        │     ├── refine_plan(plan, feedback) → ExecutionPlan
    │        │     ├── estimate_complexity(query) → float
    │        │     ├── estimate_complexity_with_llm(query) → float
    │        │     ├── _create_fallback_plan(query) → ExecutionPlan
    │        │     └── _get_available_collections() → list
    │        └── create_planner(config, model_name) → Planner
    │
    └──► .config (内部)
             └── get_config()
             └── GraceConfig
                   └── replan: ReplanConfig
                         ├── max_replans: int = 3
                         ├── confidence_threshold: float = 0.4
                         ├── partial_replan_threshold: float = 0.6
                         └── cooldown_seconds: int = 5
```

### Planner → Gemini API 連携

```
ReplanManager
    │
    └──► Planner.create_plan(enhanced_query)
              │
              ├── estimate_complexity_with_llm(query)
              │        └── Gemini API (temperature=0.1, max_tokens=10)
              │
              └── generate_content(PLAN_GENERATION_PROMPT)
                       └── Gemini API (response_schema=ExecutionPlan)
                                │
                                └── ExecutionPlan (JSON)
```

---

## 関連ドキュメント

| ドキュメント | 説明 |
|-------------|------|
| `config.md` | GraceConfig設定管理の詳細ドキュメント |
| `schemas.md` | ExecutionPlan, PlanStep, StepResultの詳細 |
| `planner.md` | Plannerクラスの詳細ドキュメント（計画生成ロジック） |
| `executor.md` | 計画実行エージェントのドキュメント |
| `confidence.md` | 信頼度計算システムのドキュメント |

---

## 補足情報

### Planner との連携詳細

ReplanManager は内部で Planner を使用して新計画を生成します。以下に連携の詳細を示します。

**Planner クラスの概要**:

| メソッド | 用途 | ReplanManagerでの使用 |
|---------|------|----------------------|
| `create_plan(query)` | クエリから新規計画を生成 | `_create_full_replan()`, `_create_partial_replan()` |
| `refine_plan(plan, feedback)` | フィードバックに基づく計画修正 | （将来の拡張用） |
| `estimate_complexity_with_llm(query)` | LLMで複雑度を推定 | 間接的に使用（create_plan内部） |

**`_create_full_replan()` での使用**:
```python
def _create_full_replan(self, context: ReplanContext) -> ExecutionPlan:
    # エラー情報を含めたクエリを生成
    enhanced_query = self._enhance_query_with_context(
        context.original_query,
        context
    )

    # Planner.create_plan() を呼び出して新計画を生成
    planner = self._get_planner()
    new_plan = planner.create_plan(enhanced_query)

    # リプラン後は確認を推奨
    new_plan.requires_confirmation = True
    return new_plan
```

**`_create_partial_replan()` での使用**:
```python
def _create_partial_replan(self, context, current_plan) -> ExecutionPlan:
    # 完了済みステップを保持
    completed_steps = [step for step in current_plan.steps
                       if step.step_id < context.failed_step_id]

    # 残りステップ用のクエリを生成
    remaining_query = self._create_remaining_query(context, completed_steps)

    # Planner.create_plan() で残りを再計画
    planner = self._get_planner()
    new_partial = planner.create_plan(remaining_query)

    # ステップIDを調整して結合
    adjusted_steps = self._adjust_step_ids(new_partial.steps, ...)
    final_steps = completed_steps + adjusted_steps

    return ExecutionPlan(steps=final_steps, ...)
```

**`_enhance_query_with_context()` によるクエリ拡張**:

リプラン時には、エラー情報や進捗情報を含めた拡張クエリがPlannerに渡されます：

```python
def _enhance_query_with_context(self, original_query, context) -> str:
    hints = []

    if context.error_message:
        hints.append(f"注意: 前回の試行で「{context.error_message}」というエラーが発生")

    if context.completed_results:
        completed_info = [f"ステップ{sid}は完了済み"
                         for sid in sorted(context.completed_results.keys())]
        hints.append(f"進捗: {', '.join(completed_info)}")

    if context.user_feedback:
        hints.append(f"ユーザーフィードバック: {context.user_feedback}")

    if context.new_information:
        hints.append(f"追加情報: {context.new_information}")

    if hints:
        return f"{original_query}\n\n【追加情報】\n" + "\n".join(hints)

    return original_query
```

**Plannerのフォールバック動作**:

Plannerは計画生成に失敗した場合、自動的にフォールバック計画を生成します：

```python
# Planner._create_fallback_plan() の構造
ExecutionPlan(
    original_query=query,
    complexity=0.5,
    estimated_steps=2,
    requires_confirmation=False,
    steps=[
        PlanStep(step_id=1, action="rag_search", collection="wikipedia_ja", ...),
        PlanStep(step_id=2, action="reasoning", depends_on=[1], ...)
    ],
    success_criteria="ユーザーの質問に適切に回答できている"
)
```

### 未使用・将来の拡張機能

以下の機能は定義されていますが、現在のコードでは未使用または将来の拡張用です：

| 項目 | 状態 | 説明 |
|------|------|------|
| `ReplanTrigger.NEW_INFORMATION` | 未使用 | 新情報発見時のトリガー（将来の拡張用） |
| `config.replan.cooldown_seconds` | 未使用 | リプラン間隔の制御（将来の拡張用） |
| `Planner.refine_plan()` | 未使用 | フィードバック反映（現在は `create_plan()` で代替） |
| `Planner._create_plan_legacy()` | バックアップ | Legacy Agent委譲版（run_legacy_agentアクション使用） |

### Planner が使用するプロンプト

**PLAN_GENERATION_PROMPT の主要ルール**:

1. **検索アクション統合**: `rag_search` は可能な限り1ステップにまとめる
2. **クエリ保持**: ユーザーの元の質問文を完全一致でコピー
3. **コレクション省略**: `collection` 引数は原則 `null`（システムが自動選択）
4. **最終ステップ**: 必ず `reasoning` で回答を生成

**複雑度の目安**:

| 範囲 | 説明 | ステップ数 |
|------|------|-----------|
| 0.0-0.3 | 単純な質問 | 1-2 |
| 0.4-0.6 | 中程度の質問 | 2-3 |
| 0.7-1.0 | 複雑な質問 | 4以上 |
