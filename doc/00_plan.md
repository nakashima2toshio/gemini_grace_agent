# GRACE Agent プロジェクト計画書

> **GRACE** = **G**uided **R**easoning with **A**daptive **C**onfidence **E**xecution
> 適応型計画実行エージェント

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [アーキテクチャ概要](#2-アーキテクチャ概要)
3. [技術的特徴の比較](#3-技術的特徴の比較)
4. [実装詳細設計](#4-実装詳細設計)
5. [実装ロードマップ](#5-実装ロードマップ)
6. [評価方法](#6-評価方法)

---

## 1. プロジェクト概要

### 1.1 背景

現在の `Gemini_Agent_RAG` プロジェクトは、**ReAct + Reflection** パターンを実装した学習用エージェントである。このプロジェクトを発展させ、より高度なエージェントアーキテクチャを学習・実装する。

### 1.2 命名

| 項目 | 名称 |
|------|------|
| **プロジェクト名** | `grace-agent` |
| **アプリ表示名** | GRACE |
| **サブタイトル** | Adaptive Research Agent |
| **Pythonパッケージ名** | `grace` |
| **正式名称** | Guided Reasoning with Adaptive Confidence Execution |
| **日本語名** | 適応型計画実行エージェント |

### 1.3 命名の由来

```
G - Guided      : 計画に導かれる + HITL（人間介入）
R - Reasoning   : ReActの思考部分を継承
A - Adaptive    : 動的リプランニング
C - Confidence  : 信頼度ベース判断
E - Execution   : 実行フェーズ
```

### 1.4 ディレクトリ構成

```
gemini_gracr_agent/
├── README.md
├── pyproject.toml          # name = "grace-agent"
├── grace/                   # メインパッケージ
│   ├── __init__.py
│   ├── app.py              # Streamlit エントリーポイント
│   ├── config.py           # 設定・閾値管理
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py      # 計画生成エージェント
│   │   ├── executor.py     # 計画実行エージェント
│   │   ├── reflector.py    # 自己評価エージェント
│   │   └── confidence.py   # 信頼度スコア計算
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag_search.py   # RAGツール
│   │   ├── web_search.py   # Web検索ツール
│   │   ├── ask_user.py     # HITLツール
│   │   └── tool_router.py  # ツール選択ロジック
│   ├── schemas/
│   │   ├── plan_schema.py  # 計画のJSONスキーマ
│   │   └── step_result.py  # ステップ結果のスキーマ
│   └── prompts/
│       ├── planner_system.txt
│       ├── executor_system.txt
│       └── reflector_system.txt
├── docs/
└── tests/
```

---

## 2. アーキテクチャ概要

### 2.1 GRACE はハイブリッドアーキテクチャ

GRACE は単一のパターンではなく、**複数のエージェントパターンを統合したハイブリッドアーキテクチャ**である。

```
┌─────────────────────────────────────────────────────────┐
│                    GRACE Agent                          │
│         (Hybrid Agentic Architecture)                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
│   │ Plan-and-   │ + │ ReAct       │ + │ Reflection  │  │
│   │ Execute     │   │ (継承)      │   │ (継承)      │  │
│   └─────────────┘   └─────────────┘   └─────────────┘  │
│          │                                              │
│          ▼                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
│   │ Confidence  │ + │ HITL        │ + │ Adaptive    │  │
│   │ -aware      │   │             │   │ Replanning  │  │
│   └─────────────┘   └─────────────┘   └─────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 学術的な分類

```
GRACE = Plan-and-Execute
        + ReAct (within execution)
        + Reflection
        + Confidence-based HITL
        + Adaptive Replanning
```

| 分類軸 | GRACE の位置づけ |
|--------|-----------------|
| 制御フロー | Plan-and-Execute ベース |
| 推論方式 | ReAct（実行時） |
| 品質管理 | Reflection + Confidence |
| 協調方式 | Human-in-the-Loop |
| 適応方式 | Adaptive Replanning |

### 2.3 典型的な Plan-and-Execute との違い

```
┌─────────────────────────────────────────────────────────┐
│  典型的な Plan-and-Execute                              │
├─────────────────────────────────────────────────────────┤
│  Plan → Execute Step 1 → Step 2 → ... → Output         │
│  （計画は固定、失敗したら終わり）                         │
└─────────────────────────────────────────────────────────┘

                        vs

┌─────────────────────────────────────────────────────────┐
│  GRACE Agent                                            │
├─────────────────────────────────────────────────────────┤
│  Plan ──┬──→ Execute (ReAct) ──→ Evaluate (Confidence) │
│         │         │                    │                │
│         │         ▼                    ▼                │
│         │    [失敗/低信頼]        [要確認]              │
│         │         │                    │                │
│         ◀─────────┴── Replan ◀── HITL ◀┘               │
│                                                         │
│  （動的に計画修正、人間と協調、自己評価）                 │
└─────────────────────────────────────────────────────────┘
```

### 2.4 呼び方のバリエーション

| 文脈 | 呼び方 |
|------|--------|
| 一般向け | GRACE Agent（適応型リサーチエージェント） |
| 技術説明 | Hybrid Agentic Architecture with Plan-Execute-Reflect loop |
| 論文風 | Confidence-aware Plan-and-Execute with Human-in-the-Loop |
| 簡潔に | Adaptive Planning Agent |

---

## 3. 技術的特徴の比較

### 3.1 現行 vs 改修版

| # | 特徴 | 現行 (ReAct + Reflection) | GRACE (改修版) |
|---|------|--------------------------|----------------|
| 1 | 思考と行動 | ✅ ReAct | ✅ 継承 |
| 2 | 自己評価 | ✅ Reflection | ✅ 継承・強化 |
| 3 | 事前計画 | ❌ | ✅ **Plan-and-Execute** |
| 4 | 人間介入 | ❌ | ✅ **HITL** |
| 5 | 信頼度判断 | ❌ | ✅ **Confidence-aware** |
| 6 | 動的再計画 | ❌ | ✅ **Adaptive Replanning** |

### 3.2 各パターンの役割

| パターン | 特徴 | GRACE での役割 |
|----------|------|----------------|
| **ReAct** | Think → Act → Observe ループ | 実行フェーズ内で使用（継承） |
| **Reflection** | 自己評価・修正 | 継承・強化 |
| **Plan-and-Execute** | 先に計画、後で実行 | 骨格・制御フロー |
| **HITL** | 人間との協調 | 不確実時の介入 |
| **Confidence-aware** | 不確実性の定量化 | 判断の根拠 |
| **Adaptive Replanning** | 動的な計画修正 | 真の自律性 |

### 3.3 技術項目の依存関係

```
                    ┌─────────────────┐
                    │ (1) ReAct       │ ← 既存（継承）
                    │ (2) Reflection  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ (3) Plan-and-   │ ← 最初に実装
                    │     Execute     │   （他の基盤）
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │(4) Confidence│ │(3b) HITL     │ │(5) Adaptive  │
     │   -aware     │ │              │ │  Replanning  │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            └────────────────┴────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   統合・最適化    │
                    └─────────────────┘
```

### 3.4 実装優先度マトリクス

| 項目 | 重要度 | 難易度 | 学習効果 | 依存関係 | 推奨順序 |
|------|--------|--------|----------|----------|----------|
| Plan-and-Execute | ⭐⭐⭐⭐⭐ | 中 | ⭐⭐⭐⭐⭐ | なし | **1st** |
| Confidence-aware | ⭐⭐⭐⭐ | 中〜高 | ⭐⭐⭐⭐ | (3)必須 | **2nd** |
| HITL | ⭐⭐⭐⭐ | 中 | ⭐⭐⭐⭐⭐ | (3)(4)推奨 | **3rd** |
| Adaptive Replanning | ⭐⭐⭐⭐⭐ | 高 | ⭐⭐⭐⭐⭐ | (3)(4)必須 | **4th** |

---

## 4. 実装詳細設計

### 4.1 計画スキーマ（JSON Schema）

```python
# schemas/plan_schema.py

from typing import Literal
from pydantic import BaseModel, Field

class PlanStep(BaseModel):
    """計画の1ステップを表現"""
    step_id: int = Field(..., description="ステップ番号（1から開始）")
    action: Literal["rag_search", "web_search", "reasoning", "ask_user"] = Field(
        ..., description="実行するアクション種別"
    )
    description: str = Field(..., description="このステップで何をするか")
    query: str | None = Field(None, description="検索クエリ（検索系アクションの場合）")
    depends_on: list[int] = Field(default_factory=list, description="依存する先行ステップのID")
    expected_output: str = Field(..., description="期待される出力の説明")
    fallback: str | None = Field(None, description="失敗時の代替アクション")

class ExecutionPlan(BaseModel):
    """実行計画全体"""
    original_query: str = Field(..., description="ユーザーの元の質問")
    complexity: float = Field(..., ge=0.0, le=1.0, description="推定複雑度")
    estimated_steps: int = Field(..., description="推定ステップ数")
    requires_confirmation: bool = Field(..., description="実行前に確認が必要か")
    steps: list[PlanStep] = Field(..., description="実行ステップのリスト")
    success_criteria: str = Field(..., description="計画成功の判定基準")
```

### 4.2 Confidence Score 計算

```python
# agents/confidence.py

from dataclasses import dataclass

@dataclass
class ConfidenceFactors:
    """信頼度を構成する各要素"""
    search_overlap: float = 0.0      # 情報源の一致度 (0-1)
    logical_consistency: float = 0.0  # 推論の整合性 (0-1)
    tool_success: float = 0.0         # ツール実行成功度 (0-1)
    source_quality: float = 0.0       # ソースの信頼性 (0-1)
    query_coverage: float = 0.0       # クエリへの回答網羅度 (0-1)

class ConfidenceCalculator:
    """信頼度スコアの計算"""

    # 初期重み（経験則ベース、後で調整可能）
    DEFAULT_WEIGHTS = {
        "search_overlap": 0.25,      # 複数ソースの一致は重要
        "logical_consistency": 0.30,  # 推論の整合性は最重要
        "tool_success": 0.15,         # ツール成功は前提条件
        "source_quality": 0.15,       # ソース品質も考慮
        "query_coverage": 0.15,       # 回答の網羅性
    }

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._validate_weights()

    def _validate_weights(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def calculate(self, factors: ConfidenceFactors) -> float:
        """重み付き平均で信頼度を計算"""
        score = (
            factors.search_overlap * self.weights["search_overlap"] +
            factors.logical_consistency * self.weights["logical_consistency"] +
            factors.tool_success * self.weights["tool_success"] +
            factors.source_quality * self.weights["source_quality"] +
            factors.query_coverage * self.weights["query_coverage"]
        )
        return round(score, 3)
```

### 4.3 介入レベル判定（動的閾値付き）

```python
# config.py

from dataclasses import dataclass, field
from typing import Literal

InterventionLevel = Literal["silent", "notify", "confirm", "escalate"]

@dataclass
class InterventionConfig:
    """介入レベルの閾値設定"""

    # 基本閾値
    silent_threshold: float = 0.9
    notify_threshold: float = 0.7
    confirm_threshold: float = 0.4

    # 動的調整パラメータ
    adjustment_history: list[tuple[float, bool]] = field(default_factory=list)
    learning_rate: float = 0.05

    def get_level(self, confidence: float) -> InterventionLevel:
        """信頼度から介入レベルを判定"""
        if confidence > self.silent_threshold:
            return "silent"
        elif confidence > self.notify_threshold:
            return "notify"
        elif confidence > self.confirm_threshold:
            return "confirm"
        else:
            return "escalate"

    def record_feedback(self, confidence: float, was_correct: bool):
        """ユーザーフィードバックを記録し、閾値を調整"""
        self.adjustment_history.append((confidence, was_correct))

        if len(self.adjustment_history) >= 10:
            recent = self.adjustment_history[-10:]
            false_positives = sum(1 for c, correct in recent if c > 0.7 and not correct)
            false_negatives = sum(1 for c, correct in recent if c < 0.5 and correct)

            if false_positives > 3:
                self._raise_thresholds()
            elif false_negatives > 3:
                self._lower_thresholds()
```

**介入レベルの定義:**

| レベル | 判定基準 | UI上の挙動 |
|--------|----------|-----------|
| Silent | Confidence > 0.9 | バックグラウンドで進行し、最終結果だけ表示 |
| Notify | 0.7 < Confidence ≤ 0.9 | 「～を検索中...」などのステータスを逐次表示 |
| Confirm | 0.4 < Confidence ≤ 0.7 | 「この方針で実行して良いですか？」とボタンを表示 |
| Escalate | Confidence ≤ 0.4 | 「情報が不足しています。〇〇について教えてください」と入力を促す |

### 4.4 動的リプランニングのトリガー条件

```python
# agents/executor.py

from enum import Enum
from dataclasses import dataclass

class StepOutcome(Enum):
    """ステップ実行結果の分類"""
    SUCCESS = "success"           # 成功、次へ進む
    PARTIAL = "partial"           # 部分的成功、補完が必要
    RETRY_SAME = "retry_same"     # 同じステップを再試行
    RETRY_DIFFERENT = "retry_diff" # 別のアプローチで再試行
    INVALIDATE = "invalidate"     # 計画全体を破棄して再計画
    ESCALATE = "escalate"         # 人間に判断を委ねる

@dataclass
class StepEvaluation:
    """ステップ実行後の評価結果"""
    outcome: StepOutcome
    confidence: float
    reason: str
    suggested_action: str | None = None

class ReplanTrigger:
    """リプランニングの判定ロジック"""

    # INVALIDATEのトリガー条件
    INVALIDATE_CONDITIONS = [
        "consecutive_failures >= 2",      # 連続2回失敗
        "total_failures >= 3",            # 累計3回失敗
        "critical_step_failed",           # 必須ステップの失敗
        "assumption_violated",            # 前提条件の崩壊
        "timeout_exceeded",               # 時間切れ
    ]

    def __init__(self):
        self.failure_count = 0
        self.consecutive_failures = 0
        self.step_results: list[StepEvaluation] = []

    def should_invalidate_plan(self) -> tuple[bool, str]:
        """計画全体を破棄すべきか判定"""

        if self.consecutive_failures >= 2:
            return True, "連続2回の失敗"

        if self.failure_count >= 3:
            return True, "累計3回の失敗"

        if len(self.step_results) >= 2:
            recent_confidence = [r.confidence for r in self.step_results[-2:]]
            if all(c < 0.4 for c in recent_confidence):
                return True, "直近ステップの信頼度が継続的に低い"

        return False, ""
```

### 4.5 HITL ツール定義（Function Calling）

```python
# tools/ask_user.py

from typing import Literal

# Gemini Function Calling 用の定義
ASK_USER_TOOL = {
    "name": "ask_user_for_clarification",
    "description": """
    ユーザーに追加情報を求めるツール。
    以下の場合にのみ使用:
    - 質問の意図が曖昧で、複数の解釈が可能
    - 必要な情報が検索で見つからない
    - 矛盾する情報があり、どちらを優先すべきか不明

    使用は最後の手段。まず他のツールで解決を試みること。
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "ユーザーへの質問文（明確かつ簡潔に）"
            },
            "reason": {
                "type": "string",
                "description": "なぜこの質問が必要か（ユーザーに表示）"
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "選択肢がある場合のリスト（任意）"
            },
            "urgency": {
                "type": "string",
                "enum": ["blocking", "optional"],
                "description": "blocking: 回答がないと進めない, optional: 推測で進めることも可能"
            }
        },
        "required": ["question", "reason", "urgency"]
    }
}
```

### 4.6 Planner システムプロンプト

```text
# prompts/planner_system.txt

あなたは「計画立案エージェント」です。
ユーザーの質問を分析し、回答を得るための実行計画を JSON 形式で出力します。

## あなたの責務

1. **質問の分解**: 複雑な質問を、実行可能な小さなステップに分解する
2. **複雑度の推定**: タスクの難易度を 0.0〜1.0 で評価する
3. **ツール選択**: 各ステップで使用すべきツールを決定する
4. **依存関係の特定**: ステップ間の実行順序を明確にする

## 利用可能なツール

- `rag_search`: 内部知識ベースの検索（過去の文書、マニュアル等）
- `web_search`: インターネット検索（最新情報、ニュース等）
- `reasoning`: 収集した情報からの推論・分析
- `ask_user`: ユーザーへの追加質問（情報不足時のみ）

## 複雑度の判定基準

| 複雑度 | 条件 |
|--------|------|
| 0.0-0.3 | 単一ソースで即答可能、事実確認のみ |
| 0.3-0.6 | 複数ソース必要、軽い分析が必要 |
| 0.6-0.8 | 複数視点の統合、比較分析が必要 |
| 0.8-1.0 | 矛盾する情報の調停、高度な推論が必要 |

## 重要なルール

1. **ステップ数は2〜5に収める**: 細かすぎると管理困難、粗すぎると Context Drift
2. **最初のステップは必ず情報収集**: いきなり reasoning から始めない
3. **ask_user は最後の手段**: まず他のツールで解決を試みる
4. **fallback を設定**: 重要なステップには代替案を用意
5. **requires_confirmation**: complexity > 0.6 なら true
```

---

## 5. 実装ロードマップ

### 5.1 全体タイムライン

```
Month 1                          Month 2                          Month 3
├─────────────────────────────────┼─────────────────────────────────┤
│ Phase 1: Plan-and-Execute       │ Phase 3: HITL                   │
│ [===========]                   │ [===========]                   │
│             Phase 2: Confidence │             Phase 4: Replanning │
│             [===========]       │             [===============]   │
│                                 │                                 │
▼                                 ▼                                 ▼
MVP: 計画実行できる               v0.5: 人間と協調できる            v1.0: 自律的に適応
```

### 5.2 Phase 1: Plan-and-Execute（2-3週間）

> **基盤構築 - 他のすべての前提条件**

#### Week 1-2: 計画生成

- Planner プロンプト設計
- JSON Schema 定義（Pydantic）
- 複雑度推定ロジック
- 単体テスト

#### Week 2-3: 計画実行

- Executor 基本実装
- ステップ順次実行
- 結果の収集・保持
- 統合テスト

#### 学習ポイント

| トピック | 習得スキル |
|----------|-----------|
| Structured Output | LLMにJSON出力させる技術 |
| Task Decomposition | 複雑な問題を分解する設計思考 |
| State Management | 実行状態の管理パターン |

#### 成果物イメージ

```python
planner = Planner()
plan = await planner.generate(query)
# => ExecutionPlan(steps=[...], complexity=0.6)

executor = Executor()
results = await executor.run(plan)
```

#### 評価基準

- [ ] 10種類の質問で計画生成できる
- [ ] 計画通りに順次実行できる
- [ ] 各ステップの結果を次のステップで使える

---

### 5.3 Phase 2: Confidence-aware（2週間）

> **判断の質を可視化 - HITL/Replanの判断根拠**

#### Week 4: スコア算出

- ConfidenceFactors 定義
- 各要素の計算ロジック
  - search_overlap（情報源一致度）
  - logical_consistency（推論整合性）
  - tool_success（ツール成功率）
- 重み付け計算
- 単体テスト

#### Week 5: 閾値と判定

- InterventionLevel 定義
- 閾値の初期設定
- レベル判定ロジック
- UI表示（デバッグ用）

#### 学習ポイント

| トピック | 習得スキル |
|----------|-----------|
| Uncertainty Quantification | 不確実性の定量化 |
| LLM Self-Evaluation | LLMに自己評価させる技術 |
| Threshold Tuning | 閾値調整の考え方 |

#### 成果物イメージ

```python
calculator = ConfidenceCalculator()
factors = ConfidenceFactors(
    search_overlap=0.8,
    logical_consistency=0.7,
    tool_success=1.0
)
score = calculator.calculate(factors)  # => 0.78
level = config.get_level(score)  # => "notify"
```

#### 評価基準

- [ ] 各ステップでConfidenceスコアを算出できる
- [ ] スコアに基づいて4段階のレベル判定ができる
- [ ] UI上でスコアを確認できる（デバッグモード）

---

### 5.4 Phase 3: HITL - Human-in-the-Loop（2週間）

> **賢い「待機」の実装 - ユーザー体験の向上**

#### Week 6: 基本HITL

- ask_user ツール定義（Function Calling）
- Streamlit UI コンポーネント
  - Silent（表示なし）
  - Notify（ステータス表示）
  - Confirm（確認ダイアログ）
  - Escalate（入力要求）
- 応答待機ロジック
- 統合テスト

#### Week 7: 計画確認フロー

- 計画表示UI
- 修正入力の受付
- 修正の計画への反映
- E2Eテスト

#### 学習ポイント

| トピック | 習得スキル |
|----------|-----------|
| Function Calling | LLMにツールを使わせる設計 |
| Async UI Pattern | 非同期での人間介入パターン |
| UX Design | 適切な介入タイミングの設計 |

#### 成果物イメージ

```python
# エージェントが自発的に質問
if confidence < 0.4:
    response = await ask_user(
        question="〇〇について詳しく教えてください",
        reason="情報が不足しています",
        urgency="blocking"
    )
```

#### 評価基準

- [ ] 4つの介入レベルがUIで動作する
- [ ] エージェントが自発的に質問できる
- [ ] ユーザーの回答を計画に反映できる

---

### 5.5 Phase 4: Adaptive Replanning（2-3週間）

> **最も複雑 - 真の自律性の実現**

#### Week 8: 失敗検知

- StepOutcome 定義（6種類）
- 失敗条件の判定ロジック
- 連続失敗カウント
- 単体テスト

#### Week 9: リプラン実行

- Invalidate トリガー条件（5種類）
- 部分リプラン（1ステップ修正）
- 全体リプラン（計画再生成）
- リプラン回数制限
- 統合テスト

#### Week 10: 最適化

- リプラン戦略の選択ロジック
- 履歴を活かした再計画
- 無限ループ防止
- E2Eテスト

#### 学習ポイント

| トピック | 習得スキル |
|----------|-----------|
| Error Recovery | 障害復旧パターン |
| State Machine | 状態遷移の設計 |
| Recursion Control | 再帰・ループ制御 |

#### 成果物イメージ

```python
class Executor:
    async def run(self, plan):
        for step in plan.steps:
            result = await self.execute_step(step)
            evaluation = self.evaluate(result)

            match evaluation.outcome:
                case StepOutcome.SUCCESS:
                    continue
                case StepOutcome.RETRY_SAME:
                    result = await self.retry(step)
                case StepOutcome.INVALIDATE:
                    plan = await self.replan(plan, results)
                case StepOutcome.ESCALATE:
                    await self.ask_user(...)
```

#### 評価基準

- [ ] 失敗を検知して自動リトライできる
- [ ] 連続失敗時に計画を再生成できる
- [ ] 無限ループに陥らない
- [ ] リプラン履歴を保持・表示できる

---

## 6. 評価方法

### 6.1 テストシナリオ

| Phase | テスト内容 | 成功基準 |
|-------|-----------|----------|
| 1 | 「Pythonの非同期処理を説明して」 | 2-3ステップの計画を生成・実行 |
| 2 | RAGで矛盾する情報を返す | Confidence低下を検知 |
| 3 | 「最新の〇〇」（曖昧な質問） | 自発的に clarification を要求 |
| 4 | 存在しない情報を検索させる | 失敗→リプラン→代替手段で回答 |

### 6.2 定量評価指標

```python
@dataclass
class GraceMetrics:
    # Phase 1
    plan_generation_success_rate: float  # 計画生成成功率
    plan_execution_completion_rate: float  # 計画完遂率

    # Phase 2
    confidence_accuracy: float  # 信頼度と実際の正確性の相関
    false_positive_rate: float  # 高信頼度での誤り率

    # Phase 3
    unnecessary_hitl_rate: float  # 不要な人間介入率
    user_satisfaction: float  # ユーザー満足度（手動評価）

    # Phase 4
    replan_success_rate: float  # リプラン後の成功率
    avg_replan_count: float  # 平均リプラン回数
```

### 6.3 週次ロードマップ

| Week | タスク | 成果物 | 検証方法 |
|------|--------|--------|----------|
| 1 | Planner実装 | `planner.py`, プロンプト | 10パターンの質問で計画生成テスト |
| 2 | Executor実装 | `executor.py` | 計画の順次実行確認 |
| 3 | Confidence計算 | `confidence.py` | 手動評価との比較 |
| 4 | 閾値・レベル判定 | `config.py` | 4レベルの判定確認 |
| 5 | HITL基本実装 | `ask_user.py`, UI | 介入UIの動作確認 |
| 6 | 計画確認フロー | UI統合 | E2Eテスト |
| 7 | 失敗検知 | `StepOutcome` | 意図的失敗でテスト |
| 8 | リプラン実行 | リプランロジック | 自動リプラン確認 |
| 9 | 最適化・統合 | End-to-End | 複雑な質問5件で全フロー確認 |

---

## 付録

### A. 最初の1週間でやること

#### Day 1-2: 環境準備

```bash
# リポジトリ作成
mkdir grace-agent && cd grace-agent
git init

# 基本構造
mkdir -p grace/{agents,tools,schemas,prompts}
touch grace/__init__.py
touch pyproject.toml
```

#### Day 3-4: Planner プロンプト作成

1. `prompts/planner_system.txt` を作成
2. Gemini API で動作確認
3. JSON出力の安定化

#### Day 5-7: 最小限の Plan-and-Execute

```python
# 目標: この動作を確認
query = "Pythonのasyncについて教えて"
plan = await planner.generate(query)
print(plan.steps)  # [Step1: RAG検索, Step2: 整理]

for step in plan.steps:
    result = await executor.execute(step)
    print(f"Step {step.step_id}: {result}")
```

### B. 重要度ランキング（まとめ）

| 順位 | 項目 | 理由 |
|------|------|------|
| 1 | Plan-and-Execute | すべての基盤、これなしに他は不可能 |
| 2 | Confidence-aware | HITL/Replanの判断根拠、品質の可視化 |
| 3 | HITL | ユーザー体験向上、実用性の鍵 |
| 4 | Adaptive Replanning | 最も複雑だが、真の自律性を実現 |

---

**Document Version:** 1.0
**Created:** 2025-01
**Status:** Planning Phase

