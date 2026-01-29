# confidence.py - 信頼度計算システム ドキュメント

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
   - [Enum一覧](#32-enum一覧)
   - [クラス一覧](#33-クラス一覧)
   - [ファクトリ関数一覧](#34-ファクトリ関数一覧)
5. [クラス・関数 IPO詳細](#4-クラス関数-ipo詳細)
   - [ConfidenceFactors データクラス](#41-confidencefactors-データクラス)
   - [ConfidenceScore データクラス](#42-confidencescore-データクラス)
   - [InterventionLevel Enum](#43-interventionlevel-enum)
   - [ActionDecision データクラス](#44-actiondecision-データクラス)
   - [ConfidenceCalculator クラス](#45-confidencecalculator-クラス)
   - [LLMSelfEvaluator クラス](#46-llmselfevaluator-クラス)
   - [SourceAgreementCalculator クラス](#47-sourceagreementcalculator-クラス)
   - [QueryCoverageCalculator クラス](#48-querycoveragecalculator-クラス)
   - [ConfidenceAggregator クラス](#49-confidenceaggregator-クラス)
   - [ファクトリ関数](#410-ファクトリ関数)
6. [設定・定数](#5-設定定数)
7. [使用例](#6-使用例)
   - [基本的なワークフロー](#61-基本的なワークフロー)
   - [LLM評価を使用したワークフロー](#62-llm評価を使用したワークフロー)
   - [複数ステップの集計](#63-複数ステップの集計)
8. [エクスポート](#7-エクスポート)
9. [変更履歴](#8-変更履歴)
10. [付録: 依存関係図](#付録-依存関係図)
11. [関連ドキュメント](#関連ドキュメント)

---

## 概要

`confidence.py`は、GRACEシステムにおける多軸信頼度計算を担当するモジュールです。ハイブリッド方式（重み付き平均 + LLM自己評価）により、RAG検索結果、ツール実行結果、複数ソースの一致度などを総合的に評価し、信頼度スコアを算出します。

### 主な責務

- RAG検索品質に基づく信頼度計算
- LLMによる自己評価の実行
- 複数情報源間の一致度算出（Embedding類似度）
- クエリに対する回答網羅度の評価
- 信頼度に基づく介入レベルの決定
- 複数ステップの信頼度集計

### 主要機能一覧

| 機能 | 説明 |
|------|------|
| `ConfidenceFactors` | 信頼度計算に使用する各要素を保持するデータクラス |
| `ConfidenceScore` | 計算された信頼度スコアと内訳を保持 |
| `InterventionLevel` | 介入レベルを定義するEnum |
| `ActionDecision` | 信頼度に基づくアクション決定を保持 |
| `ConfidenceCalculator` | ハイブリッド方式による信頼度計算の主クラス |
| `ConfidenceCalculator.calculate()` | 重み付き平均による信頼度計算 |
| `ConfidenceCalculator.llm_calculate()` | LLMを使用した信頼度計算 |
| `ConfidenceCalculator.decide_action()` | 信頼度に基づく介入レベル決定 |
| `LLMSelfEvaluator` | LLMによる自己評価クラス |
| `LLMSelfEvaluator.evaluate()` | 質問・回答ペアの自己評価 |
| `LLMSelfEvaluator.evaluate_with_factors()` | Factorsを考慮した総合評価 |
| `SourceAgreementCalculator` | 複数ソース間の一致度計算 |
| `QueryCoverageCalculator` | クエリ網羅度計算 |
| `ConfidenceAggregator` | 複数ステップの信頼度集計 |
| `create_confidence_calculator()` | ConfidenceCalculatorのファクトリ関数 |
| `create_llm_evaluator()` | LLMSelfEvaluatorのファクトリ関数 |
| `create_source_agreement_calculator()` | SourceAgreementCalculatorのファクトリ関数 |
| `create_query_coverage_calculator()` | QueryCoverageCalculatorのファクトリ関数 |
| `create_confidence_aggregator()` | ConfidenceAggregatorのファクトリ関数 |

---

## 1. アーキテクチャ構成図

### 1.1 システム全体構成

```
┌─────────────────────────────────────────────────────────────────┐
│                        GRACE エージェント層                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │     Executor     │  │     Planner      │  │   Reasoner   │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘  │
└───────────┼─────────────────────┼───────────────────┼──────────┘
            │                     │                   │
            └──────────────────┬──┴───────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      confidence.py                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  ConfidenceCalculator  │  LLMSelfEvaluator                 │ │
│  │  SourceAgreementCalc   │  QueryCoverageCalc                │ │
│  │  ConfidenceAggregator  │  ActionDecision                   │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       外部サービス層                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │  Gemini API    │  │  Embedding API │  │    config.py   │    │
│  │  (生成/評価)   │  │  (類似度計算)  │  │   (設定管理)   │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 データフロー

1. エージェント層（Executor/Planner/Reasoner）からステップ実行結果を受信
2. ConfidenceFactorsに検索結果・ツール実行結果などの要素を格納
3. ConfidenceCalculatorが重み付き平均またはLLM評価で信頼度を計算
4. 必要に応じてSourceAgreementCalculator/QueryCoverageCalculatorで補完評価
5. decide_action()で介入レベル（SILENT/NOTIFY/CONFIRM/ESCALATE）を決定
6. ActionDecisionをエージェント層に返却

---

## 2. モジュール構成図

### 2.1 内部モジュール構成

```
confidence.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[データクラス]
  ├── ConfidenceFactors       - 信頼度を構成する各要素
  ├── ConfidenceScore         - 信頼度スコアと内訳
  └── ActionDecision          - アクション決定結果

[Enum]
  └── InterventionLevel       - 介入レベル定義

[クラス]
  ├── ConfidenceCalculator    - ハイブリッド信頼度計算
  │     ├── __init__()
  │     ├── _validate_weights()
  │     ├── calculate()
  │     ├── llm_calculate()
  │     ├── _calc_search_quality()
  │     ├── _calc_tool_success()
  │     ├── _apply_penalties()
  │     └── decide_action()
  │
  ├── LLMSelfEvaluator        - LLM自己評価
  │     ├── __init__()
  │     ├── evaluate()
  │     └── evaluate_with_factors()
  │
  ├── SourceAgreementCalculator - ソース一致度計算
  │     ├── __init__()
  │     ├── calculate()
  │     └── _cosine_similarity()
  │
  ├── QueryCoverageCalculator - クエリ網羅度計算
  │     ├── __init__()
  │     └── calculate()
  │
  └── ConfidenceAggregator    - 複数ステップ集計
        ├── __init__()
        ├── aggregate()
        └── aggregate_with_critical_check()

[ファクトリ関数]
  ├── create_confidence_calculator()
  ├── create_llm_evaluator()
  ├── create_source_agreement_calculator()
  ├── create_query_coverage_calculator()
  └── create_confidence_aggregator()
```

### 2.2 外部依存関係

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| `google-genai` | - | Gemini API クライアント（生成・Embedding） |
| `dataclasses` | 標準 | データクラス定義 |
| `typing` | 標準 | 型ヒント |
| `enum` | 標準 | 列挙型定義 |
| `logging` | 標準 | ログ出力 |

### 2.3 内部依存モジュール

| モジュール | インポート | 用途 |
|-----------|-----------|------|
| `.config` | `get_config` | 設定取得関数（シングルトン） |
| `.config` | `GraceConfig` | GRACE統合設定モデル |

**GraceConfigから使用するサブ設定**:

| サブ設定 | 説明 |
|---------|------|
| `config.confidence.weights` | ConfidenceWeights - 信頼度計算の重み |
| `config.confidence.thresholds` | ConfidenceThresholds - 介入レベルの閾値 |
| `config.llm.model` | LLMモデル名（デフォルト: gemini-2.5-flash） |
| `config.embedding.model` | Embeddingモデル名（デフォルト: gemini-embedding-001） |

---

## 3. クラス・関数一覧表

### 3.1 データクラス一覧

#### ConfidenceFactors

| フィールド | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `search_result_count` | int | 0 | 検索結果数 |
| `search_avg_score` | float | 0.0 | 平均類似度スコア |
| `search_max_score` | float | 0.0 | 最高類似度スコア |
| `search_score_variance` | float | 1.0 | スコアの分散 |
| `source_agreement` | float | 0.0 | 情報源間の一致度 (0-1) |
| `source_count` | int | 0 | 引用ソース数 |
| `llm_self_confidence` | float | 0.5 | LLMの自己評価 (0-1) |
| `tool_success_rate` | float | 1.0 | ツール成功率 |
| `tool_execution_count` | int | 0 | 実行ツール数 |
| `tool_success_count` | int | 0 | 成功ツール数 |
| `query_coverage` | float | 0.0 | クエリへの回答網羅度 |
| `is_search_step` | bool | False | 検索ステップかどうか |

#### ConfidenceScore

| フィールド/プロパティ | 型 | 説明 |
|-----------|------|------|
| `score` | float | 最終スコア (0.0-1.0) |
| `factors` | ConfidenceFactors | 計算に使用した要素 |
| `breakdown` | Dict[str, float] | 各要素のスコア内訳 |
| `penalties_applied` | List[str] | 適用されたペナルティ |
| `reason` | str | 信頼度スコアの理由 |
| `level` (property) | str | 信頼度レベル (high/medium/low/very_low) |

#### ActionDecision

| フィールド/プロパティ | 型 | 説明 |
|-----------|------|------|
| `level` | InterventionLevel | 介入レベル |
| `confidence_score` | float | 信頼度スコア |
| `reason` | str | 理由 |
| `suggested_action` | Optional[str] | 推奨アクション |
| `should_proceed` (property) | bool | 自動進行可能か |
| `needs_confirmation` (property) | bool | 確認が必要か |
| `needs_user_input` (property) | bool | ユーザー入力が必要か |

### 3.2 Enum一覧

#### InterventionLevel

| 値 | 説明 |
|------|------|
| `SILENT` | バックグラウンドで進行 |
| `NOTIFY` | ステータス表示 |
| `CONFIRM` | 確認を求める |
| `ESCALATE` | ユーザー入力を要求 |

### 3.3 クラス一覧

#### ConfidenceCalculator

| メソッド | 概要 |
|---------|------|
| `__init__(config)` | コンストラクタ（設定指定） |
| `calculate(factors)` | 重み付き平均による信頼度計算 |
| `llm_calculate(factors, step_description, tool_output)` | LLMを使用した信頼度計算 |
| `decide_action(score)` | 信頼度に基づくアクション決定 |

#### LLMSelfEvaluator

| メソッド | 概要 |
|---------|------|
| `__init__(config, model_name)` | コンストラクタ |
| `evaluate(query, answer, sources)` | 質問・回答ペアの自己評価 |
| `evaluate_with_factors(description, output, factors)` | Factorsを考慮した総合評価 |

#### SourceAgreementCalculator

| メソッド | 概要 |
|---------|------|
| `__init__(config)` | コンストラクタ |
| `calculate(answers)` | 複数回答間の一致度計算 |

#### QueryCoverageCalculator

| メソッド | 概要 |
|---------|------|
| `__init__(config, model_name)` | コンストラクタ |
| `calculate(query, answer)` | クエリ網羅度計算 |

#### ConfidenceAggregator

| メソッド | 概要 |
|---------|------|
| `__init__(config)` | コンストラクタ |
| `aggregate(scores, method)` | 複数スコアの集計 |
| `aggregate_with_critical_check(scores, critical_threshold)` | 重要度チェック付き集計 |

### 3.4 ファクトリ関数一覧

| 関数名 | 概要 |
|-------|------|
| `create_confidence_calculator(config)` | ConfidenceCalculatorインスタンス作成 |
| `create_llm_evaluator(config, model_name)` | LLMSelfEvaluatorインスタンス作成 |
| `create_source_agreement_calculator(config)` | SourceAgreementCalculatorインスタンス作成 |
| `create_query_coverage_calculator(config, model_name)` | QueryCoverageCalculatorインスタンス作成 |
| `create_confidence_aggregator(config)` | ConfidenceAggregatorインスタンス作成 |

---

## 4. クラス・関数 IPO詳細

### 4.1 ConfidenceFactors データクラス

**概要**: 信頼度計算に使用する各要素を保持するデータクラス。RAG検索結果、ツール実行結果、LLM自己評価などの要素を格納します。

```python
@dataclass
class ConfidenceFactors:
    search_result_count: int = 0
    search_avg_score: float = 0.0
    search_max_score: float = 0.0
    search_score_variance: float = 1.0
    source_agreement: float = 0.0
    source_count: int = 0
    llm_self_confidence: float = 0.5
    tool_success_rate: float = 1.0
    tool_execution_count: int = 0
    tool_success_count: int = 0
    query_coverage: float = 0.0
    is_search_step: bool = False
```

**戻り値例**:
```python
ConfidenceFactors(
    search_result_count=5,
    search_avg_score=0.75,
    search_max_score=0.92,
    search_score_variance=0.05,
    source_agreement=0.85,
    source_count=3,
    llm_self_confidence=0.8,
    tool_success_rate=1.0,
    tool_execution_count=1,
    tool_success_count=1,
    query_coverage=0.9,
    is_search_step=True
)
```

```python
# 使用例
factors = ConfidenceFactors(
    search_result_count=5,
    search_max_score=0.92,
    is_search_step=True
)
print(f"検索結果: {factors.search_result_count}件")
# 出力: 検索結果: 5件
```

---

### 4.2 ConfidenceScore データクラス

**概要**: 計算された信頼度スコアと内訳を保持するデータクラス。`level`プロパティで信頼度レベル（high/medium/low/very_low）を取得可能。

```python
@dataclass
class ConfidenceScore:
    score: float
    factors: ConfidenceFactors
    breakdown: Dict[str, float] = field(default_factory=dict)
    penalties_applied: List[str] = field(default_factory=list)
    reason: str = ""
```

| 項目 | 内容 |
|------|------|
| **Input** | `score: float`, `factors: ConfidenceFactors`, `breakdown: Dict`, `penalties_applied: List`, `reason: str` |
| **Process** | データを保持し、levelプロパティでスコアに応じたレベル文字列を返す |
| **Output** | ConfidenceScoreインスタンス |

**levelプロパティの閾値**:

| スコア範囲 | レベル |
|-----------|--------|
| 0.9 以上 | high |
| 0.7 以上 | medium |
| 0.4 以上 | low |
| 0.4 未満 | very_low |

**戻り値例**:
```python
ConfidenceScore(
    score=0.85,
    factors=ConfidenceFactors(...),
    breakdown={
        "search_quality": 0.92,
        "source_agreement": 0.85,
        "llm_self_eval": 0.8,
        "tool_success": 1.0,
        "query_coverage": 0.9
    },
    penalties_applied=[],
    reason=""
)
# score.level -> "medium"
```

---

### 4.3 InterventionLevel Enum

**概要**: 信頼度に基づく介入レベルを定義するEnum。

```python
class InterventionLevel(str, Enum):
    SILENT = "silent"
    NOTIFY = "notify"
    CONFIRM = "confirm"
    ESCALATE = "escalate"
```

| 値 | 説明 | 動作 |
|------|------|------|
| `SILENT` | silent | バックグラウンドで自動進行 |
| `NOTIFY` | notify | ステータス表示しながら進行 |
| `CONFIRM` | confirm | ユーザーに確認を求める |
| `ESCALATE` | escalate | ユーザー入力を要求 |

---

### 4.4 ActionDecision データクラス

**概要**: 信頼度に基づくアクション決定を保持するデータクラス。プロパティで進行可否を判定可能。

```python
@dataclass
class ActionDecision:
    level: InterventionLevel
    confidence_score: float
    reason: str
    suggested_action: Optional[str] = None
```

| 項目 | 内容 |
|------|------|
| **Input** | `level: InterventionLevel`, `confidence_score: float`, `reason: str`, `suggested_action: Optional[str]` |
| **Process** | データ保持、プロパティで進行可否を判定 |
| **Output** | ActionDecisionインスタンス |

**プロパティ**:

| プロパティ | 条件 |
|-----------|------|
| `should_proceed` | level が SILENT または NOTIFY の場合 True |
| `needs_confirmation` | level が CONFIRM の場合 True |
| `needs_user_input` | level が ESCALATE の場合 True |

**戻り値例**:
```python
ActionDecision(
    level=InterventionLevel.NOTIFY,
    confidence_score=0.75,
    reason="中程度の信頼度: ステータス表示しながら進行",
    suggested_action="proceed_with_status"
)
# decision.should_proceed -> True
# decision.needs_confirmation -> False
```

---

### 4.5 ConfidenceCalculator クラス

ハイブリッド方式によるConfidence計算の主クラス。

#### コンストラクタ: `__init__`

**概要**: ConfidenceCalculatorを初期化し、重みの妥当性を検証します。

```python
def __init__(self, config: Optional[GraceConfig] = None)
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定（Noneの場合はデフォルト） |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig] = None` |
| **Process** | 1. 設定を取得（Noneならデフォルト）<br>2. 重みを設定から読み込み<br>3. 重みの合計が1.0か検証 |
| **Output** | ConfidenceCalculatorインスタンス |

```python
# 使用例
from grace.config import get_config
from grace.confidence import ConfidenceCalculator

calculator = ConfidenceCalculator()
# または
config = get_config()
calculator = ConfidenceCalculator(config=config)
```

---

#### メソッド: `calculate`

**概要**: 重み付き平均による信頼度計算。検索ステップと非検索ステップで計算ロジックが異なります。

```python
def calculate(self, factors: ConfidenceFactors) -> ConfidenceScore
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `factors` | ConfidenceFactors | - | 信頼度要素 |

| 項目 | 内容 |
|------|------|
| **Input** | `factors: ConfidenceFactors` |
| **Process** | 1. 各要素を0-1にスケーリング<br>2. 検索/非検索ステップで異なる重み計算<br>3. ペナルティ適用<br>4. 0.0-1.0の範囲に収める |
| **Output** | `ConfidenceScore`: 信頼度スコアと内訳 |

**検索ステップの場合**:
- 検索品質（search_quality）をベースにする
- tool_successは減点として扱う

**非検索ステップの場合**:
- 有効な要素だけで加重平均を計算
- 重み: 検索品質(0.6)、ツール成功(0.4)、ソース一致度(0.2)、LLM自己評価(0.3)、クエリ網羅度(0.1)

**戻り値例**:
```python
ConfidenceScore(
    score=0.85,
    factors=factors,
    breakdown={
        "search_quality": 0.92,
        "source_agreement": 0.85,
        "llm_self_eval": 0.0,
        "tool_success": 1.0,
        "query_coverage": 0.0
    },
    penalties_applied=[]
)
```

```python
# 使用例
calculator = ConfidenceCalculator()
factors = ConfidenceFactors(
    search_result_count=5,
    search_max_score=0.92,
    is_search_step=True
)
score = calculator.calculate(factors)
print(f"信頼度: {score.score}, レベル: {score.level}")
# 出力: 信頼度: 0.92, レベル: high
```

---

#### メソッド: `llm_calculate`

**概要**: LLMを使用した信頼度計算。統計的要因とLLM評価を組み合わせます。

```python
def llm_calculate(
    self,
    factors: ConfidenceFactors,
    step_description: str = "",
    tool_output: str = ""
) -> ConfidenceScore
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `factors` | ConfidenceFactors | - | 統計的要因（参考情報） |
| `step_description` | str | "" | ステップの目的 |
| `tool_output` | str | "" | ツールの出力 |

| 項目 | 内容 |
|------|------|
| **Input** | `factors: ConfidenceFactors`, `step_description: str`, `tool_output: str` |
| **Process** | 1. LLMSelfEvaluatorで評価実行<br>2. ガードレール: 検索スコアが高い場合は優先<br>3. 内訳を作成 |
| **Output** | `ConfidenceScore`: LLM評価に基づく信頼度 |

**戻り値例**:
```python
ConfidenceScore(
    score=0.8,
    factors=factors,
    breakdown={
        "llm_score": 0.8,
        "reason": 1.0
    },
    reason="主要な情報が取得でき、信頼できる回答が可能",
    penalties_applied=[]
)
```

```python
# 使用例
calculator = ConfidenceCalculator()
factors = ConfidenceFactors(
    search_result_count=3,
    search_max_score=0.85,
    is_search_step=True
)
score = calculator.llm_calculate(
    factors=factors,
    step_description="東京の天気を検索",
    tool_output="検索結果: 東京は晴れ、気温25度..."
)
print(f"LLM評価: {score.score}, 理由: {score.reason}")
```

---

#### メソッド: `decide_action`

**概要**: 信頼度スコアに基づいて介入レベルを決定します。

```python
def decide_action(self, score: ConfidenceScore) -> ActionDecision
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `score` | ConfidenceScore | - | 信頼度スコア |

| 項目 | 内容 |
|------|------|
| **Input** | `score: ConfidenceScore` |
| **Process** | 設定の閾値と比較して介入レベルを決定 |
| **Output** | `ActionDecision`: アクション決定 |

**閾値（設定から読み込み）**:

| 介入レベル | 条件 | 推奨アクション |
|-----------|------|---------------|
| SILENT | score >= thresholds.silent | proceed |
| NOTIFY | score >= thresholds.notify | proceed_with_status |
| CONFIRM | score >= thresholds.confirm | ask_confirmation |
| ESCALATE | score < thresholds.confirm | request_clarification |

**戻り値例**:
```python
ActionDecision(
    level=InterventionLevel.SILENT,
    confidence_score=0.92,
    reason="高い信頼度: 自動進行",
    suggested_action="proceed"
)
```

```python
# 使用例
calculator = ConfidenceCalculator()
score = calculator.calculate(factors)
action = calculator.decide_action(score)

if action.should_proceed:
    print("自動進行")
elif action.needs_confirmation:
    print("確認が必要です")
elif action.needs_user_input:
    print("追加情報が必要です")
```

---

### 4.6 LLMSelfEvaluator クラス

LLMによる自己評価を実行するクラス。

#### コンストラクタ: `__init__`

**概要**: LLMSelfEvaluatorを初期化し、Gemini Clientを準備します。

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
| `model_name` | Optional[str] | None | 使用するモデル名（Noneの場合は設定から取得） |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]`, `model_name: Optional[str]` |
| **Process** | 1. 設定を取得<br>2. モデル名を設定<br>3. Gemini Client初期化 |
| **Output** | LLMSelfEvaluatorインスタンス |

---

#### メソッド: `evaluate`

**概要**: 質問・回答ペアに対するLLMの自己評価を実行します。

```python
def evaluate(
    self,
    query: str,
    answer: str,
    sources: Optional[List[str]] = None
) -> float
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `query` | str | - | 元の質問 |
| `answer` | str | - | 生成された回答 |
| `sources` | Optional[List[str]] | None | 使用した情報源のリスト |

| 項目 | 内容 |
|------|------|
| **Input** | `query: str`, `answer: str`, `sources: Optional[List[str]]` |
| **Process** | 1. 評価プロンプトを作成<br>2. LLMに送信（temperature=0.0）<br>3. 数値を抽出して0.0-1.0に収める |
| **Output** | `float`: 信頼度 (0.0-1.0) |

**評価基準**:
- 正確性 (Accuracy): 情報源に基づいているか
- 適切性 (Relevance): 質問に直接答えているか
- スタイル (Style): 読みやすい日本語か

**戻り値例**:
```python
0.8  # ほぼ確実（信頼できる情報源あり）
```

```python
# 使用例
evaluator = LLMSelfEvaluator()
confidence = evaluator.evaluate(
    query="東京の人口は？",
    answer="東京都の人口は約1400万人です。",
    sources=["総務省統計局"]
)
print(f"自己評価: {confidence}")
# 出力: 自己評価: 0.85
```

---

#### メソッド: `evaluate_with_factors`

**概要**: ConfidenceFactorsとコンテキストを考慮した総合評価を実行します。

```python
def evaluate_with_factors(
    self,
    description: str,
    output: str,
    factors: ConfidenceFactors
) -> Dict[str, Any]
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `description` | str | - | ステップの目的 |
| `output` | str | - | ツールの出力内容 |
| `factors` | ConfidenceFactors | - | 統計的要因 |

| 項目 | 内容 |
|------|------|
| **Input** | `description: str`, `output: str`, `factors: ConfidenceFactors` |
| **Process** | 1. 統計データを含むプロンプト作成<br>2. LLMに送信（JSON出力）<br>3. スコアと理由を抽出 |
| **Output** | `Dict[str, Any]`: `{"score": float, "reason": str}` |

**評価項目**:
1. 検索品質: 根拠となる情報が十分にマッチしているか
2. ツール成功: エラーなく期待される情報を返しているか
3. ソース一致度: 複数の情報源が矛盾していないか
4. 目標達成度: ステップの目的を達成できているか

**戻り値例**:
```python
{
    "score": 0.8,
    "reason": "検索結果が質問に関連しており、信頼できる情報が得られました。"
}
```

```python
# 使用例
evaluator = LLMSelfEvaluator()
factors = ConfidenceFactors(
    search_result_count=5,
    search_max_score=0.92
)
result = evaluator.evaluate_with_factors(
    description="東京の天気を検索",
    output="検索結果: 東京は晴れ...",
    factors=factors
)
print(f"スコア: {result['score']}, 理由: {result['reason']}")
```

---

### 4.7 SourceAgreementCalculator クラス

複数ソース間の一致度を計算するクラス。

#### コンストラクタ: `__init__`

**概要**: SourceAgreementCalculatorを初期化し、Embedding用のクライアントを準備します。

```python
def __init__(self, config: Optional[GraceConfig] = None)
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]` |
| **Process** | 1. 設定を取得<br>2. Gemini Client初期化<br>3. Embeddingモデル名を設定 |
| **Output** | SourceAgreementCalculatorインスタンス |

---

#### メソッド: `calculate`

**概要**: 複数の回答間の一致度をEmbedding類似度で計算します。

```python
def calculate(self, answers: List[str]) -> float
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `answers` | List[str] | - | 回答のリスト |

| 項目 | 内容 |
|------|------|
| **Input** | `answers: List[str]` |
| **Process** | 1. 各回答のEmbeddingを取得<br>2. ペアワイズでコサイン類似度を計算<br>3. 平均一致度を返す |
| **Output** | `float`: 一致度 (0.0-1.0) |

> 📝 **注意**: 単一ソース（1件以下）の場合は1.0（完全一致）を返します。

**戻り値例**:
```python
0.85  # 3つの回答が高い一致度
```

```python
# 使用例
calculator = SourceAgreementCalculator()
answers = [
    "東京の人口は約1400万人です。",
    "東京都には約1400万人が住んでいます。",
    "東京の人口は1380万人程度です。"
]
agreement = calculator.calculate(answers)
print(f"一致度: {agreement}")
# 出力: 一致度: 0.92
```

---

### 4.8 QueryCoverageCalculator クラス

クエリに対する回答の網羅度を計算するクラス。

#### コンストラクタ: `__init__`

**概要**: QueryCoverageCalculatorを初期化します。

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
| **Process** | 1. 設定を取得<br>2. モデル名を設定<br>3. Gemini Client初期化 |
| **Output** | QueryCoverageCalculatorインスタンス |

---

#### メソッド: `calculate`

**概要**: クエリに対する回答の網羅度をLLMで評価します。

```python
def calculate(self, query: str, answer: str) -> float
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `query` | str | - | 元の質問 |
| `answer` | str | - | 生成された回答 |

| 項目 | 内容 |
|------|------|
| **Input** | `query: str`, `answer: str` |
| **Process** | 1. 網羅度評価プロンプトを作成<br>2. LLMに送信（temperature=0.0）<br>3. 数値を抽出して0.0-1.0に収める |
| **Output** | `float`: 網羅度 (0.0-1.0) |

**網羅度の目安**:

| スコア | 説明 |
|--------|------|
| 1.0 | すべての質問要素に完全に回答 |
| 0.8 | ほぼすべての要素に回答 |
| 0.6 | 主要な要素に回答 |
| 0.4 | 一部の要素のみに回答 |
| 0.2 | ほとんど回答できていない |
| 0.0 | 全く回答できていない |

**戻り値例**:
```python
0.8  # ほぼすべての要素に回答
```

```python
# 使用例
calculator = QueryCoverageCalculator()
coverage = calculator.calculate(
    query="東京の人口と面積を教えてください",
    answer="東京都の人口は約1400万人です。"
)
print(f"網羅度: {coverage}")
# 出力: 網羅度: 0.5（面積が回答されていない）
```

---

### 4.9 ConfidenceAggregator クラス

複数ステップの信頼度を集計するクラス。

#### コンストラクタ: `__init__`

**概要**: ConfidenceAggregatorを初期化します。

```python
def __init__(self, config: Optional[GraceConfig] = None)
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]` |
| **Process** | 設定を取得 |
| **Output** | ConfidenceAggregatorインスタンス |

---

#### メソッド: `aggregate`

**概要**: 複数の信頼度スコアを指定した方法で集計します。

```python
def aggregate(
    self,
    scores: List[ConfidenceScore],
    method: Literal["mean", "min", "weighted"] = "mean"
) -> float
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `scores` | List[ConfidenceScore] | - | 信頼度スコアのリスト |
| `method` | Literal["mean", "min", "weighted"] | "mean" | 集計方法 |

| 項目 | 内容 |
|------|------|
| **Input** | `scores: List[ConfidenceScore]`, `method: Literal["mean", "min", "weighted"]` |
| **Process** | 指定された方法でスコアを集計 |
| **Output** | `float`: 集計された信頼度 |

**集計方法**:

| method | 説明 |
|--------|------|
| `mean` | 平均値 |
| `min` | 最小値（最も弱い部分を重視） |
| `weighted` | 重み付き平均（後半のステップを重視） |

**戻り値例**:
```python
0.75  # 3ステップの平均
```

```python
# 使用例
aggregator = ConfidenceAggregator()
scores = [score1, score2, score3]  # 各ステップの信頼度

avg_score = aggregator.aggregate(scores, method="mean")
min_score = aggregator.aggregate(scores, method="min")
weighted_score = aggregator.aggregate(scores, method="weighted")

print(f"平均: {avg_score}, 最小: {min_score}, 重み付き: {weighted_score}")
```

---

#### メソッド: `aggregate_with_critical_check`

**概要**: 重要度チェック付きの集計。いずれかのステップが閾値を下回る場合、全体の信頼度を低下させます。

```python
def aggregate_with_critical_check(
    self,
    scores: List[ConfidenceScore],
    critical_threshold: float = 0.3
) -> tuple[float, bool]
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `scores` | List[ConfidenceScore] | - | 信頼度スコアのリスト |
| `critical_threshold` | float | 0.3 | 重要閾値 |

| 項目 | 内容 |
|------|------|
| **Input** | `scores: List[ConfidenceScore]`, `critical_threshold: float = 0.3` |
| **Process** | 1. 閾値未満のステップをチェック<br>2. 平均を計算<br>3. 重要ステップ失敗時は0.7倍のペナルティ |
| **Output** | `tuple[float, bool]`: (集計スコア, 重要ステップ失敗フラグ) |

**戻り値例**:
```python
(0.525, True)  # 重要ステップ失敗（0.75 * 0.7 = 0.525）
```

```python
# 使用例
aggregator = ConfidenceAggregator()
scores = [score1, score2, score3]

final_score, has_failure = aggregator.aggregate_with_critical_check(
    scores,
    critical_threshold=0.3
)

if has_failure:
    print(f"警告: 重要ステップに失敗があります（スコア: {final_score}）")
else:
    print(f"全ステップ正常（スコア: {final_score}）")
```

---

### 4.10 ファクトリ関数

#### `create_confidence_calculator`

**概要**: ConfidenceCalculatorインスタンスを作成するファクトリ関数。

```python
def create_confidence_calculator(
    config: Optional[GraceConfig] = None
) -> ConfidenceCalculator
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig] = None` |
| **Process** | ConfidenceCalculatorをインスタンス化 |
| **Output** | `ConfidenceCalculator`: インスタンス |

```python
# 使用例
from grace.confidence import create_confidence_calculator

calculator = create_confidence_calculator()
```

---

#### `create_llm_evaluator`

**概要**: LLMSelfEvaluatorインスタンスを作成するファクトリ関数。

```python
def create_llm_evaluator(
    config: Optional[GraceConfig] = None,
    model_name: Optional[str] = None
) -> LLMSelfEvaluator
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |
| `model_name` | Optional[str] | None | 使用するモデル名 |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]`, `model_name: Optional[str]` |
| **Process** | LLMSelfEvaluatorをインスタンス化 |
| **Output** | `LLMSelfEvaluator`: インスタンス |

```python
# 使用例
from grace.confidence import create_llm_evaluator

evaluator = create_llm_evaluator(model_name="gemini-2.0-flash")
```

---

#### `create_source_agreement_calculator`

**概要**: SourceAgreementCalculatorインスタンスを作成するファクトリ関数。

```python
def create_source_agreement_calculator(
    config: Optional[GraceConfig] = None
) -> SourceAgreementCalculator
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig] = None` |
| **Process** | SourceAgreementCalculatorをインスタンス化 |
| **Output** | `SourceAgreementCalculator`: インスタンス |

```python
# 使用例
from grace.confidence import create_source_agreement_calculator

calculator = create_source_agreement_calculator()
```

---

#### `create_query_coverage_calculator`

**概要**: QueryCoverageCalculatorインスタンスを作成するファクトリ関数。

```python
def create_query_coverage_calculator(
    config: Optional[GraceConfig] = None,
    model_name: Optional[str] = None
) -> QueryCoverageCalculator
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |
| `model_name` | Optional[str] | None | 使用するモデル名 |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig]`, `model_name: Optional[str]` |
| **Process** | QueryCoverageCalculatorをインスタンス化 |
| **Output** | `QueryCoverageCalculator`: インスタンス |

```python
# 使用例
from grace.confidence import create_query_coverage_calculator

calculator = create_query_coverage_calculator()
```

---

#### `create_confidence_aggregator`

**概要**: ConfidenceAggregatorインスタンスを作成するファクトリ関数。

```python
def create_confidence_aggregator(
    config: Optional[GraceConfig] = None
) -> ConfidenceAggregator
```

| パラメータ | 型 | デフォルト | 説明 |
|------------|------|-----------|------|
| `config` | Optional[GraceConfig] | None | GRACE設定 |

| 項目 | 内容 |
|------|------|
| **Input** | `config: Optional[GraceConfig] = None` |
| **Process** | ConfidenceAggregatorをインスタンス化 |
| **Output** | `ConfidenceAggregator`: インスタンス |

```python
# 使用例
from grace.confidence import create_confidence_aggregator

aggregator = create_confidence_aggregator()
```

---

## 5. 設定・定数

### 5.1 ConfidenceWeights（重み設定）

信頼度計算に使用する各要素の重み。合計は1.0である必要があります。

```python
class ConfidenceWeights(BaseModel):
    search_quality: float = 0.25
    source_agreement: float = 0.20
    llm_self_eval: float = 0.25
    tool_success: float = 0.15
    query_coverage: float = 0.15
```

| キー | デフォルト値 | 説明 |
|-----|-------------|------|
| `search_quality` | 0.25 | RAG検索品質の重み |
| `source_agreement` | 0.20 | ソース一致度の重み |
| `llm_self_eval` | 0.25 | LLM自己評価の重み |
| `tool_success` | 0.15 | ツール成功率の重み |
| `query_coverage` | 0.15 | クエリ網羅度の重み |

### 5.2 ConfidenceThresholds（閾値設定）

介入レベルを決定する閾値。

```python
class ConfidenceThresholds(BaseModel):
    silent: float = 0.9
    notify: float = 0.7
    confirm: float = 0.4
```

| キー | デフォルト値 | 説明 | 介入レベル |
|-----|-------------|------|-----------|
| `silent` | 0.9 | 自動進行の閾値 | score >= 0.9 → SILENT |
| `notify` | 0.7 | ステータス表示の閾値 | score >= 0.7 → NOTIFY |
| `confirm` | 0.4 | 確認要求の閾値 | score >= 0.4 → CONFIRM |
| (それ以下) | - | エスカレーションの閾値 | score < 0.4 → ESCALATE |

### 5.3 EmbeddingConfig（Embedding設定）

SourceAgreementCalculatorで使用するEmbeddingモデルの設定。

```python
class EmbeddingConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-embedding-001"
    dimensions: int = 3072
```

| キー | デフォルト値 | 説明 |
|-----|-------------|------|
| `provider` | "gemini" | Embeddingプロバイダー |
| `model` | "gemini-embedding-001" | 使用するEmbeddingモデル |
| `dimensions` | 3072 | Embeddingの次元数 |

### 5.4 LLMConfig（LLM設定）

LLMSelfEvaluator、QueryCoverageCalculatorで使用するLLMの設定。

```python
class LLMConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 30
```

| キー | デフォルト値 | 説明 |
|-----|-------------|------|
| `provider` | "gemini" | LLMプロバイダー |
| `model` | "gemini-2.5-flash" | 使用するモデル |
| `temperature` | 0.7 | 生成時の温度（評価時は0.0に上書き） |
| `max_tokens` | 4096 | 最大トークン数 |
| `timeout` | 30 | タイムアウト秒数 |

### 5.5 LLMSelfEvaluator.EVAL_PROMPT

LLM自己評価用のプロンプトテンプレート。

```python
EVAL_PROMPT = """以下の基準に基づいて、回答の確信度を0.0から1.0の数値で評価してください。

【評価基準】
1. 正確性 (Accuracy):
   - 回答は提供された情報源（検索結果）に基づいているか？
   - 情報源にない情報を捏造していないか？
2. 適切性 (Relevance):
   - ユーザーの質問に直接的かつ明確に答えているか？
   - 質問の意図を正しく理解しているか？
3. スタイル (Style):
   - 親しみやすく、丁寧な日本語（です・ます調）か？
   - 読みやすい構成か？

【スコアの目安】
- 1.0: 完全に正確で、適切かつスタイルも完璧
- 0.8: ほぼ確実（信頼できる情報源あり、回答も適切）
- 0.6: やや確信あり（関連情報はあるが、完全ではない）
- 0.4: 不確実（情報が限定的、または質問への回答として不十分）
- 0.2: 推測に近い（根拠が弱い）
- 0.0: 全く分からない、または不適切な回答
...
"""
```

### 5.6 QueryCoverageCalculator.COVERAGE_PROMPT

クエリ網羅度評価用のプロンプトテンプレート。

```python
COVERAGE_PROMPT = """以下の質問に対する回答が、質問のすべての要素をカバーしているか評価してください。

質問: {query}
回答: {answer}

網羅度（0.0-1.0の数値のみ回答）:
- 1.0: すべての質問要素に完全に回答
- 0.8: ほぼすべての要素に回答
- 0.6: 主要な要素に回答
- 0.4: 一部の要素のみに回答
- 0.2: ほとんど回答できていない
- 0.0: 全く回答できていない

数値のみ回答:"""
```

---

## 6. 使用例

### 6.1 基本的なワークフロー

```python
from grace.confidence import (
    ConfidenceFactors,
    create_confidence_calculator,
)

# 1. Calculatorを作成
calculator = create_confidence_calculator()

# 2. 検索結果から要素を作成
factors = ConfidenceFactors(
    search_result_count=5,
    search_avg_score=0.75,
    search_max_score=0.92,
    search_score_variance=0.05,
    tool_success_rate=1.0,
    tool_execution_count=1,
    tool_success_count=1,
    is_search_step=True
)

# 3. 信頼度を計算
score = calculator.calculate(factors)
print(f"信頼度: {score.score}, レベル: {score.level}")
# 出力: 信頼度: 0.92, レベル: high

# 4. アクションを決定
action = calculator.decide_action(score)
print(f"介入レベル: {action.level}, 推奨: {action.suggested_action}")
# 出力: 介入レベル: InterventionLevel.SILENT, 推奨: proceed

# 5. 進行可否を判定
if action.should_proceed:
    print("自動進行します")
```

### 6.2 LLM評価を使用したワークフロー

```python
from grace.confidence import (
    ConfidenceFactors,
    create_confidence_calculator,
    create_llm_evaluator,
)

# 1. Calculatorを作成
calculator = create_confidence_calculator()

# 2. 要素を作成
factors = ConfidenceFactors(
    search_result_count=3,
    search_max_score=0.85,
    is_search_step=True
)

# 3. LLM評価による信頼度計算
score = calculator.llm_calculate(
    factors=factors,
    step_description="東京の天気を検索して回答する",
    tool_output="検索結果: 東京は晴れ、気温25度、湿度60%..."
)

print(f"LLM評価スコア: {score.score}")
print(f"評価理由: {score.reason}")

# 4. 別途、自己評価のみを実行する場合
evaluator = create_llm_evaluator()
confidence = evaluator.evaluate(
    query="東京の天気は？",
    answer="東京は現在晴れで、気温は25度です。",
    sources=["気象庁", "Yahoo天気"]
)
print(f"自己評価: {confidence}")
```

### 6.3 複数ステップの集計

```python
from grace.confidence import (
    ConfidenceFactors,
    ConfidenceScore,
    create_confidence_calculator,
    create_confidence_aggregator,
)

# 1. 各ステップの信頼度を計算
calculator = create_confidence_calculator()

step1_factors = ConfidenceFactors(search_max_score=0.9, is_search_step=True)
step2_factors = ConfidenceFactors(search_max_score=0.7, is_search_step=True)
step3_factors = ConfidenceFactors(
    llm_self_confidence=0.8,
    tool_success_rate=1.0,
    is_search_step=False
)

scores = [
    calculator.calculate(step1_factors),
    calculator.calculate(step2_factors),
    calculator.calculate(step3_factors),
]

# 2. 集計
aggregator = create_confidence_aggregator()

# 平均
avg = aggregator.aggregate(scores, method="mean")
print(f"平均信頼度: {avg}")

# 最小値（最も弱い部分を重視）
min_score = aggregator.aggregate(scores, method="min")
print(f"最小信頼度: {min_score}")

# 重み付き（後半のステップを重視）
weighted = aggregator.aggregate(scores, method="weighted")
print(f"重み付き信頼度: {weighted}")

# 3. 重要度チェック付き集計
final_score, has_failure = aggregator.aggregate_with_critical_check(
    scores,
    critical_threshold=0.3
)

if has_failure:
    print(f"警告: 重要ステップに問題があります（最終スコア: {final_score}）")
else:
    print(f"全ステップ正常（最終スコア: {final_score}）")
```

---

## 7. エクスポート

`__all__`でエクスポートされる要素：

```python
__all__ = [
    # Data classes
    "ConfidenceFactors",
    "ConfidenceScore",
    "ActionDecision",

    # Enums
    "InterventionLevel",

    # Calculators
    "ConfidenceCalculator",
    "LLMSelfEvaluator",
    "SourceAgreementCalculator",
    "QueryCoverageCalculator",
    "ConfidenceAggregator",

    # Factory functions
    "create_confidence_calculator",
    "create_llm_evaluator",
    "create_source_agreement_calculator",
    "create_query_coverage_calculator",
    "create_confidence_aggregator",
]
```

---

## 8. 変更履歴

| バージョン | 変更内容 |
|-----------|---------|
| 1.0 | 初版作成（2025-01-29） |
| 1.1 | config.pyの情報を反映：ConfidenceWeights、ConfidenceThresholds、EmbeddingConfig、LLMConfigの具体的なデフォルト値を追加 |

> 📝 **注意**: コード内のコメントによると、2025-12-26にLLM化の改修が行われています。

---

## 付録: 依存関係図

```
confidence.py
    │
    ├──► google-genai
    │        └── genai.Client
    │        └── genai.types.GenerateContentConfig
    │
    ├──► dataclasses
    │        └── dataclass
    │        └── field
    │
    ├──► typing
    │        └── Optional, List, Literal, Dict, Any
    │
    ├──► enum
    │        └── Enum
    │
    ├──► logging
    │        └── getLogger
    │
    └──► .config (内部)
             └── get_config()
             └── GraceConfig
```

---

## 関連ドキュメント

| ドキュメント | 説明 |
|-------------|------|
| `config.md` | GraceConfig設定管理の詳細ドキュメント |
| `planner.md` | 計画生成エージェントのドキュメント |
| `executor.md` | 計画実行エージェントのドキュメント |
| `reasoner.md` | 推論エージェントのドキュメント |
