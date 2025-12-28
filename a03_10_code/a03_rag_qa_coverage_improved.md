# a03_rag_qa_coverage_improved.py - 高カバレッジ・多様性Q/A生成システム

**バージョン**: 2.0
**作成日**: 2025-11-08
**最終更新**: 2025-12-18（全面改修・IPO追加）

---

## 目次

1. [概要](#1-概要)
2. [方式・手法の位置づけ](#2-方式手法の位置づけ)
3. [アーキテクチャ](#3-アーキテクチャ)
4. [チャンキング技術詳細](#4-チャンキング技術詳細)
5. [キーワード抽出](#5-キーワード抽出)
6. [階層化質問タイプ](#6-階層化質問タイプ)
7. [チャンク複雑度分析](#7-チャンク複雑度分析)
8. [ドメイン適応戦略](#8-ドメイン適応戦略)
9. [Q/A生成戦略](#9-qa生成戦略)
10. [カバレッジ分析](#10-カバレッジ分析)
11. [関数リファレンス（IPO形式）](#11-関数リファレンスipo形式)
12. [コマンドラインオプション](#12-コマンドラインオプション)
13. [実行方法](#13-実行方法)
14. [出力ファイル](#14-出力ファイル)
15. [トラブルシューティング](#15-トラブルシューティング)
16. [付録: 品質改善機能一覧](#16-付録-品質改善機能一覧)

---

## 1. 概要

### 1.1 目的

`a03_rag_qa_coverage_improved.py`は、**95%以上のカバレッジ**と**質問タイプの多様性**を最優先するQ/A生成システムです。
「基礎」「理解」「応用」といった階層化された質問タイプ定義（3階層11タイプ）に基づき、テンプレートベースの手法とLLM (`gemini-2.0-flash`) をハイブリッドに組み合わせることで、ドキュメントの情報を余すことなくQ/A化します。

### 1.2 特徴

- **高カバレッジ重視**: 95%以上のカバレッジ達成を目標
- **質問タイプの多様性**: 3階層11タイプの階層化質問定義
- **テンプレート主体**: ルール/テンプレートベース + オプションLLM
- **ドメイン適応**: データセット別の最適化戦略
- **チャンク複雑度分析**: 動的なQ/A生成戦略調整

### 1.3 推奨用途

> **【使い分けガイド】**
> - **効率・速度重視**: [a10_qa_optimized_hybrid_batch.py](a10_qa_optimized_hybrid_batch.md) を使用（通常はこちらで十分な品質が得られます）
> - **品質・網羅性重視**: 本スクリプト (`a03`) を使用。教育用コンテンツ作成や、情報の取りこぼしが許されない高精度な検索システム構築に適しています
> - **大規模分散処理**: [a02_make_qa_para.py](a02_make_qa_para.md) を使用（Celery/Redis並列処理）

### 1.4 起動コマンド

```bash
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_cc_news.csv \
  --dataset cc_news \
  --analyze-coverage \
  --coverage-threshold 0.60 \
  --qa-per-chunk 10 \
  --max-chunks 2000 \
  --model gemini-2.0-flash
```

### 1.5 主要機能

| 機能 | 説明 | 技術 |
|------|------|------|
| セマンティックチャンク分割 | 段落優先のセマンティック分割 | `helper_rag_qa.SemanticCoverage` |
| 階層化質問タイプ | 3階層11タイプの質問タイプ定義 | `QUESTION_TYPES_HIERARCHY` |
| チャンク複雑度分析 | 専門用語密度・平均文長分析 | `analyze_chunk_complexity()` |
| ドメイン適応戦略 | データセット別の最適化戦略 | `DOMAIN_SPECIFIC_STRATEGIES` |
| 包括的Q/A生成 | ルール/テンプレート/LLMハイブリッド | `generate_comprehensive_qa_for_chunk()` |
| バッチ埋め込み生成 | Gemini API制限考慮のバッチ処理 | `calculate_improved_coverage()` |
| 改良版カバレッジ分析 | 3段階の分布評価（高・中・低） | `gemini-embedding-001` |

### 1.6 対応データセット

| データセット | キー | 言語 | 説明 |
|------------|------|------|------|
| CC-News | `cc_news` | 英語 | 英語ニュース記事 |
| CC100日本語 | `japanese_text` | 日本語 | Webテキストコーパス |
| Wikipedia日本語版 | `wikipedia_ja` | 日本語 | 百科事典的知識 |
| Livedoorニュース | `livedoor` | 日本語 | ニュースコーパス |

---

## 2. 方式・手法の位置づけ

### 2.1 Q/A生成システム比較表

本プロジェクトには3つのQ/A生成スクリプトがあり、それぞれ異なる用途に最適化されています。

| 項目 | a02 (分散並列) | a03 (カバレッジ重視) | a10 (推奨・ハイブリッド) |
|------|---------------|---------------------|-------------------------|
| **推奨度** | ★★☆ | ★★★ | ★★★★★ |
| **主目的** | 大規模分散処理 | カバレッジ最大化・多様性確保 | 品質とコストの最適バランス |
| **Q/A生成手法** | LLM（Gemini） | テンプレート + オプションLLM | ルール + LLMハイブリッド |
| **並列処理** | Celery/Redis分散 | なし（シングルプロセス） | 3並列バッチ |
| **品質モード** | なし | 複雑度ベース動的調整 | quick/balanced/thorough |
| **期待カバレッジ** | 90-95% | **95%+** | 80%+ |
| **実行速度** | 高速（分散時） | 高速（テンプレート）〜中速 | 高速 |
| **コスト** | 中 | **低〜中** | 低 |
| **用途** | 大規模データ処理 | 教育コンテンツ・高精度検索 | 一般的なRAG用途 |

### 2.2 技術要素の対応表

| 技術要素 | a02 | a03 | a10 |
|---------|-----|-----|-----|
| SemanticCoverage | ✅ | ✅ | ✅ |
| KeywordExtractor | ✅ | ✅ | ✅ |
| LLM Q/A生成 | ✅ (メイン) | ✅ (オプション) | ✅ (ハイブリッド) |
| テンプレートQ/A | - | ✅ (メイン) | ✅ |
| Celery/Redis | ✅ | - | - |
| バッチ処理 | - | ✅ (埋め込み) | ✅ (生成+埋め込み) |
| 品質モード | - | 複雑度ベース | 3段階モード |
| 階層化質問タイプ | - | ✅ (3階層11タイプ) | - |
| ドメイン適応 | - | ✅ (4データセット) | - |

### 2.3 処理フローの違い

```
a02 (分散並列):
  データ → チャンク分割 → [Celeryタスク分散] → LLM生成 → 結果統合

a03 (カバレッジ重視):
  データ → チャンク分割 → 複雑度分析 → テンプレート生成(+LLM) → カバレッジ分析

a10 (ハイブリッド): ★推奨
  データ → チャンク分割 → [品質モード選択] → ルール+LLMハイブリッド → カバレッジ分析
```

### 2.4 選択ガイドライン

```
Q: どのスクリプトを使うべき？

├── 大規模データ（10万チャンク以上）
│   └── a02 (Celery分散処理)
│
├── カバレッジ95%以上が必須
│   └── a03 (本スクリプト)
│
├── 教育用コンテンツ作成
│   └── a03 (階層化質問タイプ)
│
├── ドメイン特化の最適化が必要
│   └── a03 (ドメイン適応戦略)
│
└── 一般的なRAG用途
    └── a10 (推奨・ハイブリッド)
```

### 2.5 期待される性能

| 設定 | 文書数 | チャンク数 | Q/A数 | 実行時間 | カバレッジ予想 | コスト |
|-----|-------|--------|---------|--------|--------------|--------|
| テスト | 150 | 609 | 7,308 | 2分 | 99.7% (0.52) | $0.001 |
| 推奨 | 自動 | 2,000 | 20,000 | 8-10分 | 95%+ (0.60) | $0.005 |
| 中規模 | 1,000 | 2,400 | 24,000 | 10-12分 | 95%+ (0.60) | $0.006 |
| 全文書 | 7,499 | 18,000 | 144,000 | 60-90分 | 95%+ (0.60) | $0.025 |

---

## 3. アーキテクチャ

### 3.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────────┐
│                 a03_rag_qa_coverage_improved.py                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ [1] データ読み込み                                            │   │
│  │     load_input_data()                                        │   │
│  │     ├── CSV: pandas.read_csv()                               │   │
│  │     └── TXT: テキスト直接読み込み                              │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ [2] セマンティックチャンク分割                                │   │
│  │     SemanticCoverage.create_semantic_chunks()                │   │
│  │     ├── max_tokens: 200                                      │   │
│  │     ├── min_tokens: 50                                       │   │
│  │     └── prefer_paragraphs: True（段落優先）                   │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ [3] Q/A生成 (process_with_improved_methods)                  │   │
│  │     ├── テンプレート: generate_comprehensive_qa_for_chunk()  │   │
│  │     │   ├── 包括的Q/A (full_chunk)                           │   │
│  │     │   ├── 詳細事実Q/A (factual_detailed)                   │   │
│  │     │   ├── 文脈Q/A (contextual)                             │   │
│  │     │   ├── キーワードQ/A (keyword_based)                    │   │
│  │     │   └── テーマQ/A (thematic)                             │   │
│  │     │                                                        │   │
│  │     └── LLM (オプション): generate_llm_qa()                  │   │
│  │         └── gemini-2.0-flash (構造化出力)                    │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ [4] カバレッジ分析                                            │   │
│  │     calculate_improved_coverage()                            │   │
│  │     ├── チャンク埋め込み: gemini-embedding-001               │   │
│  │     ├── Q/A埋め込み: バッチ処理 (MAX_BATCH_SIZE=100)         │   │
│  │     └── コサイン類似度行列計算                                │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ [5] 結果保存                                                  │   │
│  │     save_results() → qa_output/a03/                          │   │
│  │     ├── qa_pairs_{dataset}_{timestamp}.json                  │   │
│  │     ├── qa_pairs_{dataset}_{timestamp}.csv                   │   │
│  │     ├── coverage_{dataset}_{timestamp}.json                  │   │
│  │     └── summary_{dataset}_{timestamp}.json                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 依存モジュール

```python
# 内部モジュール
from helper_rag_qa import SemanticCoverage
from helper_llm import create_llm_client, LLMClient
from models import QAPairsResponse

# 外部ライブラリ
import pandas as pd
import numpy as np
from collections import Counter
import MeCab  # オプション（フォールバック対応）
```

### 3.3 クラス・定数構成

```
a03_rag_qa_coverage_improved.py
├── KeywordExtractor (クラス)
│   ├── __init__(prefer_mecab: bool)
│   ├── _check_mecab_availability() -> bool
│   ├── extract(text, top_n) -> List[str]
│   ├── _extract_with_mecab(text, top_n) -> List[str]
│   ├── _extract_with_regex(text, top_n) -> List[str]
│   └── _filter_and_count(words, top_n) -> List[str]
│
├── QUESTION_TYPES_HIERARCHY (定数)
│   ├── basic: {definition, identification, enumeration}
│   ├── understanding: {cause_effect, process, mechanism, comparison}
│   └── application: {synthesis, evaluation, prediction, practical}
│
├── DOMAIN_SPECIFIC_STRATEGIES (定数)
│   ├── cc_news
│   ├── wikipedia_ja
│   ├── livedoor
│   └── japanese_text
│
├── DATASET_CONFIGS (定数)
│
└── 関数群
    ├── get_keyword_extractor() - シングルトン
    ├── analyze_chunk_complexity() - 複雑度分析
    ├── generate_llm_qa() - LLM Q/A生成
    ├── generate_comprehensive_qa_for_chunk() - テンプレートQ/A
    ├── generate_advanced_qa_for_chunk() - 高度Q/A生成
    ├── load_input_data() - データ読み込み
    ├── process_with_improved_methods() - メイン処理
    ├── calculate_improved_coverage() - カバレッジ計算
    ├── save_results() - 結果保存
    └── main() - エントリポイント
```

---

## 4. チャンキング技術詳細

### 4.1 チャンキング階層

```
                    ┌─────────────────┐
                    │   入力テキスト    │
                    │ (CSV/TXT形式)    │
                    └────────┬────────┘
                             │
                             ▼
            ┌────────────────────────────────┐
            │      SemanticCoverage           │
            │   create_semantic_chunks()      │
            │                                 │
            │   設定:                          │
            │   ├── max_tokens: 200           │
            │   ├── min_tokens: 50            │
            │   └── prefer_paragraphs: True   │
            └────────────────┬───────────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │   段落優先セマンティック   │
                │       チャンク群         │
                │                        │
                │  [chunk_0] 150 tokens  │
                │  [chunk_1] 180 tokens  │
                │  [chunk_2] 120 tokens  │
                │        ...             │
                └────────────────────────┘
```

### 4.2 SemanticCoverage設定

本スクリプトでのSemanticCoverage初期化パラメータ:

```python
analyzer = SemanticCoverage(embedding_model="gemini-embedding-001")
chunks = analyzer.create_semantic_chunks(
    document=document_text,
    max_tokens=200,          # チャンクの最大トークン数
    min_tokens=50,           # 最小トークン数（小さすぎるチャンクは自動マージ）
    prefer_paragraphs=True,  # 段落優先モード（セマンティック境界を重視）
    verbose=False
)
```

### 4.3 段落優先モードの特徴

| 特徴 | 説明 |
|------|------|
| セマンティック境界 | 段落の区切りを優先的にチャンク境界として使用 |
| 自動マージ | min_tokens未満の小さなチャンクは隣接チャンクと自動統合 |
| 文完結性 | 文の途中でチャンクが切れないよう配慮 |
| 言語対応 | 日本語（。）と英語（.）の文区切りを自動判定 |

### 4.4 チャンクサンプリング戦略

大規模データの場合、均等サンプリングを実施:

```python
if total_chunks > max_chunks_to_process:
    # 均等にサンプリング
    step = total_chunks // max_chunks_to_process
    selected_chunks = [chunks[i] for i in range(0, total_chunks, step)][:max_chunks_to_process]
```

---

## 5. キーワード抽出

### 5.1 KeywordExtractorクラス

MeCabと正規表現を統合したキーワード抽出クラスです。

```python
class KeywordExtractor:
    """
    MeCabが利用可能な場合は複合名詞抽出を優先し、
    利用不可の場合は正規表現版に自動フォールバック
    """
```

### 5.2 抽出モード

| モード | 条件 | 精度 | 速度 |
|--------|------|------|------|
| MeCab | MeCabインストール済み | 高（複合名詞対応） | 中 |
| 正規表現 | MeCab未インストール | 中 | 高 |

### 5.3 ストップワード定義

```python
self.stopwords = {
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
    'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる',
    'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と',
    'の', 'から', 'まで', '等', 'など', 'よる', 'おく', 'くる'
}
```

### 5.4 正規表現パターン

```python
# カタカナ語、漢字複合語、英数字を抽出
pattern = r'[ァ-ヴー]{2,}|[一-龥]{2,}|[A-Za-z]{2,}[A-Za-z0-9]*'
```

### 5.5 シングルトンパターン

```python
_keyword_extractor = None  # グローバルインスタンス

def get_keyword_extractor() -> KeywordExtractor:
    """KeywordExtractorのシングルトンインスタンスを取得"""
    global _keyword_extractor
    if _keyword_extractor is None:
        _keyword_extractor = KeywordExtractor()
    return _keyword_extractor
```

---

## 6. 階層化質問タイプ

### 6.1 3階層11タイプの定義

```python
QUESTION_TYPES_HIERARCHY = {
    "basic": {
        "definition":      {"ja": "定義型（〜とは何ですか？）",          "en": "Definition"},
        "identification":  {"ja": "識別型（〜の例を挙げてください）",    "en": "Identification"},
        "enumeration":     {"ja": "列挙型（〜の種類/要素は？）",        "en": "Enumeration"}
    },
    "understanding": {
        "cause_effect":    {"ja": "因果関係型（〜の結果/影響は？）",    "en": "Cause-Effect"},
        "process":         {"ja": "プロセス型（〜はどのように行われますか？）", "en": "Process"},
        "mechanism":       {"ja": "メカニズム型（〜の仕組みは？）",     "en": "Mechanism"},
        "comparison":      {"ja": "比較型（〜と〜の違いは？）",         "en": "Comparison"}
    },
    "application": {
        "synthesis":       {"ja": "統合型（〜を組み合わせるとどうなりますか？）", "en": "Synthesis"},
        "evaluation":      {"ja": "評価型（〜の長所と短所は？）",       "en": "Evaluation"},
        "prediction":      {"ja": "予測型（〜の場合どうなりますか？）",  "en": "Prediction"},
        "practical":       {"ja": "実践型（〜はどのように活用されますか？）", "en": "Practical"}
    }
}
```

### 6.2 階層別の特徴

| 階層 | タイプ数 | 目的 | 難易度 | 期待比率 |
|------|---------|------|--------|---------|
| basic | 3 | 基本的な事実確認 | 低 | 33% |
| understanding | 4 | 概念の理解・関係性 | 中 | 50% |
| application | 4 | 応用・実践・評価 | 高 | 17% |

### 6.3 質問タイプの分布例

```
📊 Q/A品質統計（期待値）:
  - 基礎レベル: 1426件（定義・識別・列挙型）     ~33%
  - 理解レベル: 2137件（因果関係・プロセス・メカニズム・比較型） ~50%
  - 応用レベル: 715件（統合・評価・予測・実践型）  ~17%
```

---

## 7. チャンク複雑度分析

### 7.1 analyze_chunk_complexity関数

チャンクの複雑度を分析して、適切なQ/A生成戦略を決定します。

### 7.2 複雑度指標

| 指標 | 説明 | 計算方法 |
|------|------|----------|
| complexity_level | 複雑度レベル (high/medium/low) | スコアから判定 |
| complexity_score | 複雑度スコア (0-6) | 各指標の合計 |
| technical_terms | 専門用語リスト | 正規表現抽出 |
| avg_sentence_length | 平均文長 | tokens / sentences |
| concept_density | 概念密度 (%) | terms / tokens × 100 |
| sentence_count | 文数 | 句点区切り |
| token_count | トークン数 | LLMClient.count_tokens() |
| has_statistics | 統計情報有無 | 数値パターン検出 |

### 7.3 複雑度レベル判定ロジック

```python
# スコア加算ルール
if concept_density > 5:     complexity_score += 3
elif concept_density > 2:   complexity_score += 2
else:                       complexity_score += 1

if avg_sentence_length > 30: complexity_score += 2
elif avg_sentence_length > 20: complexity_score += 1

if has_statistics:          complexity_score += 1

# レベル判定
if complexity_score >= 5:   complexity_level = "high"
elif complexity_score >= 3: complexity_level = "medium"
else:                       complexity_level = "low"
```

### 7.4 複雑度に基づくQ/A分布

| 複雑度レベル | basic | understanding | application |
|-------------|-------|---------------|-------------|
| high | 1 | 3 | 1 |
| medium | 2 | 2 | 1 |
| low | 3 | 1 | 0 |

---

## 8. ドメイン適応戦略

### 8.1 DOMAIN_SPECIFIC_STRATEGIES

データセット（ドメイン）ごとに、生成する質問タイプの重点や避けるべきタイプを定義しています。

```python
DOMAIN_SPECIFIC_STRATEGIES = {
    "cc_news": {
        "focus_types": ["cause_effect", "process", "comparison", "evaluation"],
        "avoid_types": ["definition"],  # ニュースでは基本定義は少なめ
        "emphasis": "時事性と社会的影響"
    },
    "wikipedia_ja": {
        "focus_types": ["definition", "mechanism", "enumeration", "comparison"],
        "avoid_types": ["prediction"],  # 百科事典では推測は避ける
        "emphasis": "正確な定義と体系的な知識"
    },
    "livedoor": {
        "focus_types": ["cause_effect", "evaluation", "practical", "comparison"],
        "avoid_types": ["mechanism"],  # 一般ニュースでは技術的詳細は少なめ
        "emphasis": "読者の関心と実用性"
    },
    "japanese_text": {
        "focus_types": ["definition", "process", "practical", "comparison"],
        "avoid_types": [],
        "emphasis": "一般的な理解と応用"
    }
}
```

### 8.2 ドメイン別の質問タイプ重点

| ドメイン | 重点タイプ | 回避タイプ | 特徴 |
|---------|-----------|-----------|------|
| cc_news | 因果関係, プロセス, 比較, 評価 | 定義 | 時事性重視 |
| wikipedia_ja | 定義, メカニズム, 列挙, 比較 | 予測 | 体系的知識 |
| livedoor | 因果関係, 評価, 実践, 比較 | メカニズム | 実用性重視 |
| japanese_text | 定義, プロセス, 実践, 比較 | なし | バランス型 |

### 8.3 DATASET_CONFIGS

```python
DATASET_CONFIGS = {
    "cc_news": {
        "name": "CC-News英語ニュース",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en"
    },
    "japanese_text": {
        "name": "日本語Webテキスト",
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja"
    },
    "wikipedia_ja": {
        "name": "Wikipedia日本語版",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja"
    },
    "livedoor": {
        "name": "ライブドアニュース",
        "text_column": "Combined_Text",
        "title_column": "title",
        "category_column": "category",
        "lang": "ja"
    }
}
```

---

## 9. Q/A生成戦略

### 9.1 生成方式の概要

本スクリプトは2つの生成方式をサポートしています:

| 方式 | 関数 | デフォルト | 特徴 |
|------|------|-----------|------|
| テンプレート | `generate_comprehensive_qa_for_chunk()` | ✅ | 高速・低コスト・高カバレッジ |
| LLM | `generate_llm_qa()` | オプション | 高品質・中コスト |

### 9.2 テンプレートベース生成 (generate_comprehensive_qa_for_chunk)

5つの戦略を組み合わせてQ/Aを生成:

| 戦略 | タイプ | 説明 |
|------|--------|------|
| 1. 包括的 | comprehensive | チャンク全体の要約的質問 |
| 2. 詳細事実 | factual_detailed | 文ごとの詳細質問 |
| 3. 文脈 | contextual | 前文との関係性質問 |
| 4. キーワード | keyword_based | 重要語句に基づく質問 |
| 5. テーマ | thematic | 主要テーマに関する質問 |

### 9.3 LLMベース生成 (generate_llm_qa)

```python
def generate_llm_qa(chunk_text: str, chunk_idx: int, model: str, qa_per_chunk: int = 2) -> List[Dict]:
    """
    LLMを使用して高品質なQ/Aペアを生成
    - UnifiedLLMClient.generate_structured を利用
    - Pydanticモデル (QAPairsResponse) で構造化出力
    """
```

### 9.4 高度Q/A生成 (generate_advanced_qa_for_chunk)

複雑度分析とドメイン適応を組み合わせた高度な生成:

```python
def generate_advanced_qa_for_chunk(chunk_text, chunk_idx, qa_per_chunk=5,
                                   lang="auto", dataset_type="custom"):
    # 1. チャンク複雑度分析
    complexity = analyze_chunk_complexity(chunk_text, lang)

    # 2. ドメイン適応戦略の取得
    domain_strategy = DOMAIN_SPECIFIC_STRATEGIES.get(dataset_type, {...})

    # 3. 複雑度に基づく質問タイプの選択
    # 4. 高度なQ/A生成戦略を呼び出し
    #    - コンテキスト強化型Q/A生成
    #    - マルチホップ推論型Q/A生成
    #    - 階層化質問タイプに基づくQ/A生成

    # 5. 品質スコアリング
```

### 9.5 重複除去

質問の先頭30文字をキーとして簡易的な重複除去を実施:

```python
unique_questions = {}
for qa in all_qas:
    q = qa.get('question', '')
    q_key = q[:30]  # 簡易的な重複チェック
    if q_key not in unique_questions:
        unique_questions[q_key] = qa
```

---

## 10. カバレッジ分析

### 10.1 改良版カバレッジ計算

`calculate_improved_coverage()`関数でバッチ処理による効率的なカバレッジ計算を実施します。

### 10.2 バッチ処理設定

```python
MAX_BATCH_SIZE = 100  # Gemini APIの安全なバッチサイズ

if len(qa_texts) <= MAX_BATCH_SIZE:
    # 一度にすべて処理可能
    qa_embeddings = analyzer.generate_embeddings(qa_chunks)
else:
    # バッチサイズを超える場合は分割処理
    for i in range(0, len(qa_texts), MAX_BATCH_SIZE):
        batch = qa_texts[i:i+MAX_BATCH_SIZE]
        batch_embeddings = analyzer.generate_embeddings(batch_chunks)
        qa_embeddings.extend(batch_embeddings)
```

### 10.3 Q/Aテキストの結合方式

回答を強調するための重み付け結合:

```python
# 質問1回 + 回答2回で回答を強調
combined_text = f"{question} {answer} {answer}"
```

### 10.4 カバレッジ結果の構造

```python
coverage_results = {
    "coverage_rate": float,           # カバレッジ率 (0.0-1.0)
    "covered_chunks": int,            # カバーされたチャンク数
    "total_chunks": int,              # 総チャンク数
    "threshold": float,               # 判定閾値
    "avg_max_similarity": float,      # 平均最大類似度
    "min_max_similarity": float,      # 最小最大類似度
    "max_max_similarity": float,      # 最大最大類似度
    "uncovered_chunks": List[int],    # 未カバーチャンクのインデックス
    "coverage_distribution": {
        "high_coverage": int,         # 高カバレッジ (≥0.7) チャンク数
        "medium_coverage": int,       # 中カバレッジ (0.5-0.7) チャンク数
        "low_coverage": int           # 低カバレッジ (<0.5) チャンク数
    }
}
```

### 10.5 カバレッジ分布評価

| レベル | 類似度範囲 | 意味 |
|--------|-----------|------|
| 高カバレッジ | ≥ 0.7 | Q/Aがチャンク内容を高精度でカバー |
| 中カバレッジ | 0.5-0.7 | Q/Aがチャンク内容を部分的にカバー |
| 低カバレッジ | < 0.5 | Q/Aがチャンク内容を十分にカバーしていない |

---

## 11. 関数リファレンス（IPO形式）

### 11.1 データ読み込み系

#### load_input_data()

| 項目 | 内容 |
|------|------|
| **機能** | 入力ファイルからテキストデータを読み込み |
| **定義** | `a03_rag_qa_coverage_improved.py:448-498` |

| INPUT | 型 | 説明 |
|-------|---|------|
| input_file | str | 入力ファイルパス（CSV/TXT） |
| dataset_type | Optional[str] | データセットタイプ（DATASET_CONFIGSのキー） |
| max_docs | Optional[int] | 処理する最大文書数 |

| PROCESS |
|---------|
| 1. ファイルの存在確認 |
| 2. CSVの場合: pandas.read_csv()で読み込み |
| 3. データセット設定に基づきテキストカラムを特定 |
| 4. max_docs指定時は先頭N件に制限 |
| 5. TXTの場合: 直接テキスト読み込み |
| 6. テキストを結合して返却 |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| combined_text | str | 結合されたテキスト |

---

### 11.2 キーワード抽出系

#### KeywordExtractor.__init__()

| 項目 | 内容 |
|------|------|
| **機能** | KeywordExtractorの初期化 |
| **定義** | `a03_rag_qa_coverage_improved.py:157-176` |

| INPUT | 型 | 説明 |
|-------|---|------|
| prefer_mecab | bool | MeCabを優先使用するか（デフォルト: True） |

| PROCESS |
|---------|
| 1. MeCab利用可能性をチェック |
| 2. ストップワードセットを初期化 |
| 3. 利用可能モードをログ出力 |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| self | KeywordExtractor | 初期化済みインスタンス |

#### KeywordExtractor.extract()

| 項目 | 内容 |
|------|------|
| **機能** | テキストからキーワードを抽出（自動フォールバック対応） |
| **定義** | `a03_rag_qa_coverage_improved.py:189-209` |

| INPUT | 型 | 説明 |
|-------|---|------|
| text | str | 分析対象テキスト |
| top_n | int | 抽出するキーワード数（デフォルト: 5） |

| PROCESS |
|---------|
| 1. MeCab利用可能 & prefer_mecab=Trueなら_extract_with_mecab() |
| 2. MeCab抽出失敗時は_extract_with_regex()にフォールバック |
| 3. 頻度ベースで上位N件を返却 |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| keywords | List[str] | 抽出されたキーワードリスト |

#### get_keyword_extractor()

| 項目 | 内容 |
|------|------|
| **機能** | KeywordExtractorのシングルトンインスタンスを取得 |
| **定義** | `a03_rag_qa_coverage_improved.py:271-276` |

| INPUT | 型 | 説明 |
|-------|---|------|
| なし | | |

| PROCESS |
|---------|
| 1. グローバル変数_keyword_extractorがNoneかチェック |
| 2. Noneなら新規インスタンス作成 |
| 3. キャッシュされたインスタンスを返却 |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| extractor | KeywordExtractor | シングルトンインスタンス |

---

### 11.3 複雑度分析系

#### analyze_chunk_complexity()

| 項目 | 内容 |
|------|------|
| **機能** | チャンクの複雑度を分析してQ/A生成戦略を決定 |
| **定義** | `a03_rag_qa_coverage_improved.py:306-389` |

| INPUT | 型 | 説明 |
|-------|---|------|
| chunk_text | str | 分析対象テキスト |
| lang | str | 言語（"ja", "en", "auto"）デフォルト: "auto" |

| PROCESS |
|---------|
| 1. 言語の自動検出（日本語インジケータの出現頻度） |
| 2. LLMClientでトークン数をカウント |
| 3. 文分割と基本メトリクス計算 |
| 4. 専門用語の検出（言語別正規表現） |
| 5. 平均文長と概念密度を計算 |
| 6. 数値・統計情報の存在をチェック |
| 7. 複雑度スコアを集計 |
| 8. 複雑度レベルを判定（high/medium/low） |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| Dict | Dict | 複雑度指標の辞書（complexity_level, complexity_score, technical_terms, avg_sentence_length, concept_density, sentence_count, token_count, has_statistics, numeric_data, lang） |

---

### 11.4 Q/A生成系

#### generate_llm_qa()

| 項目 | 内容 |
|------|------|
| **機能** | LLMを使用して高品質なQ/Aペアを生成 |
| **定義** | `a03_rag_qa_coverage_improved.py:93-142` |

| INPUT | 型 | 説明 |
|-------|---|------|
| chunk_text | str | チャンクのテキスト |
| chunk_idx | int | チャンクのインデックス |
| model | str | 使用するモデル（例: "gemini-2.0-flash"） |
| qa_per_chunk | int | チャンクあたりのQ/A数（デフォルト: 2） |

| PROCESS |
|---------|
| 1. Gemini LLMクライアントを作成 |
| 2. システムプロンプトとユーザープロンプトを構築 |
| 3. generate_structured()で構造化出力を取得 |
| 4. QAPairsResponseからQ/Aペアを抽出 |
| 5. メタデータ（type, question_type, chunk_idx, generation_method）を付与 |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| qa_pairs | List[Dict] | 生成されたQ/Aペアリスト |

#### generate_comprehensive_qa_for_chunk()

| 項目 | 内容 |
|------|------|
| **機能** | 単一チャンクに対して包括的なQ/Aを生成（テンプレートベース） |
| **定義** | `a03_rag_qa_coverage_improved.py:564-740` |

| INPUT | 型 | 説明 |
|-------|---|------|
| chunk_text | str | チャンクのテキスト |
| chunk_idx | int | チャンクのインデックス |
| qa_per_chunk | int | チャンクあたりのQ/A数（デフォルト: 5） |
| lang | str | 言語コード（"en", "ja", "auto"） |

| PROCESS |
|---------|
| 1. 言語の自動判定（英語/日本語インジケータ比較） |
| 2. チャンクを文に分割 |
| 3. 戦略1: チャンク全体の包括的Q/A生成（comprehensive） |
| 4. 戦略2: 文ごとの詳細Q/A生成（factual_detailed） |
| 5. 戦略3: 前文との文脈Q/A生成（contextual） |
| 6. 戦略4: キーワード抽出型Q/A生成（keyword_based） |
| 7. 戦略5: テーマQ/A生成（thematic） |
| 8. 上限数まで切り詰め |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| qas | List[Dict] | 生成されたQ/Aペアリスト（question, answer, type, chunk_idx等） |

#### generate_advanced_qa_for_chunk()

| 項目 | 内容 |
|------|------|
| **機能** | 高度なQ/A生成システム（品質スコアリング機能付き） |
| **定義** | `a03_rag_qa_coverage_improved.py:501-562` |

| INPUT | 型 | 説明 |
|-------|---|------|
| chunk_text | str | チャンクのテキスト |
| chunk_idx | int | チャンクのインデックス |
| qa_per_chunk | int | チャンクあたりのQ/A数（デフォルト: 5） |
| lang | str | 言語コード（"en", "ja", "auto"） |
| dataset_type | str | データセットタイプ（ドメイン適応用） |

| PROCESS |
|---------|
| 1. チャンク複雑度分析（analyze_chunk_complexity） |
| 2. ドメイン適応戦略の取得 |
| 3. 複雑度に基づく質問タイプ分布の決定 |
| 4. コンテキスト強化型Q/A生成 |
| 5. マルチホップ推論型Q/A生成（medium/high複雑度のみ） |
| 6. 階層化質問タイプに基づくQ/A生成 |
| 7. 品質スコアリング |
| 8. 品質スコアでソートして上位を返却 |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| scored_qas | List[Dict] | 品質スコア付きQ/Aペアリスト |

---

### 11.5 メイン処理系

#### process_with_improved_methods()

| 項目 | 内容 |
|------|------|
| **機能** | 改良版：80%カバレッジを達成するためのQ/A生成 |
| **定義** | `a03_rag_qa_coverage_improved.py:852-969` |

| INPUT | 型 | 説明 |
|-------|---|------|
| document_text | str | 処理対象テキスト |
| methods | List[str] | 使用する手法のリスト（"rule", "template", "llm"） |
| model | str | 使用するモデル（デフォルト: "gpt-4o-mini"） |
| qa_per_chunk | int | チャンクあたりのQ/A数（デフォルト: 4） |
| max_chunks | int | 処理する最大チャンク数（デフォルト: 300） |
| lang | str | 言語コード（"en", "ja", "auto"） |

| PROCESS |
|---------|
| 1. SemanticCoverageを初期化（gemini-embedding-001） |
| 2. セマンティックチャンク分割（段落優先モード） |
| 3. 大規模データの場合は均等サンプリング |
| 4. "rule"/"template"メソッド: generate_comprehensive_qa_for_chunk()実行 |
| 5. "llm"メソッド: generate_llm_qa()実行（先頭20チャンクのみ） |
| 6. 重複除去（質問先頭30文字ベース） |
| 7. カバレッジ不足時は追加生成 |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| unique_qas | List[Dict] | 重複除去済みQ/Aペアリスト |
| analyzer | SemanticCoverage | 分析器インスタンス |
| chunks | List[Dict] | チャンクリスト |

---

### 11.6 カバレッジ分析系

#### calculate_improved_coverage()

| 項目 | 内容 |
|------|------|
| **機能** | 改善されたカバレッジ計算（バッチ処理版） |
| **定義** | `a03_rag_qa_coverage_improved.py:743-849` |

| INPUT | 型 | 説明 |
|-------|---|------|
| chunks | List[Dict] | チャンクリスト |
| qa_pairs | List[Dict] | Q/Aペアリスト |
| analyzer | SemanticCoverage | SemanticCoverageインスタンス |
| threshold | float | カバレッジ判定閾値（デフォルト: 0.65） |

| PROCESS |
|---------|
| 1. チャンクの埋め込みを生成 |
| 2. Q/Aテキストを準備（質問1回 + 回答2回で結合） |
| 3. バッチサイズ（100）を考慮してQ/A埋め込みを生成 |
| 4. カバレッジ行列（チャンク×Q/A）を計算 |
| 5. 各チャンクの最大類似度を追跡 |
| 6. 閾値超過チャンクをカバー済みとしてマーク |
| 7. 統計情報（平均/最小/最大類似度）を計算 |
| 8. カバレッジ分布（高/中/低）を集計 |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| coverage_results | Dict | カバレッジ結果辞書 |
| max_similarities | List[float] | 各チャンクの最大類似度リスト |

---

### 11.7 結果保存系

#### save_results()

| 項目 | 内容 |
|------|------|
| **機能** | 結果をファイルに保存 |
| **定義** | `a03_rag_qa_coverage_improved.py:972-1035` |

| INPUT | 型 | 説明 |
|-------|---|------|
| qa_pairs | List[Dict] | Q/Aペアリスト |
| coverage_results | Optional[Dict] | カバレッジ分析結果（オプション） |
| dataset_type | str | データセットタイプ（デフォルト: "custom"） |
| output_dir | str | 出力ディレクトリ（デフォルト: "qa_output"） |

| PROCESS |
|---------|
| 1. 出力ディレクトリ（qa_output/a03/）を作成 |
| 2. タイムスタンプを生成 |
| 3. Q/AペアをJSON形式で保存 |
| 4. Q/AペアをCSV形式（全カラム）で保存 |
| 5. Q/Aペアを統一フォーマットCSV（question/answerのみ）で保存 |
| 6. カバレッジ結果があればJSON保存 |
| 7. サマリー情報をJSON保存 |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| saved_files | Dict[str, str] | 保存されたファイルパスの辞書（qa_json, qa_csv, coverage, summary） |

---

### 11.8 エントリポイント

#### main()

| 項目 | 内容 |
|------|------|
| **機能** | メイン実行関数 |
| **定義** | `a03_rag_qa_coverage_improved.py:1038-1177` |

| INPUT | 型 | 説明 |
|-------|---|------|
| コマンドライン引数 | argparse | --input, --dataset, --max-docs, --methods, --model, --output, --analyze-coverage, --coverage-threshold, --qa-per-chunk, --max-chunks, --demo |

| PROCESS |
|---------|
| 1. ArgumentParserでコマンドライン引数を解析 |
| 2. 環境チェック（GOOGLE_API_KEY） |
| 3. データ読み込み（load_input_data or デモテキスト） |
| 4. データセットから言語情報を取得 |
| 5. Q/A生成処理（process_with_improved_methods） |
| 6. カバレッジ分析（calculate_improved_coverage）※オプション |
| 7. カバレッジ警告（70%未満の場合） |
| 8. 結果保存（save_results） |
| 9. Q/Aタイプ統計を表示 |

| OUTPUT | 型 | 説明 |
|--------|---|------|
| なし | | ファイル出力とコンソール出力 |

---

## 12. コマンドラインオプション

### 12.1 全オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `--input` | str | 必須 | 入力ファイルパス |
| `--dataset` | str | 必須 | データセットタイプ（cc_news, japanese_text, wikipedia_ja, livedoor） |
| `--max-docs` | int | None | 処理する最大文書数 |
| `--methods` | str[] | ['rule', 'template'] | 使用する手法 (llmを追加可能) |
| `--model` | str | **`gemini-2.0-flash`** | 使用するGeminiモデル |
| `--output` | str | qa_output | 出力ディレクトリ |
| `--analyze-coverage` | flag | False | カバレッジ分析を実行 |
| `--coverage-threshold` | float | 0.65 | カバレッジ判定閾値 |
| `--qa-per-chunk` | int | 4 | チャンクあたりのQ/A数 |
| `--max-chunks` | int | 300 | 処理する最大チャンク数 |
| `--demo` | flag | False | デモモード |

### 12.2 手法オプションの組み合わせ

```bash
# デフォルト（ルール+テンプレート）- 高速・低コスト
--methods rule template

# LLM追加（高品質化）- 中速・中コスト
--methods rule template llm
```

---

## 13. 実行方法

### 13.1 テスト実行（小規模）

```bash
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_cc_news.csv \
  --dataset cc_news \
  --max-docs 150 \
  --qa-per-chunk 4 \
  --max-chunks 609 \
  --analyze-coverage \
  --coverage-threshold 0.60 \
  --model gemini-2.0-flash
```

### 13.2 推奨実行（中規模）

```bash
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_cc_news.csv \
  --dataset cc_news \
  --qa-per-chunk 10 \
  --max-chunks 2000 \
  --analyze-coverage \
  --coverage-threshold 0.60 \
  --methods rule template llm
```

### 13.3 大規模実行

```bash
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_cc_news.csv \
  --dataset cc_news \
  --qa-per-chunk 8 \
  --max-chunks 5000 \
  --analyze-coverage \
  --coverage-threshold 0.60
```

### 13.4 日本語データセット実行

```bash
# Wikipedia日本語版
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_wikipedia_ja.csv \
  --dataset wikipedia_ja \
  --qa-per-chunk 6 \
  --max-chunks 1000 \
  --analyze-coverage

# Livedoorニュース
python a03_rag_qa_coverage_improved.py \
  --input OUTPUT/preprocessed_livedoor.csv \
  --dataset livedoor \
  --qa-per-chunk 6 \
  --max-chunks 1000 \
  --analyze-coverage
```

### 13.5 デモモード

```bash
python a03_rag_qa_coverage_improved.py --demo
```

### 13.6 実行時間の見積もり

| 設定 | 文書数 | チャンク数 | Q/A数 | 実行時間 | コスト |
|------|--------|----------|-------|---------|--------|
| テスト | 150 | 609 | 7,308 | 2分 | 低 |
| 推奨 | 自動 | 2,000 | 20,000 | 8-10分 | 低〜中 |
| 中規模 | 1,000 | 2,400 | 24,000 | 10-12分 | 低〜中 |
| 全文書 | 7,499 | 18,000 | 144,000 | 60-90分 | 中 |

※ Gemini 2.0 Flash は非常に安価かつ高速なため、OpenAIモデルと比較してコストパフォーマンスが良いです。

---

## 14. 出力ファイル

### 14.1 出力ディレクトリ構造

```
qa_output/
├── a03/
│   ├── qa_pairs_{dataset}_{timestamp}.json    # Q/Aペア（JSON）
│   ├── qa_pairs_{dataset}_{timestamp}.csv     # Q/Aペア（CSV全カラム）
│   ├── coverage_{dataset}_{timestamp}.json    # カバレッジ分析結果
│   └── summary_{dataset}_{timestamp}.json     # サマリー情報
│
└── a03_qa_pairs_{dataset}.csv                 # 統一フォーマット（question/answerのみ）
```

### 14.2 Q/Aペア JSON形式

```json
[
  {
    "question": "このセクションにはどのような情報が含まれていますか？",
    "answer": "テキスト内容...",
    "type": "comprehensive",
    "chunk_idx": 0,
    "coverage_strategy": "full_chunk"
  },
  {
    "question": "「キーワード」について何が述べられていますか？",
    "answer": "回答内容...",
    "type": "keyword_based",
    "chunk_idx": 0,
    "keyword": "キーワード"
  }
]
```

### 14.3 カバレッジ結果 JSON形式

```json
{
  "coverage_rate": 0.952,
  "covered_chunks": 1608,
  "total_chunks": 1689,
  "threshold": 0.6,
  "avg_max_similarity": 0.785,
  "min_max_similarity": 0.432,
  "max_max_similarity": 0.956,
  "uncovered_chunks": [45, 102, 345, ...],
  "coverage_distribution": {
    "high_coverage": 1423,
    "medium_coverage": 234,
    "low_coverage": 32
  }
}
```

### 14.4 サマリー JSON形式

```json
{
  "dataset_type": "cc_news",
  "generated_at": "20251108_010658",
  "total_qa_pairs": 4278,
  "coverage_rate": 0.952,
  "coverage_details": {
    "high_coverage": 1423,
    "medium_coverage": 234,
    "low_coverage": 32
  },
  "files": {
    "qa_json": "qa_output/a03/qa_pairs_cc_news_20251108_010658.json",
    "qa_csv": "qa_output/a03/qa_pairs_cc_news_20251108_010658.csv",
    "coverage": "qa_output/a03/coverage_cc_news_20251108_010658.json",
    "summary": "qa_output/a03/summary_cc_news_20251108_010658.json"
  }
}
```

---

## 15. トラブルシューティング

### 15.1 APIキーエラー

**症状**: `GOOGLE_API_KEYが設定されていません`

**解決策**:
```bash
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

### 15.2 ファイルが見つからない

**症状**: `FileNotFoundError: 入力ファイルが見つかりません`

**解決策**:
```bash
ls OUTPUT/preprocessed_cc_news.csv
```

### 15.3 カバレッジが低い

**症状**: カバレッジ率が70%未満

**解決策**:
```bash
# Q/A数を増やす
python a03_rag_qa_coverage_improved.py --qa-per-chunk 10

# LLM手法を追加して多様性を高める
python a03_rag_qa_coverage_improved.py --methods rule template llm

# 閾値を下げる（必要に応じて）
python a03_rag_qa_coverage_improved.py --coverage-threshold 0.55
```

### 15.4 MeCabが利用できない

**症状**: `⚠️ MeCabが利用できません（正規表現モード）`

**解決策**: MeCabをインストールしてください。

```bash
# macOS
brew install mecab mecab-ipadic
pip install mecab-python3

# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
pip install mecab-python3
```

正規表現モードでも動作しますが、複合名詞の抽出精度が低下します。

### 15.5 メモリ不足

**症状**: 大規模データ処理時にメモリエラー

**解決策**:
```bash
# チャンク数を制限
python a03_rag_qa_coverage_improved.py --max-chunks 1000

# 文書数を制限
python a03_rag_qa_coverage_improved.py --max-docs 500
```

### 15.6 API制限エラー

**症状**: Gemini APIのレート制限エラー

**解決策**:
- バッチサイズは内部で100に制限済み
- それでも発生する場合は、--max-chunksを減らして分割実行

---

## 16. 付録: 品質改善機能一覧

### 16.1 実装済み機能

| 機能 | 説明 | 関連関数/定数 |
|------|------|--------------|
| 階層化質問タイプ | 3階層11タイプの質問分類 | `QUESTION_TYPES_HIERARCHY` |
| チャンク複雑度分析 | 専門用語密度・文長による分析 | `analyze_chunk_complexity()` |
| ドメイン適応戦略 | データセット別の最適化 | `DOMAIN_SPECIFIC_STRATEGIES` |
| MeCab/正規表現ハイブリッド | キーワード抽出の自動フォールバック | `KeywordExtractor` |
| バッチ埋め込み生成 | Gemini API制限を考慮したバッチ処理 | `calculate_improved_coverage()` |
| 3段階カバレッジ分布 | 高・中・低の詳細評価 | カバレッジ結果 |
| LLMハイブリッド生成 | Gemini 2.0 Flashによる高品質Q/A生成 | `generate_llm_qa()` |
| 5戦略テンプレート生成 | 包括的・詳細・文脈・キーワード・テーマ | `generate_comprehensive_qa_for_chunk()` |
| 重複除去 | 質問先頭30文字ベースの簡易重複チェック | `process_with_improved_methods()` |
| シングルトンパターン | KeywordExtractorの効率的な再利用 | `get_keyword_extractor()` |

### 16.2 品質指標の期待値

| 指標 | 期待値 | 条件 |
|------|--------|------|
| カバレッジ率 | 95%+ | 閾値0.60, qa_per_chunk≥4 |
| 平均最大類似度 | 0.78+ | 推奨設定時 |
| 平均品質スコア | 0.82 | LLM手法併用時 |
| 平均多様性スコア | 0.88 | 階層化質問タイプ使用時 |

### 16.3 今後の拡張候補

| 機能 | 説明 | 優先度 |
|------|------|--------|
| マルチホップ推論Q/A | 複数チャンクを跨ぐ推論質問 | 中 |
| コンテキスト強化Q/A | 前後文脈を考慮した質問 | 中 |
| 品質スコアリング | Q/A品質の自動評価 | 高 |
| 並列処理対応 | マルチスレッド/プロセス処理 | 低 |
| ストリーミング対応 | 大規模ファイルのストリーミング処理 | 低 |

---

**ドキュメント更新履歴**:
- 2025-12-18: v2.0 全面改修（方式・手法の位置づけ表追加、IPO形式関数リファレンス追加、チャンキング技術詳細追加）
- 2025-11-08: v1.0 初版作成（Gemini移行対応）