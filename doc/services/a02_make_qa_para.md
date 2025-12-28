### 問題点、課題：
- 関数の責務の混在: データ読み込み、チャンク作成、Q/A生成、カバレッジ分析、結果保存が全て1ファイル

# a02_make_qa_para.py - 標準並列処理Q/A生成システム (Celery/Redis版)

作成日: 2025-11-28 (最終更新: 2025-12-18)

## 目次

1. [概要](#1-概要)
2. [方式・手法の位置づけ](#2-方式手法の位置づけ)
3. [アーキテクチャ](#3-アーキテクチャ)
4. [チャンキング技術詳細](#4-チャンキング技術詳細)
5. [キーワード抽出・複雑度分析](#5-キーワード抽出複雑度分析)
6. [関数リファレンス（IPO形式）](#6-関数リファレンスipo形式)
7. [プロンプト設計](#7-プロンプト設計)
8. [Celery非同期並列処理](#8-celery非同期並列処理)
9. [カバレージ分析](#9-カバレージ分析)
10. [データセット設定](#10-データセット設定)
11. [コマンドラインオプション](#11-コマンドラインオプション)
12. [実行方法](#12-実行方法)
13. [出力ファイル](#13-出力ファイル)
14. [パフォーマンス](#14-パフォーマンス)
15. [トラブルシューティング](#15-トラブルシューティング)

---

## 1. 概要

### 1.1 目的と位置づけ

`a02_make_qa_para.py`は、**Celery + Redisを用いた分散タスクキュー方式**によるQ/Aペア自動生成システムです。
Gemini API (`gemini-2.0-flash`等) を使用し、各ドキュメントやチャンクを個別のタスクとして管理・実行します。

> **【推奨】**
> API呼び出し回数を削減し、単一サーバーで高速に処理したい場合は、**[a10_qa_optimized_hybrid_batch.py](a10_qa_optimized_hybrid_batch.md)** の使用を推奨します。
> 本スクリプト (`a02`) は、複数のサーバーにワーカーを配置して処理を水平スケールさせたい場合や、タスク単位の厳密な状態管理が必要な場合に適しています。

### 1.2 起動コマンド

```bash
# 基本実行（同期処理）
python a02_make_qa_para.py --dataset livedoor --model gemini-2.0-flash --max-docs 20

# Celery並列処理（推奨）
python a02_make_qa_para.py \
  --dataset cc_news \
  --use-celery \
  --celery-workers 8 \
  --batch-chunks 3 \
  --model gemini-2.0-flash
```

### 1.3 主要機能

| 機能 | 説明 |
|------|------|
| **セマンティック分割によるチャンク作成** | 段落境界を優先した意味的チャンク作成（`gemini-embedding-001`を使用） |
| **バッチ処理による並列Q/A生成** | Gemini API (`gemini-2.0-flash`等) を使用し、1-5チャンクを同時に処理 |
| **Celeryによる非同期並列処理** | 複数ワーカーでGemini APIへの呼び出しを非同期に実行 |
| **小チャンク自動統合による効率化** | 短すぎるチャンクを自動的に統合し、Q/A生成の効率と品質を向上 |
| **動的Q/A数決定ロジック** | チャンクのトークン数や文書位置に基づいて最適なQ/A生成数を動的に調整 |
| **多段階カバレージ分析** | `gemini-embedding-001`で評価（strict/standard/lenient） |
| **チャンク特性別カバレージ分析** | チャンクの長さや文書内の位置に応じたカバレージ分析 |
| **MeCab/正規表現キーワード抽出** | 複合名詞抽出による高精度キーワード抽出（MeCab優先、フォールバック対応） |

### 1.4 MeCab対応

| 言語 | MeCab利用可能時 | MeCab利用不可時 |
|------|----------------|----------------|
| 日本語（ja） | MeCabによる複合名詞抽出 | 正規表現にフォールバック |
| 英語（en） | 正規表現ベースの抽出 | 正規表現ベースの抽出 |

---

## 2. 方式・手法の位置づけ

### 2.1 Q/A生成手法の全体概要

本システムは複数の技術を組み合わせた**分散タスクキュー方式**を採用しています。

| カテゴリ | 手法 | 本システムでの適用 |
|---------|------|------------------|
| **文書分割** | セマンティックチャンキング | SemanticCoverageクラスによる段落優先分割 |
| **前処理** | キーワード抽出 | KeywordExtractorによるMeCab/正規表現抽出 |
| **前処理** | 複雑度分析 | トークン数・専門用語密度による動的調整 |
| **Q/A生成** | LLM生成 | Gemini APIによる構造化出力（Pydantic連携） |
| **Q/A生成** | 動的Q/A数決定 | チャンク位置・長さに基づく適応的生成 |
| **並列処理** | 分散タスクキュー | Celery/Redis による水平スケーリング |
| **品質評価** | 多段階カバレージ | strict/standard/lenient の3段階評価 |
| **品質評価** | チャンク特性分析 | 長さ別・位置別のカバレージ分析 |

### 2.2 他システムとの比較

| 項目 | **a02（Celery並列）** | a03（テンプレート版） | a10（ハイブリッドバッチ） |
|------|---------------------|---------------------|----------------------|
| **Q/A生成手法** | **LLM (`gemini-2.0-flash`)** | テンプレートのみ | ルールベース+LLM |
| **チャンキング** | **SemanticCoverage** | 固定長分割 | SemanticCoverage |
| **並列処理** | **Celery/Redis分散** | なし | なし |
| **スケーラビリティ** | **水平スケール可能** | 低い | 中程度 |
| **API呼び出し** | 多い（チャンク単位） | 最小 | 中程度（バッチ化） |
| **処理時間** | 23分（8ワーカー） | 60-90分 | 61分 |
| **コスト** | 高い | 極めて低い | 中程度 |
| **カバレージ** | 90-95% | 95%+ | 95%（品質モード） |
| **適用シナリオ** | **マルチサーバー分散** | 低コスト処理 | 単一サーバー高速処理 |

### 2.3 技術的特徴の比較表

| 技術要素 | 詳細 | 利点 | 欠点 |
|---------|------|------|------|
| **Celery/Redis** | 分散タスクキュー | 水平スケール、フォールトトレラント | インフラ構築が必要 |
| **KeywordExtractor** | MeCab/正規表現統合 | 高精度キーワード抽出 | MeCab依存（オプション） |
| **複雑度分析** | 専門用語密度計算 | 動的Q/A数最適化 | 処理オーバーヘッド |
| **多段階カバレージ** | 3段階閾値評価 | 詳細な品質評価 | 評価時間増加 |
| **構造化出力** | Pydantic + Gemini | 型安全なQ/A生成 | スキーマ定義が必要 |

### 2.4 処理モード比較

| モード | API呼び出し | 実行時間 | 効率化率 | 推奨用途 |
|--------|------------|---------|---------|---------|
| 同期処理 | 1800回 | 180分 | 1.0x | 小規模テスト |
| **Celery並列** | 1800回 | **23分** | **7.8x** | 中規模処理 |
| ハイブリッド | 600回 | 8分 | 22.5x | 大規模処理 |

---

## 3. アーキテクチャ

### 3.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                   a02_make_qa_para.py                           │
├─────────────────────────────────────────────────────────────────┤
│  [1] データ読み込み                                              │
│      load_preprocessed_data() / load_uploaded_file()            │
│                              │                                  │
│                              ▼                                  │
│  [2] チャンク作成 + 前処理                                       │
│      create_document_chunks() → merge_small_chunks()            │
│      KeywordExtractor → analyze_chunk_complexity()              │
│                              │                                  │
│                              ▼                                  │
│  [3] Q/A生成（並列処理）                                         │
│      ├─ 同期モード: generate_qa_for_dataset()                   │
│      └─ Celeryモード: submit_unified_qa_generation()            │
│                              │                                  │
│                              ▼                                  │
│  [4] カバレージ分析                                              │
│      analyze_coverage() → multi_threshold_coverage()            │
│                              │                                  │
│                              ▼                                  │
│  [5] 結果保存                                                    │
│      save_results() → qa_output/a02/                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 処理フロー詳細図

```
入力データ (CSV/JSON/TXT)
      │
      ▼
┌──────────────────┐
│ データ読み込み    │ load_preprocessed_data() / load_uploaded_file()
│ (pandas)         │
└──────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────┐
│              チャンク作成・前処理                       │
├──────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────┐  │
│  │ SemanticCoverage (helper_rag_qa.py)           │  │
│  │  ├─ create_semantic_chunks()                  │  │
│  │  ├─ 段落優先分割                              │  │
│  │  └─ MeCab/正規表現文分割                      │  │
│  └────────────────────────────────────────────────┘  │
│                       │                               │
│                       ▼                               │
│  ┌────────────────────────────────────────────────┐  │
│  │ 小チャンク統合 merge_small_chunks()            │  │
│  │  min_tokens: 150, max_tokens: 400             │  │
│  └────────────────────────────────────────────────┘  │
│                       │                               │
│                       ▼                               │
│  ┌────────────────────────────────────────────────┐  │
│  │ 複雑度分析・キーワード抽出                     │  │
│  │  ├─ KeywordExtractor (MeCab/正規表現)         │  │
│  │  └─ analyze_chunk_complexity()                │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────┐
│              Q/A生成（並列処理選択）                   │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ 同期モード       │    │ Celeryモード            │  │
│  │ generate_qa_    │    │ submit_unified_qa_      │  │
│  │ for_dataset()   │    │ generation()            │  │
│  │                 │    │         │               │  │
│  │ ループ処理      │    │         ▼               │  │
│  │ (リトライ付き)  │    │ ┌─────────────────────┐ │  │
│  │                 │    │ │ Redis Task Queue    │ │  │
│  │                 │    │ └─────────────────────┘ │  │
│  │                 │    │         │               │  │
│  │                 │    │         ▼               │  │
│  │                 │    │ ┌─────────────────────┐ │  │
│  │                 │    │ │ Celery Workers (8)  │ │  │
│  │                 │    │ │  ├─ Worker 1        │ │  │
│  │                 │    │ │  ├─ Worker 2        │ │  │
│  │                 │    │ │  └─ ...             │ │  │
│  │                 │    │ └─────────────────────┘ │  │
│  └─────────────────┘    └─────────────────────────┘  │
│                                                       │
│  共通: Gemini API (gemini-2.0-flash)                 │
│        構造化出力 (QAPairsResponse Pydanticモデル)   │
└──────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────┐
│              カバレージ分析                            │
├──────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────┐  │
│  │ analyze_coverage()                             │  │
│  │  ├─ 埋め込み生成 (gemini-embedding-001)       │  │
│  │  ├─ コサイン類似度計算                        │  │
│  │  ├─ multi_threshold_coverage()                │  │
│  │  │   ├─ strict (0.80)                         │  │
│  │  │   ├─ standard (0.70)                       │  │
│  │  │   └─ lenient (0.60)                        │  │
│  │  └─ analyze_chunk_characteristics_coverage()  │  │
│  │      ├─ by_length (short/medium/long)         │  │
│  │      └─ by_position (beginning/middle/end)    │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────┐
│ 結果保存         │ save_results()
│ (JSON/CSV)       │
└──────────────────┘
```

### 3.3 依存モジュール

```python
from helper_llm import create_llm_client, LLMClient
from helper_rag_qa import SemanticCoverage
from models import QAPairsResponse
from config import DATASET_CONFIGS, QAGenerationConfig
from celery_tasks import submit_unified_qa_generation, collect_results
import tiktoken
import MeCab  # オプション
```

---

## 4. チャンキング技術詳細

### 4.1 セマンティックチャンキングの概要

本システムは`helper_rag_qa.SemanticCoverage`を使用した**セマンティックチャンキング**を採用しています。

```
┌────────────────────────────────────────────────────────────┐
│                セマンティックチャンキング階層               │
├────────────────────────────────────────────────────────────┤
│  優先度1: 段落境界 (\n\n)                                  │
│    └─ 著者が意図した最も重要なセマンティック境界            │
│                                                            │
│  優先度2: 文境界                                            │
│    ├─ 日本語: MeCab形態素解析（句点・疑問符・感嘆符）      │
│    └─ 英語: 正規表現 (?<=[。．.!?])\s*                     │
│                                                            │
│  優先度3: トークン強制分割（最終手段）                      │
│    └─ 上限超過時にtiktokenで強制分割                       │
└────────────────────────────────────────────────────────────┘
```

### 4.2 チャンキングパラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `max_tokens` | 200 (chunk_size) | チャンクの最大トークン数 |
| `min_tokens` | 50 | チャンクの最小トークン数 |
| `prefer_paragraphs` | True | 段落ベースの分割を優先 |

### 4.3 小チャンク統合

`merge_small_chunks()` 関数により、短すぎるチャンクを自動的に統合します。

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `min_tokens` | 150 | このトークン数未満のチャンクは統合対象 |
| `max_tokens` | 400 | 統合後の最大トークン数 |

```
統合前: 100個のチャンク
        │
        ▼
┌─────────────────────────────────────────────┐
│ merge_small_chunks()                         │
│  ├─ 同一文書内のチャンクのみ統合             │
│  ├─ min_tokens (150) 未満を統合対象          │
│  └─ max_tokens (400) 以内で統合              │
└─────────────────────────────────────────────┘
        │
        ▼
統合後: 85個のチャンク（15%削減）
```

### 4.4 チャンクタイプ一覧

| タイプ | 説明 | 生成条件 |
|--------|------|----------|
| `paragraph` | 段落単位チャンク | 段落がmax_tokens以下 |
| `sentence_group` | 文グループチャンク | 複数文の結合 |
| `forced_split` | 強制分割チャンク | 単一文がmax_tokens超過 |
| `merged` | 統合チャンク | merge_small_chunks()により統合 |

---

## 5. キーワード抽出・複雑度分析

### 5.1 KeywordExtractorクラス

MeCabと正規表現を統合したキーワード抽出クラスです。

```python
class KeywordExtractor:
    """
    MeCabと正規表現を統合したキーワード抽出クラス
    MeCabが利用可能な場合は複合名詞抽出を優先し、
    利用不可の場合は正規表現版に自動フォールバック
    """
```

#### 抽出方式

| 方式 | 優先度 | 特徴 |
|------|--------|------|
| MeCab複合名詞抽出 | 高 | 形態素解析による高精度抽出 |
| 正規表現抽出 | フォールバック | カタカナ語、漢字複合語、英数字を抽出 |

#### ストップワード

```python
stopwords = {
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
    'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる',
    'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と',
    'の', 'から', 'まで', '等', 'など', 'よる', 'おく', 'くる'
}
```

### 5.2 複雑度分析

`analyze_chunk_complexity()` 関数により、チャンクの複雑度を分析します。

| 指標 | 計算方法 | 用途 |
|------|----------|------|
| `complexity_level` | concept_density + avg_sentence_length | Q/A数の動的決定 |
| `technical_terms` | 正規表現パターンマッチ | プロンプト強化 |
| `avg_sentence_length` | tokens / sentences | 文の複雑度評価 |
| `concept_density` | technical_terms / tokens * 100 | 概念密度評価 |

#### 複雑度レベル判定

| レベル | 条件 | Q/A数への影響 |
|--------|------|--------------|
| `high` | concept_density > 5 or avg_sentence_length > 30 | +2-3 |
| `medium` | concept_density > 2 or avg_sentence_length > 20 | +1-2 |
| `low` | それ以外 | 基本値 |

### 5.3 動的Q/A数決定

`determine_qa_count()` 関数により、チャンクに最適なQ/A数を動的に決定します。

| トークン数 | Q/A数 | 備考 |
|-----------|-------|------|
| < 50 | 2 | 短いチャンクでも最低2個 |
| 50-99 | 3 | Shortチャンク強化 |
| 100-199 | base + 1 | Mediumチャンク |
| 200-299 | base + 2 | Longチャンク |
| >= 300 | base + 3 | 超長文 |

**位置バイアス補正**: 6番目以降のチャンクは+1（文書後半のカバレージ向上）

---

## 6. 関数リファレンス（IPO形式）

### 6.1 データ読み込み関数

#### `load_preprocessed_data(dataset_type)`

| 項目 | 内容 |
|------|------|
| **概要** | preprocessedデータセットをCSVから読み込み |
| **INPUT** | `dataset_type`: str - データセットタイプ（cc_news, japanese_text, wikipedia_ja, livedoor） |
| **PROCESS** | 1. DATASET_CONFIGSからファイルパス取得<br>2. タイムスタンプ付きファイルの自動検索<br>3. pandas.read_csv()でCSV読み込み<br>4. 空テキスト除外 |
| **OUTPUT** | `pd.DataFrame` - テキストデータを含むDataFrame |

#### `load_uploaded_file(file_path)`

| 項目 | 内容 |
|------|------|
| **概要** | ローカルファイルを読み込み（CSV/TXT/JSON/JSONL対応） |
| **INPUT** | `file_path`: str - ファイルパス |
| **PROCESS** | 1. ファイル形式判定<br>2. 形式別読み込み（CSV/TXT/JSON/JSONL）<br>3. Combined_Textカラム自動生成<br>4. 空データ除外 |
| **OUTPUT** | `pd.DataFrame` - Combined_Textカラムを含むDataFrame |

#### `load_local_qa_file(file_path)`

| 項目 | 内容 |
|------|------|
| **概要** | ローカルQ/A CSVファイルを読み込み |
| **INPUT** | `file_path`: str - CSVファイルパス |
| **PROCESS** | 1. question/answerカラム検索<br>2. カラム名統一<br>3. 空データ・重複除去 |
| **OUTPUT** | `pd.DataFrame` - question, answerカラムを含むDataFrame |

### 6.2 チャンキング関数

#### `create_semantic_chunks(text, lang, max_tokens, chunk_id_prefix)`

| 項目 | 内容 |
|------|------|
| **概要** | セマンティック分割によるチャンク作成（段落優先） |
| **INPUT** | `text`: str - 分割対象テキスト<br>`lang`: str - 言語（"ja"/"en"）<br>`max_tokens`: int - 最大トークン数（デフォルト: 200）<br>`chunk_id_prefix`: str - チャンクIDプレフィックス |
| **PROCESS** | 1. SemanticCoverage.create_semantic_chunks()呼び出し<br>2. 段落優先モードで分割<br>3. 出力形式をa02形式に変換 |
| **OUTPUT** | `List[Dict]` - チャンクリスト（id, text, tokens, type, sentences） |

#### `create_document_chunks(df, dataset_type, max_docs, config)`

| 項目 | 内容 |
|------|------|
| **概要** | DataFrameから文書チャンクを作成 |
| **INPUT** | `df`: pd.DataFrame - 入力データ<br>`dataset_type`: str - データセットタイプ<br>`max_docs`: Optional[int] - 最大文書数<br>`config`: Optional[Dict] - データセット設定 |
| **PROCESS** | 1. 文書ごとにcreate_semantic_chunks()実行<br>2. メタデータ追加（doc_id, doc_idx, chunk_idx, dataset_type）<br>3. 進捗ログ出力（10件ごと） |
| **OUTPUT** | `List[Dict]` - 全チャンクリスト |

#### `merge_small_chunks(chunks, min_tokens, max_tokens)`

| 項目 | 内容 |
|------|------|
| **概要** | 小さいチャンクを統合して適切なサイズにする |
| **INPUT** | `chunks`: List[Dict] - チャンクリスト<br>`min_tokens`: int - 統合対象の最小トークン数（デフォルト: 150）<br>`max_tokens`: int - 統合後の最大トークン数（デフォルト: 400） |
| **PROCESS** | 1. 各チャンクのトークン数計算<br>2. min_tokens未満のチャンクを統合候補に<br>3. 同一文書内でmax_tokens以内で統合<br>4. mergedフラグと統合元リスト付与 |
| **OUTPUT** | `List[Dict]` - 統合後チャンクリスト |

### 6.3 キーワード抽出・複雑度分析関数

#### `KeywordExtractor.extract(text, top_n)`

| 項目 | 内容 |
|------|------|
| **概要** | テキストからキーワードを抽出（自動フォールバック対応） |
| **INPUT** | `text`: str - 分析対象テキスト<br>`top_n`: int - 抽出するキーワード数（デフォルト: 5） |
| **PROCESS** | 1. MeCab利用可能かチェック<br>2. MeCab: 複合名詞抽出<br>3. フォールバック: 正規表現抽出<br>4. ストップワード除外・頻度カウント |
| **OUTPUT** | `List[str]` - キーワードリスト |

#### `analyze_chunk_complexity(chunk_text, lang)`

| 項目 | 内容 |
|------|------|
| **概要** | チャンクの複雑度を分析 |
| **INPUT** | `chunk_text`: str - 分析対象テキスト<br>`lang`: str - 言語 |
| **PROCESS** | 1. 文分割・トークン化<br>2. 専門用語検出（正規表現）<br>3. 平均文長・概念密度計算<br>4. 複雑度レベル判定 |
| **OUTPUT** | `Dict` - complexity_level, technical_terms, avg_sentence_length, concept_density, sentence_count, token_count |

#### `extract_key_concepts(chunk_text, lang, top_n)`

| 項目 | 内容 |
|------|------|
| **概要** | チャンクから主要概念を抽出 |
| **INPUT** | `chunk_text`: str - テキスト<br>`lang`: str - 言語<br>`top_n`: int - 抽出する概念数（デフォルト: 5） |
| **PROCESS** | 1. KeywordExtractor.extract()でキーワード抽出<br>2. analyze_chunk_complexity()で専門用語取得<br>3. 統合・重複除去 |
| **OUTPUT** | `List[str]` - 主要概念リスト |

### 6.4 Q/A生成関数

#### `determine_qa_count(chunk, config)`

| 項目 | 内容 |
|------|------|
| **概要** | チャンクに最適なQ/A数を動的決定 |
| **INPUT** | `chunk`: Dict - チャンクデータ<br>`config`: Dict - データセット設定 |
| **PROCESS** | 1. トークン数計算<br>2. トークン数に基づく基本Q/A数決定<br>3. 位置バイアス補正（6番目以降+1） |
| **OUTPUT** | `int` - Q/Aペア数（上限8） |

#### `generate_qa_pairs_for_batch(chunks, config, model, client)`

| 項目 | 内容 |
|------|------|
| **概要** | 複数チャンクから一度にQ/Aペアを生成（バッチ処理） |
| **INPUT** | `chunks`: List[Dict] - チャンクリスト（1-5個）<br>`config`: Dict - データセット設定<br>`model`: str - モデル名（デフォルト: gemini-2.0-flash）<br>`client`: Optional[LLMClient] - LLMクライアント |
| **PROCESS** | 1. 単一チャンクは個別処理に委譲<br>2. 複数チャンクを結合してプロンプト構築<br>3. Gemini API構造化出力で生成<br>4. 生成Q/Aを各チャンクに分配<br>5. エラー時は個別処理にフォールバック |
| **OUTPUT** | `List[Dict]` - Q/Aペアリスト |

#### `generate_qa_pairs_for_chunk(chunk, config, model, client)`

| 項目 | 内容 |
|------|------|
| **概要** | 単一チャンクからQ/Aペアを生成 |
| **INPUT** | `chunk`: Dict - チャンクデータ<br>`config`: Dict - データセット設定<br>`model`: str - モデル名<br>`client`: Optional[LLMClient] - LLMクライアント |
| **PROCESS** | 1. determine_qa_count()でQ/A数決定<br>2. 言語別プロンプト構築<br>3. テキスト短縮（2000文字上限）<br>4. Gemini API構造化出力で生成 |
| **OUTPUT** | `List[Dict]` - Q/Aペアリスト |

#### `generate_qa_for_dataset(chunks, dataset_type, model, chunk_batch_size, merge_chunks, min_tokens, max_tokens, config)`

| 項目 | 内容 |
|------|------|
| **概要** | データセット全体のQ/Aペア生成 |
| **INPUT** | `chunks`: List[Dict] - チャンクリスト<br>`dataset_type`: str - データセットタイプ<br>`model`: str - モデル名<br>`chunk_batch_size`: int - バッチサイズ（1-5）<br>`merge_chunks`: bool - チャンク統合フラグ<br>`min_tokens`: int - 統合最小トークン数<br>`max_tokens`: int - 統合最大トークン数<br>`config`: Optional[Dict] - データセット設定 |
| **PROCESS** | 1. チャンク統合（オプション）<br>2. バッチ処理ループ<br>3. リトライ機能（最大3回）<br>4. 個別処理フォールバック<br>5. API制限対策（0.2秒待機） |
| **OUTPUT** | `List[Dict]` - 全Q/Aペアリスト |

### 6.5 Celery関連関数

#### `check_celery_workers(required_workers)`

| 項目 | 内容 |
|------|------|
| **概要** | Celeryワーカーの状態を確認（リトライ機能付き） |
| **INPUT** | `required_workers`: int - 必要なワーカー数（デフォルト: 8） |
| **PROCESS** | 1. celery_tasksインポート<br>2. ワーカーstats取得（最大3回リトライ）<br>3. プールサイズからワーカー数算出<br>4. 不足時は警告出力 |
| **OUTPUT** | `bool` - ワーカー正常稼働時True |

### 6.6 カバレージ分析関数

#### `get_optimal_thresholds(dataset_type)`

| 項目 | 内容 |
|------|------|
| **概要** | データセット別の最適閾値を取得 |
| **INPUT** | `dataset_type`: str - データセットタイプ |
| **PROCESS** | OPTIMAL_THRESHOLDSから閾値取得 |
| **OUTPUT** | `Dict[str, float]` - {strict, standard, lenient} |

#### `multi_threshold_coverage(coverage_matrix, chunks, qa_pairs, thresholds)`

| 項目 | 内容 |
|------|------|
| **概要** | 複数閾値でカバレージを評価 |
| **INPUT** | `coverage_matrix`: np.ndarray - カバレージ行列<br>`chunks`: List[Dict] - チャンクリスト<br>`qa_pairs`: List[Dict] - Q/Aペアリスト<br>`thresholds`: Dict[str, float] - 閾値辞書 |
| **PROCESS** | 1. 各閾値でカバー判定<br>2. 未カバーチャンク特定<br>3. gap（閾値との差）計算 |
| **OUTPUT** | `Dict` - 各レベルのカバレージ結果 |

#### `analyze_chunk_characteristics_coverage(chunks, coverage_matrix, qa_pairs, threshold)`

| 項目 | 内容 |
|------|------|
| **概要** | チャンク特性別のカバレージ分析 |
| **INPUT** | `chunks`: List[Dict] - チャンクリスト<br>`coverage_matrix`: np.ndarray - カバレージ行列<br>`qa_pairs`: List[Dict] - Q/Aペアリスト<br>`threshold`: float - 判定閾値 |
| **PROCESS** | 1. 長さ別分析（short/medium/long）<br>2. 位置別分析（beginning/middle/end）<br>3. インサイト生成 |
| **OUTPUT** | `Dict` - by_length, by_position, summary |

#### `analyze_coverage(chunks, qa_pairs, dataset_type, custom_threshold)`

| 項目 | 内容 |
|------|------|
| **概要** | 生成Q/Aペアのカバレージを分析（メイン関数） |
| **INPUT** | `chunks`: List[Dict] - チャンクリスト<br>`qa_pairs`: List[Dict] - Q/Aペアリスト<br>`dataset_type`: str - データセットタイプ<br>`custom_threshold`: Optional[float] - カスタム閾値 |
| **PROCESS** | 1. チャンク埋め込み生成<br>2. Q/Aペア埋め込み生成（バッチAPI）<br>3. カバレージ行列計算<br>4. multi_threshold_coverage()実行<br>5. analyze_chunk_characteristics_coverage()実行 |
| **OUTPUT** | `Dict` - coverage_rate, multi_threshold, chunk_analysis 等 |

### 6.7 結果保存関数

#### `save_results(qa_pairs, coverage_results, dataset_type, output_dir)`

| 項目 | 内容 |
|------|------|
| **概要** | 結果をファイルに保存 |
| **INPUT** | `qa_pairs`: List[Dict] - Q/Aペアリスト<br>`coverage_results`: Dict - カバレージ分析結果<br>`dataset_type`: str - データセットタイプ<br>`output_dir`: str - 出力ディレクトリ |
| **PROCESS** | 1. 出力ディレクトリ作成<br>2. Q/Aペア保存（JSON, CSV全カラム, CSV統一フォーマット）<br>3. カバレージ結果保存（JSON）<br>4. サマリー情報保存（JSON） |
| **OUTPUT** | `Dict[str, str]` - 保存ファイルパス |

### 6.8 メイン関数

#### `main()`

| 項目 | 内容 |
|------|------|
| **概要** | コマンドライン引数を解析しメイン処理を実行 |
| **INPUT** | コマンドライン引数（argparse） |
| **PROCESS** | 1. 引数パース・検証<br>2. APIキー確認<br>3. データ読み込み<br>4. Celeryワーカー確認（use-celery時）<br>5. チャンク作成<br>6. Q/A生成（同期/Celery）<br>7. カバレージ分析（オプション）<br>8. 結果保存<br>9. 統計情報表示 |
| **OUTPUT** | なし（ファイル出力・コンソール表示） |

---

## 7. プロンプト設計

### 7.1 2段階プロンプト構造

| 段階 | 役割 | 内容 |
|------|------|------|
| システムプロンプト | 役割・ルール定義 | Q/A生成専門家としての役割、生成ルール |
| ユーザープロンプト | タスク・データ提示 | テキスト、Q/A数、質問タイプ、出力形式 |

### 7.2 システムプロンプト

**日本語版:**
```
あなたは教育コンテンツ作成の専門家です。
与えられた日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成
```

**英語版:**
```
You are an expert in educational content creation.
Generate high-quality Q&A pairs from the given English text.

Generation rules:
1. Questions should be clear and specific
2. Answers should be concise and accurate (1-2 sentences)
3. Stay faithful to the text content
4. Create questions from diverse perspectives
```

### 7.3 質問タイプ

| タイプ | 日本語 | 英語 |
|--------|--------|------|
| `fact` | 事実確認型（〜は何ですか？） | Factual questions (What is...?) |
| `reason` | 理由説明型（なぜ〜ですか？） | Explanatory questions (Why...?) |
| `comparison` | 比較型（〜と〜の違いは？） | Comparative questions (What's the difference...?) |
| `application` | 応用型（〜はどのように活用されますか？） | Application questions (How is... used?) |

### 7.4 JSON出力フォーマット

```json
{
  "qa_pairs": [
    {
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }
  ]
}
```

---

## 8. Celery非同期並列処理

### 8.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                      Celery/Redis 構成                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ a02_make_    │    │    Redis     │    │ Celery Workers   │  │
│  │ qa_para.py   │───▶│  (Broker)    │───▶│ (8ワーカー推奨)  │  │
│  │ (Producer)   │    │              │    │                  │  │
│  └──────────────┘    └──────────────┘    │  ┌────────────┐  │  │
│         │                    │           │  │ Worker 1   │  │  │
│         │                    │           │  ├────────────┤  │  │
│         │                    │           │  │ Worker 2   │  │  │
│         │                    │           │  ├────────────┤  │  │
│         │                    │           │  │ ...        │  │  │
│         │                    │           │  ├────────────┤  │  │
│         │                    │           │  │ Worker 8   │  │  │
│         │                    │           │  └────────────┘  │  │
│         │                    │           └──────────────────┘  │
│         │                    │                    │             │
│         │                    ▼                    │             │
│         │           ┌──────────────┐              │             │
│         └──────────│    Redis     │◀─────────────┘             │
│                     │  (Backend)   │                            │
│                     └──────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 ワーカー管理コマンド

```bash
# ワーカー起動（8ワーカー推奨）
./start_celery.sh start -w 8

# ステータス確認
./start_celery.sh status

# 再起動（Redis FLUSHDB含む）
redis-cli FLUSHDB && ./start_celery.sh restart -w 8

# 停止
./start_celery.sh stop
```

### 8.3 並列タスク投入

```python
from celery_tasks import submit_unified_qa_generation, collect_results

# タスク投入
tasks = submit_unified_qa_generation(
    processed_chunks, config, model, provider="gemini"
)

# 結果収集（タイムアウト設定）
timeout_seconds = min(max(len(tasks) * 10, 600), 1800)
qa_pairs = collect_results(tasks, timeout=timeout_seconds)
```

### 8.4 主要ファイル

| ファイル | 役割 |
|---------|------|
| `celery_tasks.py` | タスク定義、submit/collect関数 |
| `start_celery.sh` | ワーカー起動・管理スクリプト |
| `logs/celery_qa_*.log` | ワーカーログ |

---

## 9. カバレージ分析

### 9.1 データセット別最適閾値

```python
OPTIMAL_THRESHOLDS = {
    "cc_news": {
        "strict": 0.80,
        "standard": 0.70,
        "lenient": 0.60
    },
    "japanese_text": {
        "strict": 0.75,
        "standard": 0.65,
        "lenient": 0.55
    },
    "wikipedia_ja": {
        "strict": 0.85,   # 専門的な内容 → 高い類似度要求
        "standard": 0.75,
        "lenient": 0.65
    },
    "livedoor": {
        "strict": 0.78,
        "standard": 0.68,
        "lenient": 0.58
    }
}
```

### 9.2 多段階カバレージ分析

| レベル | 閾値（cc_news） | 用途 |
|--------|----------------|------|
| `strict` | 0.80 | 高品質要求時 |
| `standard` | 0.70 | 通常評価 |
| `lenient` | 0.60 | 最低限の網羅性確認 |

### 9.3 チャンク特性別分析

#### 長さ別分析

| カテゴリ | トークン数 | 分析観点 |
|---------|-----------|----------|
| `short` | < 100 | 短いチャンクのカバレージ |
| `medium` | 100-199 | 標準サイズのカバレージ |
| `long` | >= 200 | 長いチャンクのカバレージ |

#### 位置別分析

| カテゴリ | 位置 | 分析観点 |
|---------|------|----------|
| `beginning` | 前1/3 | 文書冒頭のカバレージ |
| `middle` | 中1/3 | 文書中盤のカバレージ |
| `end` | 後1/3 | 文書後半のカバレージ |

---

## 10. データセット設定

### 10.1 対応データセット一覧

```python
DATASET_CONFIGS = {
    "cc_news": {
        "name": "CC-News英語ニュース",
        "file": "OUTPUT/preprocessed_cc_news.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en",
        "chunk_size": 200,
        "qa_per_chunk": 3
    },
    "japanese_text": {
        "name": "日本語Webテキスト",
        "file": "OUTPUT/preprocessed_japanese_text.csv",
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja",
        "chunk_size": 200,
        "qa_per_chunk": 3
    },
    "wikipedia_ja": {
        "name": "Wikipedia日本語版",
        "file": "OUTPUT/preprocessed_wikipedia_ja.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
        "chunk_size": 200,
        "qa_per_chunk": 3
    },
    "livedoor": {
        "name": "Livedoorニュースコーパス",
        "file": "OUTPUT/preprocessed_livedoor.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
        "chunk_size": 200,
        "qa_per_chunk": 3
    }
}
```

---

## 11. コマンドラインオプション

### 11.1 全オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `--dataset` | str | None | データセットタイプ（--input-fileと排他） |
| `--input-file` | str | None | ローカルファイルパス（--datasetと排他） |
| `--model` | str | `gemini-2.0-flash` | 使用するGeminiモデル |
| `--output` | str | `qa_output/a02` | 出力ディレクトリ |
| `--max-docs` | int | None | 処理する最大文書数 |
| `--analyze-coverage` | flag | False | カバレージ分析を実行 |
| `--batch-chunks` | int | 3 | 1回のAPIで処理するチャンク数（1-5） |
| `--merge-chunks` | flag | True | 小さいチャンクを統合 |
| `--no-merge-chunks` | flag | - | チャンク統合を無効化 |
| `--min-tokens` | int | 150 | 統合対象の最小トークン数 |
| `--max-tokens` | int | 400 | 統合後の最大トークン数 |
| `--use-celery` | flag | False | Celery非同期並列処理を使用 |
| `--celery-workers` | int | 8 | Celeryワーカー数 |
| `--coverage-threshold` | float | None | カバレージ閾値（データセット別最適値を上書き） |

---

## 12. 実行方法

### 12.1 環境準備

```bash
# 1. 環境変数設定
echo "GOOGLE_API_KEY=your-api-key-here" > .env

# 2. Redisサーバー起動（Celery使用時）
brew services start redis  # macOS
# または
redis-server              # Linux/手動起動
```

### 12.2 テスト実行（同期処理）

```bash
python a02_make_qa_para.py \
    --dataset livedoor \
    --model gemini-2.0-flash \
    --max-docs 20 \
    --analyze-coverage
```

### 12.3 Celery並列実行（推奨）

```bash
# 1. ワーカー起動
./start_celery.sh start -w 8

# 2. 並列実行
python a02_make_qa_para.py \
    --dataset cc_news \
    --use-celery \
    --celery-workers 8 \
    --batch-chunks 3 \
    --merge-chunks \
    --min-tokens 150 \
    --max-tokens 400 \
    --model gemini-2.0-flash \
    --analyze-coverage

# 3. ワーカー停止
./start_celery.sh stop
```

### 12.4 ローカルファイル処理

```bash
python a02_make_qa_para.py \
    --input-file qa_output/my_data.csv \
    --use-celery \
    --celery-workers 8 \
    --max-docs 100 \
    --model gemini-2.0-flash
```

### 12.5 実行時の進捗表示

```
進捗: 成功=3/17, 失敗=0, 実行中=4, 待機中=10, 経過時間=15.2秒
進捗: 成功=7/17, 失敗=0, 実行中=4, 待機中=6, 経過時間=20.4秒
```

---

## 13. 出力ファイル

### 13.1 出力ディレクトリ構造

```
qa_output/
├── a02/
│   ├── qa_pairs_{dataset}_{timestamp}.json    # 全Q/Aペア（JSON）
│   ├── qa_pairs_{dataset}_{timestamp}.csv     # 全Q/Aペア（CSV全カラム）
│   ├── coverage_{dataset}_{timestamp}.json    # カバレージ分析結果
│   └── summary_{dataset}_{timestamp}.json     # サマリー情報
└── a02_qa_pairs_{dataset}.csv                 # 統一フォーマット（question/answerのみ）
```

### 13.2 Q/Aペア出力形式

**JSON形式:**
```json
[
  {
    "question": "質問文",
    "answer": "回答文",
    "question_type": "fact",
    "source_chunk_id": "cc_news_0_chunk_0",
    "doc_id": "cc_news_0",
    "dataset_type": "cc_news",
    "chunk_idx": 0
  }
]
```

**CSV全カラム版:**
```csv
question,answer,question_type,source_chunk_id,doc_id,dataset_type,chunk_idx
質問文,回答文,fact,cc_news_0_chunk_0,cc_news_0,cc_news,0
```

**統一フォーマット版:**
```csv
question,answer
質問文,回答文
```

### 13.3 カバレージ分析出力

```json
{
  "coverage_rate": 0.85,
  "covered_chunks": 85,
  "total_chunks": 100,
  "threshold": 0.70,
  "multi_threshold": {
    "strict": {"threshold": 0.80, "coverage_rate": 0.75},
    "standard": {"threshold": 0.70, "coverage_rate": 0.85},
    "lenient": {"threshold": 0.60, "coverage_rate": 0.92}
  },
  "chunk_analysis": {
    "by_length": {
      "short": {"coverage_rate": 0.80},
      "medium": {"coverage_rate": 0.88},
      "long": {"coverage_rate": 0.85}
    },
    "by_position": {
      "beginning": {"coverage_rate": 0.90},
      "middle": {"coverage_rate": 0.85},
      "end": {"coverage_rate": 0.80}
    }
  }
}
```

---

## 14. パフォーマンス

### 14.1 実行時間見積もり

| 処理モード | 文書数 | チャンク数 | API呼び出し | 実行時間 |
|-----------|--------|-----------|------------|---------|
| 同期処理 | 497 | 1,825 | 約365回 | 60-75分 |
| Celery並列（8ワーカー） | 497 | 1,825 | 約365回 | **8-15分** |
| Celery並列（4ワーカー） | 497 | 1,825 | 約365回 | 15-25分 |

### 14.2 処理モード比較

| モード | API呼び出し | 実行時間 | 効率化率 |
|--------|------------|---------|---------|
| 同期処理 | 1800回 | 180分 | 1.0x |
| **Celery並列** | 1800回 | **23分** | **7.8x** |
| ハイブリッド | 600回 | 8分 | 22.5x |

### 14.3 推奨ワーカー数

| 環境 | 推奨ワーカー数 | 備考 |
|------|--------------|------|
| 開発/テスト | 4 | 低リソース環境 |
| **本番（推奨）** | **8** | Gemini APIレート制限対策 |
| 高スループット | 16 | 複数APIキー時 |

---

## 15. トラブルシューティング

### 15.1 APIキーエラー

**症状:** `GOOGLE_API_KEYが設定されていません`

**解決策:**
```bash
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

### 15.2 Celeryワーカーが起動しない

**症状:** `Celeryワーカーが起動していません（応答なし）`

**解決策:**
```bash
# ログ確認
tail -f logs/celery_qa_*.log

# プロセス確認
ps aux | grep celery

# Redisの状態確認
redis-cli INFO clients

# ワーカー再起動
redis-cli FLUSHDB && ./start_celery.sh restart -w 8
```

### 15.3 タスクが処理されない

**症状:** タスクがキューに滞留

**解決策:**
```bash
# キューの状態確認
redis-cli LLEN celery

# ワーカー再起動
./start_celery.sh restart -w 8
```

### 15.4 MeCabが利用できない

**症状:** `⚠️ MeCabが利用できません（正規表現モード）`

**影響:** キーワード抽出精度がやや低下（機能には影響なし）

**解決策（オプション）:**
```bash
# macOS
brew install mecab mecab-ipadic
pip install mecab-python3

# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
pip install mecab-python3
```

### 15.5 カバレージが低い

**症状:** カバレージ率が目標に達しない

**解決策:**
```bash
# 1. 位置バイアス補正を確認（文書後半のカバレージ向上）
# determine_qa_count()が6番目以降のチャンクで+1しているか確認

# 2. バッチサイズを小さくする
--batch-chunks 1

# 3. カスタム閾値を下げる
--coverage-threshold 0.60
```

### 15.6 メモリ不足

**症状:** `MemoryError` または処理が極端に遅くなる

**解決策:**
```bash
# 文書数を制限
--max-docs 100

# バッチサイズを小さくする
--batch-chunks 1

# Celeryワーカー数を減らす
--celery-workers 4
```

---

## 付録A: 関連ファイル一覧

| ファイル | 説明 |
|---------|------|
| `a02_make_qa_para.py` | メインスクリプト |
| `helper_rag_qa.py` | SemanticCoverageクラス |
| `helper_llm.py` | LLMクライアント（create_llm_client） |
| `models.py` | QAPairsResponse Pydanticモデル |
| `config.py` | DATASET_CONFIGS, QAGenerationConfig |
| `celery_tasks.py` | Celeryタスク定義 |
| `start_celery.sh` | ワーカー管理スクリプト |
| `OUTPUT/preprocessed_*.csv` | 前処理済み入力データ |
| `qa_output/a02/` | 出力ディレクトリ |

---

## 付録B: 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-11-28 | 1.0 | 初版作成（Gemini移行対応） |
| 2025-12-18 | 2.0 | 全面再構成、方式位置づけ表・関数IPO・チャンキング技術詳細を追加 |