# a10_qa_optimized_hybrid_batch.py - 【推奨】最適化ハイブリッドバッチQ/A生成システム

作成日: 2025-11-28 (最終更新: 2025-12-18)

## 目次

1. [概要](#1-概要)
2. [方式・手法の位置づけ](#2-方式手法の位置づけ)
3. [アーキテクチャ](#3-アーキテクチャ)
4. [チャンキング技術詳細](#4-チャンキング技術詳細)
5. [関数リファレンス（IPO形式）](#5-関数リファレンスipo形式)
6. [データセット設定](#6-データセット設定)
7. [バッチ処理の詳細](#7-バッチ処理の詳細)
8. [品質重視モード](#8-品質重視モード)
9. [キャッシュ機能](#9-キャッシュ機能)
10. [比較実行モード](#10-比較実行モード)
11. [コマンドラインオプション](#11-コマンドラインオプション)
12. [実行方法](#12-実行方法)
13. [出力ファイル](#13-出力ファイル)
14. [パフォーマンス](#14-パフォーマンス)
15. [トラブルシューティング](#15-トラブルシューティング)

---

## 1. 概要

### 1.1 目的

`a10_qa_optimized_hybrid_batch.py`は、**RAGシステム構築におけるデフォルトの選択肢**として設計された、最適化ハイブリッドQ/A生成システムです。
Gemini API (`gemini-2.0-flash`) の長文脈処理能力を活かし、複数文書をバッチ処理することで、**API呼び出し回数を約90%削減**しながら、品質重視モードで**95%のカバレッジ**を実現します。

> **【推奨】**
> 通常のユースケース（ニュース記事、社内文書、Webページ等）では、まずこのスクリプトを使用してください。
> 最も高速かつ低コストに、高品質なQ/Aペアを大量生成できます。

### 1.2 起動コマンド

```bash
# 基本使用
python a10_qa_optimized_hybrid_batch.py --dataset cc_news --model gemini-2.0-flash

# 品質重視モード（推奨）
python a10_qa_optimized_hybrid_batch.py
  --dataset cc_news
  --model gemini-2.0-flash
  --quality-mode
  --target-coverage 0.95
  --batch-size 10
  --embedding-batch-size 300
```

### 1.3 主要機能

-   **大規模バッチ処理**: 複数文書を一度のLLM呼び出しで処理し、APIコストとレイテンシを削減
-   **ハイブリッド生成戦略**: ルールベースとLLM (`gemini-2.0-flash`) を組み合わせた効率的なQ/A生成
-   **品質重視モード**: `gemini-embedding-001` を使用したカバレッジ分析に基づき、目標95%達成を目指す高品質生成
-   **キャッシュ機能**: 埋め込みベクトルのキャッシュにより、2回目以降の実行時間を短縮
-   **段階的品質向上**: 初回は速度優先、後から品質を向上させる戦略をサポート
-   **MeCab対応**: 日本語データセットで高精度文境界検出（利用可能時）
-   **比較実行モード**: 通常版とバッチ版の性能を比較し、最適化効果を可視化

### 1.4 MeCab対応

| 言語 | MeCab利用可能時 | MeCab利用不可時 |
|------|----------------|----------------|
| 日本語（ja） | MeCabによる高精度文境界検出 | 正規表現にフォールバック |
| 英語（en） | 正規表現ベースの文分割 | 正規表現ベースの文分割 |

---

## 2. 方式・手法の位置づけ

### 2.1 Q/A生成手法の全体概要

本システムは複数の技術を組み合わせた**ハイブリッドアプローチ**を採用しています。

| カテゴリ | 手法 | 本システムでの適用 |
|---------|------|------------------|
| **文書分割** | セマンティックチャンキング | 段落優先→文単位→トークン強制分割の階層的分割 |
| **Q/A生成** | ルールベース抽出 | キーワード・エンティティ抽出による自動Q/A生成 |
| **Q/A生成** | LLM生成 | Gemini APIによる高品質Q/A生成 |
| **Q/A生成** | 階層的生成 | 文書全体→段落→エンティティの3層構造 |
| **品質評価** | カバレッジ分析 | 埋め込みベクトルのコサイン類似度による網羅率計算 |
| **最適化** | バッチ処理 | 複数文書の一括処理によるAPI効率化 |
| **最適化** | フィードバックループ | 未カバー領域への追加Q/A生成 |

### 2.2 他システムとの比較

| 項目 | a02（LLM版） | a03（テンプレート版） | **a10（ハイブリッドバッチ）** |
|------|-------------|---------------------|----------------------|
| **Q/A生成手法** | LLM (`gemini-2.0-flash`) のみ | テンプレートのみ | **ルールベース+LLM (Gemini)** |
| **チャンキング** | 固定長分割 | 固定長分割 | **セマンティック分割（段落優先）** |
| **API呼び出し** | 多い | 最小（埋め込みのみ） | **中程度（バッチ化で90%削減）** |
| **処理時間** | 60-80分 | 60-90分 | **大幅短縮 (例: 61分)** |
| **コスト** | 高い | 極めて低い | **中程度（効率化）** |
| **カバレッジ** | 90-95% | 95%+ | **95%（品質モード）** |
| **Q/A品質** | 非常に高い | 高い | **非常に高い（効率的）** |
| **スケーラビリティ** | 低い（レート制限） | 高い | **高い（バッチ処理）** |

### 2.3 技術的特徴の比較表

| 技術要素 | 詳細 | 利点 | 欠点 |
|---------|------|------|------|
| **セマンティックチャンキング** | 段落→文→トークンの階層的分割 | 意味的一貫性を維持 | 処理オーバーヘッド |
| **ハイブリッド生成** | ルールベース＋LLM | 効率と品質のバランス | 実装の複雑さ |
| **バッチ処理** | 複数文書の一括API呼び出し | API呼び出し92%削減 | メモリ使用量増加 |
| **カバレージフィードバック** | 未カバー領域への追加生成 | 95%カバレッジ達成 | 処理時間増加 |
| **階層的Q/A生成** | 文書→段落→エンティティ | 網羅的なQ/A生成 | Q/A数増加 |

---

## 3. アーキテクチャ

### 3.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                a10_qa_optimized_hybrid_batch.py                 │
├─────────────────────────────────────────────────────────────────┤
│  [1] データ読み込み                                              │
│      load_preprocessed_data()                                   │
│                              │                                  │
│                              ▼                                  │
│  [2] バッチ生成器初期化                                          │
│      BatchHybridQAGenerator(model="gemini-2.0-flash", ...)    │
│                              │                                  │
│                              ▼                                  │
│  [3] バッチ処理Q/A生成                                           │
│      generator.generate_batch_hybrid_qa()                       │
│      (LLMとEmbeddingにGemini APIを利用)                         │
│                              │                                  │
│                              ▼                                  │
│  [4] 結果保存                                                    │
│      save_batch_results() → qa_output/a10/                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 処理フロー詳細図

```
入力データ (CSV)
      │
      ▼
┌──────────────────┐
│ データ読み込み    │ load_preprocessed_data()
│ (pandas)         │
└──────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────┐
│              BatchHybridQAGenerator                   │
├──────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────┐  │
│  │ SemanticCoverage (チャンキング)                │  │
│  │  ├─ _split_into_paragraphs() 段落分割         │  │
│  │  ├─ _split_into_sentences() 文分割            │  │
│  │  │   ├─ MeCab (日本語・優先)                  │  │
│  │  │   └─ 正規表現 (英語/フォールバック)        │  │
│  │  ├─ _force_split_sentence() 強制分割          │  │
│  │  └─ _adjust_chunks_for_topic_continuity()     │  │
│  └────────────────────────────────────────────────┘  │
│                       │                               │
│                       ▼                               │
│  ┌────────────────────────────────────────────────┐  │
│  │ Q/A生成エンジン                                │  │
│  │  ├─ ルールベース抽出 (qa_extractor)           │  │
│  │  ├─ LLM生成 (gemini-2.0-flash)                │  │
│  │  └─ 階層的生成 (generate_hierarchical_qa)     │  │
│  └────────────────────────────────────────────────┘  │
│                       │                               │
│                       ▼                               │
│  ┌────────────────────────────────────────────────┐  │
│  │ カバレッジ計算                                  │  │
│  │  ├─ 埋め込み生成 (gemini-embedding-001)       │  │
│  │  ├─ コサイン類似度計算                        │  │
│  │  └─ フィードバックループ (未カバー追加生成)    │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────┐
│ 結果保存         │ save_batch_results()
│ (JSON/CSV)       │
└──────────────────┘
```

### 3.3 依存モジュール

```python
from helper_rag_qa import BatchHybridQAGenerator, OptimizedHybridQAGenerator
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
```

### 3.4 主要クラス

#### BatchHybridQAGenerator

`BatchHybridQAGenerator` は、`UnifiedLLMClient` を介してGemini LLM (`gemini-2.0-flash` など) と `gemini-embedding-001` を使用し、バッチ処理でQ/A生成を行うメインクラスです。

```python
generator = BatchHybridQAGenerator(
    model=model, # 例: "gemini-2.0-flash"
    batch_size=batch_size,
    embedding_batch_size=embedding_batch_size,
    quality_mode=quality_mode,
    target_coverage=target_coverage
)

batch_results = generator.generate_batch_hybrid_qa(
    texts=texts,
    qa_count=qa_count,
    use_llm=use_llm,
    calculate_coverage=calculate_coverage,
    document_type=doc_type,
    show_progress=True,
    lang=lang
)
```

#### OptimizedHybridQAGenerator

通常版（個別処理）のQ/A生成クラスで、比較実行モードで使用されます。こちらも内部で`UnifiedLLMClient`を利用します。

```python
normal_generator = OptimizedHybridQAGenerator(model=model) # 例: "gemini-2.0-flash"

result = normal_generator.generate_hybrid_qa(
    text=text,
    qa_count=3,
    use_llm=True,
    calculate_coverage=True,
    document_type="auto"
)
```

---

## 4. チャンキング技術詳細

### 4.1 セマンティックチャンキングの概要

本システムは**階層的セマンティックチャンキング**を採用しています。従来の固定長分割と異なり、文書の意味的構造を尊重した分割を行います。

```
┌────────────────────────────────────────────────────────────┐
│                セマンティックチャンキング階層               │
├────────────────────────────────────────────────────────────┤
│  優先度1: 段落境界 (\\n\\n)                                  │
│    └─ 著者が意図した最も重要なセマンティック境界            │
│                                                            │
│  優先度2: 文境界                                            │
│    ├─ 日本語: MeCab形態素解析（句点・疑問符・感嘆符）      │
│    └─ 英語: 正規表現 (?<=[。．.!?])\\s*                     │
│                                                            │
│  優先度3: トークン強制分割（最終手段）                      │
│    └─ 上限超過時にtiktokenで強制分割                       │
└────────────────────────────────────────────────────────────┘
```

### 4.2 チャンキングパラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `max_tokens` | 200 | チャンクの最大トークン数 |
| `min_tokens` | 50 | チャンクの最小トークン数（これより小さい場合は結合検討） |
| `prefer_paragraphs` | True | 段落ベースの分割を優先するか |

### 4.3 チャンキングアルゴリズム詳細

#### Step 1: 段落分割 (`_split_into_paragraphs`)

```python
# 空行（\n\n）で段落を分割
paragraphs = re.split(r'\n\s*\n', text)
```

#### Step 2: 段落ベースチャンク化 (`_chunk_by_paragraphs`)

```
入力: 段落リスト
      │
      ▼
┌─────────────────────────────────────────────┐
│ 各段落について:                              │
│  if 段落トークン数 <= max_tokens:           │
│      → チャンクとして採用 (type: paragraph)  │
│  else:                                       │
│      → 文単位に分割して処理                   │
│         if 単一文 > max_tokens:              │
│             → 強制分割 (type: forced_split)  │
│         else:                                │
│             → 文グループ (type: sentence_group)│
└─────────────────────────────────────────────┘
```

#### Step 3: 文分割 (`_split_into_sentences`)

**日本語（MeCab利用可能時）:**
```python
def _split_sentences_mecab(self, text: str) -> List[str]:
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)
    # 句点（。）、疑問符（？）、感嘆符（！）で分割
    # 形態素解析により高精度な文境界検出
```

**英語/フォールバック:**
```python
sentences = re.split(r'(?<=[。．.!?])\s*', text)
```

#### Step 4: トピック連続性調整 (`_adjust_chunks_for_topic_continuity`)

短すぎるチャンク（< min_tokens）を隣接チャンクとマージして意味的まとまりを維持。

### 4.4 チャンクタイプ一覧

| タイプ | 説明 | 生成条件 |
|--------|------|----------|
| `paragraph` | 段落単位チャンク | 段落がmax_tokens以下 |
| `sentence_group` | 文グループチャンク | 複数文の結合 |
| `forced_split` | 強制分割チャンク | 単一文がmax_tokens超過 |

### 4.5 日本語処理の特徴

| 項目 | MeCab使用時 | 正規表現使用時 |
|------|------------|---------------|
| 文境界検出精度 | 高い（形態素解析） | 中程度（パターンマッチ） |
| 処理速度 | やや遅い | 高速 |
| 依存関係 | MeCab, mecab-ipadic | なし |
| 自動フォールバック | - | MeCab未インストール時 |

---

## 5. 関数リファレンス（IPO形式）

### 5.1 a10_qa_optimized_hybrid_batch.py の関数

#### `load_preprocessed_data(dataset_type, max_docs)`

| 項目 | 内容 |
|------|------|
| **概要** | 前処理済みデータセットをCSVから読み込み |
| **INPUT** | `dataset_type`: str - データセットタイプ（cc_news, japanese_text, wikipedia_ja, livedoor）<br>`max_docs`: Optional[int] - 処理する最大文書数 |
| **PROCESS** | 1. DATASET_CONFIGSからファイルパス取得<br>2. pandas.read_csv()でCSV読み込み<br>3. 空テキスト除外<br>4. max_docs指定時は先頭から制限 |
| **OUTPUT** | `pd.DataFrame` - テキストデータを含むDataFrame |

#### `generate_batch_qa_from_dataset(...)`

| 項目 | 内容 |
|------|------|
| **概要** | バッチ処理によるQ/A生成のメイン関数 |
| **INPUT** | `df`: pd.DataFrame - 入力データ<br>`dataset_type`: str - データセットタイプ<br>`model`: str - LLMモデル名<br>`batch_size`: int - LLMバッチサイズ<br>`embedding_batch_size`: int - 埋め込みバッチサイズ<br>`qa_count`: Optional[int] - Q/A数<br>`use_llm`: bool - LLM使用フラグ<br>`calculate_coverage`: bool - カバレッジ計算フラグ<br>`doc_type`: Optional[str] - 文書タイプ<br>`output_dir`: str - 出力ディレクトリ<br>`quality_mode`: bool - 品質重視モード<br>`target_coverage`: float - 目標カバレージ率<br>`use_cache`: bool - キャッシュ使用フラグ<br>`cache_dir`: str - キャッシュディレクトリ<br>`progressive_quality`: bool - 段階的品質向上モード<br>`initial_coverage`: float - 初期目標カバレージ率<br>`final_coverage`: float - 最終目標カバレージ率 |
| **PROCESS** | 1. BatchHybridQAGenerator初期化<br>2. テキストリスト準備<br>3. generate_batch_hybrid_qa()実行<br>4. 統計情報集計（Q/A数、コスト、カバレッジ）<br>5. サマリー作成 |
| **OUTPUT** | `Dict` - `{"summary": {...}, "results": [...]}` |

#### `compare_with_normal_version(df, dataset_type, model, sample_size)`

| 項目 | 内容 |
|------|------|
| **概要** | 通常版（個別処理）とバッチ版の性能比較 |
| **INPUT** | `df`: pd.DataFrame - 入力データ<br>`dataset_type`: str - データセットタイプ<br>`model`: str - モデル名<br>`sample_size`: int - 比較サンプル数（デフォルト: 10） |
| **PROCESS** | 1. サンプルデータ抽出<br>2. OptimizedHybridQAGeneratorで個別処理実行<br>3. BatchHybridQAGeneratorでバッチ処理実行<br>4. 処理時間・API呼出回数を比較<br>5. 改善効果を計算（削減率、高速化倍率） |
| **OUTPUT** | `Dict` - 比較結果（normal_version, batch_version, improvement） |

#### `save_batch_results(generation_results, dataset_type, model, batch_size, output_dir)`

| 項目 | 内容 |
|------|------|
| **概要** | バッチ処理結果をファイルに保存 |
| **INPUT** | `generation_results`: Dict - 生成結果<br>`dataset_type`: str - データセットタイプ<br>`model`: str - モデル名<br>`batch_size`: int - バッチサイズ<br>`output_dir`: str - 出力ディレクトリ |
| **PROCESS** | 1. 出力ディレクトリ作成（qa_output/a10/）<br>2. サマリーファイル保存（JSON）<br>3. Q/Aペア保存（CSV - 全カラム版）<br>4. 統一フォーマット保存（CSV - question/answerのみ） |
| **OUTPUT** | `Dict[str, str]` - 保存ファイルパス（summary, qa_pairs_csv） |

#### `main()`

| 項目 | 内容 |
|------|------|
| **概要** | コマンドライン引数を解析しメイン処理を実行 |
| **INPUT** | コマンドライン引数（argparse） |
| **PROCESS** | 1. 引数パース<br>2. APIキー確認<br>3. データ読み込み<br>4. 比較モード or 通常バッチ処理を選択実行<br>5. 結果保存<br>6. 完了メッセージ・統計表示 |
| **OUTPUT** | なし（ファイル出力・コンソール表示） |

### 5.2 helper_rag_qa.py の主要関数（依存モジュール）

#### `SemanticCoverage.create_semantic_chunks(document, max_tokens, min_tokens, prefer_paragraphs, verbose)`

| 項目 | 内容 |
|------|------|
| **概要** | 文書を意味的に区切られたチャンクに分割（セマンティック分割） |
| **INPUT** | `document`: str - 分割対象文書<br>`max_tokens`: int - 最大トークン数（デフォルト: 200）<br>`min_tokens`: int - 最小トークン数（デフォルト: 50）<br>`prefer_paragraphs`: bool - 段落優先（デフォルト: True）<br>`verbose`: bool - 詳細出力 |
| **PROCESS** | 1. 段落ベース分割を試行（prefer_paragraphs=True時）<br>2. 各段落をトークン数でチェック<br>3. 大きすぎる段落は文単位に分割<br>4. トピック連続性を考慮した再調整 |
| **OUTPUT** | `List[Dict]` - チャンク辞書リスト（id, text, type, sentences, start_sentence_idx, end_sentence_idx） |

#### `BatchHybridQAGenerator.generate_batch_hybrid_qa(texts, qa_count, use_llm, calculate_coverage, document_type, show_progress, lang)`

| 項目 | 内容 |
|------|------|
| **概要** | 複数テキストをバッチ処理でQ/A生成 |
| **INPUT** | `texts`: List[str] - テキストリスト<br>`qa_count`: int - Q/A数<br>`use_llm`: bool - LLM使用フラグ<br>`calculate_coverage`: bool - カバレッジ計算フラグ<br>`document_type`: str - 文書タイプ<br>`show_progress`: bool - プログレスバー表示<br>`lang`: str - 言語コード |
| **PROCESS** | 1. テキストをバッチに分割<br>2. 各バッチでLLM/ルールベースQ/A生成<br>3. カバレッジ計算（指定時）<br>4. 統計情報更新 |
| **OUTPUT** | `List[Dict]` - 各文書のQ/A生成結果 |

#### `BatchHybridQAGenerator.generate_with_coverage_feedback(text, target_coverage, max_iterations, lang)`

| 項目 | 内容 |
|------|------|
| **概要** | カバレージフィードバックループでQ/A生成（品質モード） |
| **INPUT** | `text`: str - 入力テキスト<br>`target_coverage`: float - 目標カバレージ率<br>`max_iterations`: int - 最大反復回数<br>`lang`: str - 言語コード |
| **PROCESS** | 1. 初回は階層的生成（quality_mode時）<br>2. カバレージ計算<br>3. 目標未達の場合、未カバーチャンク特定<br>4. ターゲットQ/A追加生成<br>5. 目標達成または最大反復まで繰り返し |
| **OUTPUT** | `Dict` - qa_pairs, final_coverage, iterations, total_qa |

#### `BatchHybridQAGenerator.generate_hierarchical_qa(text, lang)`

| 項目 | 内容 |
|------|------|
| **概要** | 階層的アプローチでQ/Aを生成（3段階） |
| **INPUT** | `text`: str - 対象テキスト<br>`lang`: str - 言語コード |
| **PROCESS** | 1. 第1層: 文書全体の包括的質問（1-2個）<br>2. 第2層: パラグラフレベルの詳細質問（3-4個）<br>3. 第3層: キーワード/エンティティ特化質問（5-6個） |
| **OUTPUT** | `List[Dict]` - 階層的Q/Aペア（type: comprehensive/paragraph_detail/entity_specific, layer: 1-3） |

#### `BatchHybridQAGenerator.calculate_optimal_qa_count(text, target_coverage)`

| 項目 | 内容 |
|------|------|
| **概要** | 文書の複雑度に基づいて最適なQ/A数を動的決定 |
| **INPUT** | `text`: str - 対象文書<br>`target_coverage`: float - 目標カバレージ率 |
| **PROCESS** | 1. 文書長・文数・ユニーク概念数から情報密度計算<br>2. 情報密度に応じた基本Q/A数決定<br>3. チャンク数に基づく調整<br>4. 品質モード時は50%増加 |
| **OUTPUT** | `int` - 最適Q/A数（上限15個） |

---

## 6. データセット設定

### 6.1 対応データセット一覧

```python
DATASET_CONFIGS = {
    "cc_news": {
        "name": "CC-News英語ニュース",
        "file": "OUTPUT/preprocessed_cc_news.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en",
        "default_doc_type": "news"
    },
    "japanese_text": {
        "name": "日本語Webテキスト",
        "file": "OUTPUT/preprocessed_japanese_text.csv",
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja",
        "default_doc_type": "auto"
    },
    "wikipedia_ja": {
        "name": "Wikipedia日本語版",
        "file": "OUTPUT/preprocessed_wikipedia_ja.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
        "default_doc_type": "academic"
    },
    "livedoor": {
        "name": "Livedoorニュースコーパス",
        "file": "OUTPUT/preprocessed_livedoor.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "category_column": "category",
        "lang": "ja",
        "default_doc_type": "news"
    }
}
```

### 6.2 文書タイプ（doc_type）

| タイプ | 説明 | 適用データセット | Q/A生成の特徴 |
|--------|------|-----------------|--------------|
| `news` | ニュース記事 | cc_news, livedoor | 5W1H型質問が多い |
| `academic` | 学術・専門記事 | wikipedia_ja | 定義・説明型質問が多い |
| `auto` | 自動判定 | japanese_text | 内容に応じて最適化 |
| `technical` | 技術文書 | - | 手順・仕組み型質問が多い |

---

## 7. バッチ処理の詳細

### 7.1 バッチサイズの選択基準

#### LLMバッチサイズ（--batch-size）

| バッチサイズ | API削減率 | 推奨用途 | メリット | デメリット |
|------------|---------|---------|---------|---------|
| 5 | 80% | **品質最優先** | 高精度、エラー少ない | やや低速 |
| 10 | 90% | **推奨設定** | 速度と品質のバランス | - |
| 15 | 93% | 大規模データ | 高速 | プロンプトが長大化 |
| 20 | 95% | 超大規模データ | 最高速 | プロンプトが長大化 |

#### 埋め込みバッチサイズ（--embedding-batch-size）

Gemini `gemini-embedding-001` は、100トークンあたりのコストが非常に低く、効率的なバッチ処理が可能です。

| バッチサイズ | 処理速度 | 推奨用途 |
|------------|---------|---------|
| 100 | 標準（デフォルト） | 中規模データ |
| 300 | 高速 | **推奨設定** |
| 500 | 最高速 | 大規模データ |

### 7.2 バッチ統計情報

実行後、以下の統計情報が`generator.batch_stats`に格納されます:

```python
{
    "llm_batches": 50,           # LLMバッチ処理回数
    "embedding_batches": 10,     # 埋め込みバッチ処理回数
    "total_llm_calls": 110,      # LLM API呼び出し総数
    "total_embedding_calls": 10, # 埋め込みAPI呼び出し総数
    "coverage_iterations": 2,    # カバレージ改善イテレーション数
}
```

---

## 8. 品質重視モード

### 8.1 品質重視モードとは

`--quality-mode`を指定すると、`gemini-embedding-001` を使用したカバレッジ分析に基づき、カバレッジ95%を目標とした高品質Q/A生成を行います。

```bash
python a10_qa_optimized_hybrid_batch.py
    --dataset cc_news
    --quality-mode
    --target-coverage 0.95
    --model gemini-2.0-flash
```

### 8.2 品質モードの内部動作

1. **バッチサイズ自動調整**: `batch_size = min(5, 指定値)`
2. **階層的Q/A生成**: 3層構造での網羅的生成
3. **カバレージフィードバックループ**: 未カバー領域への追加生成
4. **最適Q/A数の動的決定**: 文書複雑度に基づく調整

### 8.3 通常モードとの違い

| 項目 | 通常モード | 品質重視モード |
|------|-----------|--------------|
| **目標カバレッジ** | 85% | 95% |
| **Q/A生成数** | 少なめ | 多め（基本値×1.5） |
| **LLM呼び出し** | 最小限 | 必要に応じて増加 |
| **処理時間** | 短い | やや長い（+20-30%） |
| **コスト** | 低い | やや高い（+30-50%） |
| **フィードバックループ** | なし | あり（最大3回） |

### 8.4 カバレージ目標の調整

```bash
# 90%カバレッジ（速度重視）
--quality-mode --target-coverage 0.90

# 95%カバレッジ（推奨）
--quality-mode --target-coverage 0.95

# 98%カバレッジ（品質最優先）
--quality-mode --target-coverage 0.98
```

---

## 9. キャッシュ機能

### 9.1 キャッシュの有効化

```bash
python a10_qa_optimized_hybrid_batch.py
    --dataset cc_news
    --use-cache
    --cache-dir qa_cache
    --model gemini-2.0-flash
```

### 9.2 キャッシュの効果

| 実行回 | 処理時間 | 効果 |
|--------|---------|------|
| 初回 | 61分 | キャッシュ作成 |
| 2回目以降 | **15-25分** | **50%短縮** |

### 9.3 キャッシュディレクトリ構造

```
qa_cache/
├── cc_news_embeddings.pkl
├── cc_news_chunks.pkl
└── cc_news_qa_pairs.pkl
```

---

## 10. 比較実行モード

### 10.1 比較モードの使用

通常版（個別処理）とバッチ版の性能を比較します:

```bash
python a10_qa_optimized_hybrid_batch.py
    --dataset cc_news
    --compare
    --compare-size 10
    --model gemini-2.0-flash
```

### 10.2 比較結果の出力

```bash
================================================================================
📊 性能比較結果
================================================================================
サンプル数: 10文書

【通常版（個別処理）】
  処理時間: 45.00秒
  API呼出: 30回
  1文書あたり: 4.50秒, 3.0回

【バッチ版（バッチ処理）】
  処理時間: 12.00秒
  API呼出: 5回
  1文書あたり: 1.20秒, 0.5回

【改善効果】
  処理時間短縮: 73.3%
  API呼出削減: 83.3%
  高速化: 3.75x
================================================================================
```

### 10.3 比較結果の保存

```
qa_output/comparison_cc_news_20251127_143052.json
```

---

## 11. コマンドラインオプション

### 11.1 全オプション一覧

| オプション | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `--dataset` | str | cc_news | データセットタイプ |
| `--model` | str | **`gemini-2.0-flash`** | 使用するLLMモデル |
| `--batch-size` | int | 10 | LLMバッチサイズ |
| `--embedding-batch-size` | int | 100 | 埋め込みバッチサイズ |
| `--max-docs` | int | None | 処理する最大文書数 |
| `--qa-count` | int | None | 文書あたりのQ/A数 |
| `--doc-type` | str | None | 文書タイプ（news/technical/academic/auto） |
| `--no-llm` | flag | False | LLMを使用しない |
| `--no-coverage` | flag | False | カバレッジ計算を行わない |
| `--output` | str | qa_output | 出力ディレクトリ |
| `--compare` | flag | False | 通常版との比較実行 |
| `--compare-size` | int | 10 | 比較実行のサンプルサイズ |
| `--quality-mode` | flag | False | 品質重視モード |
| `--target-coverage` | float | 0.95 | 目標カバレッジ率 |
| `--use-cache` | flag | False | キャッシュを使用 |
| `--cache-dir` | str | qa_cache | キャッシュディレクトリ |
| `--progressive-quality` | flag | False | 段階的品質向上モード |
| `--initial-coverage` | float | 0.85 | 初期目標カバレージ率 |
| `--final-coverage` | float | 0.95 | 最終目標カバレージ率 |

---

## 12. 実行方法

### 12.1 基本実行

```bash
python a10_qa_optimized_hybrid_batch.py --dataset cc_news --model gemini-2.0-flash
```

### 12.2 品質重視モード（推奨）

```bash
python a10_qa_optimized_hybrid_batch.py
    --dataset cc_news
    --model gemini-2.0-flash
    --quality-mode
    --target-coverage 0.95
    --batch-size 10
    --embedding-batch-size 300
    --output qa_output
```

### 12.3 キャッシュ活用版（2回目以降）

```bash
python a10_qa_optimized_hybrid_batch.py
    --dataset cc_news
    --model gemini-2.0-flash
    --quality-mode
    --use-cache
    --cache-dir qa_cache
```

### 12.4 段階的品質向上版

```bash
python a10_qa_optimized_hybrid_batch.py
    --dataset cc_news
    --model gemini-2.0-flash
    --progressive-quality
    --initial-coverage 0.85
    --final-coverage 0.95
    --batch-size 15
```

### 12.5 日本語データセット（MeCab自動利用）

```bash
# Wikipedia日本語版
python a10_qa_optimized_hybrid_batch.py --dataset wikipedia_ja --model gemini-2.0-flash

# Livedoorニュースコーパス
python a10_qa_optimized_hybrid_batch.py
    --dataset livedoor
    --model gemini-2.0-flash
    --quality-mode
    --max-docs 500
    --batch-size 20
    --embedding-batch-size 500
    --use-cache
    --cache-dir qa_cache_livedoor
```

---

## 13. 出力ファイル

### 13.1 出力ディレクトリ構造

```
qa_output/
├── a10/
│   ├── batch_summary_{dataset}_{model}_b{batch}_{timestamp}.json
│   └── batch_qa_pairs_{dataset}_{model}_b{batch}_{timestamp}.csv
└── a10_qa_pairs_{dataset}.csv    # 統一フォーマット
```

### 13.2 サマリーファイル（JSON）

```json
{
  "dataset_type": "cc_news",
  "dataset_name": "CC-News英語ニュース",
  "model_used": "gemini-2.0-flash",
  "batch_processing": true,
  "batch_sizes": {
    "llm_batch_size": 10,
    "embedding_batch_size": 300
  },
  "documents_processed": 497,
  "total_qa_generated": 2485,
  "avg_qa_per_doc": 5.0,
  "processing_time": {
    "total_seconds": 3678,
    "minutes": 61.3,
    "docs_per_second": 0.135
  },
  "api_usage": {
    "total_cost": 0.18,
    "cost_per_doc": 0.00036,
    "batch_statistics": {
      "total_llm_calls": 110,
      "total_embedding_calls": 10
    }
  },
  "coverage": {
    "calculated": true,
    "avg_coverage": 95.2,
    "min_coverage": 87.0,
    "max_coverage": 99.0
  },
  "generation_timestamp": "2025-11-27T14:30:52"
}
```

### 13.3 Q/Aペアファイル（CSV）

**全カラム版（batch_qa_pairs_*.csv）**:
```csv
doc_id,question,answer,doc_title,text_length
cc_news_0,What is the main topic?,AI technology...,Article Title,1234
cc_news_1,How does it work?,It uses...,Another Title,2345
```

**統一フォーマット版（a10_qa_pairs_{dataset}.csv）**:
```csv
question,answer
What is the main topic?,AI technology...
How does it work?,It uses...
```

---

## 14. パフォーマンス

### 14.1 実行時間見積もり

| データセット | 文書数 | バッチサイズ | 実行時間 | コスト | カバレッジ |
|------------|--------|------------|---------|--------|----------|
| cc_news | 497 | 10 | 61分 | **低** | 95% |
| livedoor | 500 | 20 | 30-50分 | **低** | 95% |
| livedoor（キャッシュ） | 500 | 20 | 15-25分 | **極低** | 95% |

### 14.2 バッチ処理の効果

```
バッチ処理により以下の改善を実現：

1. **API呼び出し削減**
   - 通常版（推定）: 1491回
   - バッチ版（実際）: 110回
   - 削減率: 92.6%

2. **処理速度向上**
   - 処理速度: 0.14文書/秒
   - 497文書を61.3分で処理

3. **スケーラビリティ**
   - 大規模データセット処理が現実的に
   - レート制限リスクの大幅低減
```

### 14.3 メモリ使用量

| バッチサイズ | 推定メモリ使用量 | 備考 |
|------------|----------------|------|
| 5 | 2-4 GB | 低リソース環境向け |
| 10 | 4-6 GB | 標準設定 |
| 20 | 6-10 GB | 高メモリ環境向け |

---

## 15. トラブルシューティング

### 15.1 APIキーエラー

**症状**: `Google APIキーが設定されていません`

**解決策**:
```bash
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

### 15.2 ファイルが見つからない

**症状**: `FileNotFoundError: ファイルが見つかりません`

**解決策**:
```bash
ls OUTPUT/preprocessed_cc_news.csv
```

### 15.3 レート制限エラー

**症状**: `Resource Exhausted` (Gemini APIの場合) または `RateLimitError`

**解決策**:
```bash
# LLMバッチサイズを小さくする
--batch-size 5

# 埋め込みバッチサイズを小さくする (helper_embedding.pyの設定を確認)
--embedding-batch-size 50
```

### 15.4 カバレッジが目標に達しない

**解決策**:
```bash
# 方法1: 品質重視モードを使用
--quality-mode

# 方法2: LLMバッチサイズを小さくする（品質向上を優先）
--batch-size 5

# 方法3: 埋め込みバッチサイズを調整する (helper_embedding.pyの設定を確認)
--embedding-batch-size 50
```

### 15.5 MeCabが利用できない

**症状**: `⚠️ MeCabが利用できません（正規表現にフォールバック）`

**影響**: 日本語の文境界検出精度がやや低下（機能には影響なし）

**解決策（オプション）**:
```bash
# macOS
brew install mecab mecab-ipadic
pip install mecab-python3

# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
pip install mecab-python3
```

### 15.6 メモリ不足エラー

**症状**: `MemoryError` または処理が極端に遅くなる

**解決策**:
```bash
# バッチサイズを小さくする
--batch-size 5 --embedding-batch-size 50

# 文書数を制限する
--max-docs 100
```

---

## 付録A: 実行ログサンプル

```
=====================================
バッチ処理版ハイブリッドQ&A生成
=====================================
データセット: CC-News英語ニュース
モデル: gemini-2.0-flash
バッチサイズ: LLM=10, 埋め込み=300
出力先: qa_output
最大文書数: 制限なし

[1/3] データ読み込み...
2025-11-27 14:30:00 - INFO - データ読み込み中: OUTPUT/preprocessed_cc_news.csv
2025-11-27 14:30:01 - INFO - 読み込み完了: 497件のデータ

[2/3] バッチ処理Q/A生成...
🎯 品質重視モード: 目標カバレージ 95%
2025-11-27 14:30:01 - INFO - バッチ処理Q/A生成開始: 497件の文書
2025-11-27 14:30:01 - INFO - バッチサイズ: LLM=10, 埋め込み=300
2025-11-27 14:30:01 - INFO - データセット言語: en
2025-11-27 14:30:01 - INFO -   → 英語データセット: 正規表現ベースの文分割を使用

バッチ処理中: 100%|██████████| 50/50 [61:18<00:00, 73.57s/batch]

[3/3] 結果保存...

=====================================
処理完了
=====================================
処理文書数: 497
生成Q/A総数: 2485
平均Q/A数/文書: 5.0

処理時間:
- 合計: 3678.00秒
- 分: 61.30分
- 処理速度: 0.14文書/秒

API使用状況:
- LLM呼び出し: 110回
- 埋め込み呼び出し: 10回
- 総コスト: $0.1800

カバレージ:
- 平均: 95.2%
- 最小: 87.0%
- 最大: 99.0%

保存ファイル:
- サマリー: qa_output/a10/batch_summary_cc_news_gemini_2_0_flash_b10_20251127_153119.json
- Q/A CSV: qa_output/a10/batch_qa_pairs_cc_news_gemini_2_0_flash_b10_20251127_153119.csv
- 統一フォーマット: qa_output/a10_qa_pairs_cc_news.csv
```

---

## 付録B: 関連ファイル一覧

| ファイル | 説明 |
|---------|------|
| `a10_qa_optimized_hybrid_batch.py` | メインスクリプト |
| `helper_rag_qa.py` | BatchHybridQAGenerator, SemanticCoverageクラス |
| `helper_embedding.py` | 埋め込み生成ユーティリティ |
| `OUTPUT/preprocessed_*.csv` | 前処理済み入力データ |
| `qa_output/a10/` | 出力ディレクトリ |

---

## 付録C: 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-11-28 | 1.0 | 初版作成（Gemini移行対応） |
| 2025-12-18 | 1.1 | チャンキング技術詳細、関数IPO、方式位置づけ表を追加 |