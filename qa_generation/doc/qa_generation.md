# qa_generation モジュール完全ガイド

## 📋 目次

1. [概要](#概要)
2. [システムアーキテクチャ](#システムアーキテクチャ)
3. [データ処理フロー](#データ処理フロー)
4. [モジュール詳細](#モジュール詳細)
5. [使用方法](#使用方法)
6. [パラメータリファレンス](#パラメータリファレンス)
7. [実行例とワークフロー](#実行例とワークフロー)
8. [トラブルシューティング](#トラブルシューティング)
9. [ベストプラクティス](#ベストプラクティス)

---

## 📖 概要

`qa_generation` は、**Q/Aペアを自動生成するための総合的なPythonパッケージ**です。テキストデータから高品質なQ/Aペアを生成し、RAG（Retrieval-Augmented Generation）システムの学習データとして活用できます。

### 主な特徴

✅ **モジュラーアーキテクチャ**

- 各機能が独立したモジュールとして実装
- 柔軟な組み合わせとカスタマイズが可能

✅ **高度なチャンク処理**

- セマンティック分割による文脈を保持したチャンク作成
- 段落優先、ベクトル類似度分割、オーバーラップなど多様なオプション

✅ **LLMベースのQ/A生成**

- Gemini 2.0 FlashによるQ/A生成
- チャンク特性に応じた最適なQ/A数の自動決定

✅ **包括的なカバレージ分析**

- 多段階閾値評価（Strict/Standard/Lenient）
- チャンク特性別分析（長さ別、位置別）
- 埋め込みベクトルによる意味的カバレージ測定

✅ **並列処理対応**

- Celeryによる非同期タスク実行
- ThreadPoolExecutorによる並列チャンク作成
- 大規模データセットの高速処理

---

## 🏗️ システムアーキテクチャ

### モジュール構成図

```
 qa_generation/
├── __init__.py              # パッケージ初期化、エクスポート定義
│
├── models.py                # データモデル（Pydantic）
│   ├── QAPair               # 基本Q/Aペアモデル
│   ├── QAPairsList          # Q/Aペアリスト
│   ├── EnhancedQAPair       # 拡張Q/Aペアモデル
│   └── QAGenerationConsiderations  # 生成要件モデル
│
├── pipeline.py              # パイプライン制御（メインロジック）
│   └── QAPipeline           # データ読み込み → チャンク作成 → Q/A生成
│
├── structure.py                      # チャンク作成・統合
│   ├── create_semantic_chunks()      # セマンティック分割
│   ├── create_document_chunks()      # 文書からチャンク作成（並列）
│   └── merge_small_chunks()          # 小チャンクの統合
│
├── generation.py                     # Q/A生成（LLM呼び出し）
│   ├── QAGenerator                   # Q/A生成クラス
│   └── generate_qa_dataset()         # データセット全体のQ/A生成
│
├── semantic.py              # セマンティック分析・カバレッジ
│   └── SemanticCoverage     # 埋め込みベクトル生成、類似度計算
│
├── evaluation.py                    # カバレージ分析
│   ├── analyze_coverage()           # 総合カバレージ分析
│   ├── multi_threshold_coverage()   # 多段階閾値評価
│   └── analyze_chunk_characteristics_coverage()  # チャンク特性別分析
│
├── content.py               # コンテンツ分析・キーワード抽出
│   ├── KeywordExtractor    # MeCab/正規表現によるキーワード抽出
│   ├── analyze_chunk_complexity()   # チャンク複雑度分析
│   └── extract_key_concepts()       # 主要概念抽出
│
├── data_io.py               # データ入出力
│   ├── load_uploaded_file()         # ローカルファイル読み込み
│   ├── load_preprocessed_data()     # 前処理済みデータ読み込み
│   └── save_results()               # 結果保存（JSON/CSV）
│
└── config.py                # 設定管理
    ├── OPTIMAL_THRESHOLDS   # データセット別最適閾値
    └── LOCAL_DATASET_EXTENSIONS  # ローカル拡張設定
```

### レイヤー構成

```
┌────────────────────────────────────────────────┐
│  Application Layer (pipeline.py)              │  ← パイプライン制御
│  - QAPipeline                                 │
├────────────────────────────────────────────────┤
│  Business Logic Layer                          │
│  ├─ Q/A生成 (generation.py)                     │  ← LLMによるQ/A生成
│  ├─ チャンク処理 (structure.py)                  │  ← セマンティック分割
│  ├─ カバレージ分析 (evaluation.py)               │  ← 品質評価
│  └─ コンテンツ分析 (content.py)                  │  ← キーワード抽出
├────────────────────────────────────────────────┤
│  Infrastructure Layer                          │
│  ├─ セマンティック処理 (semantic.py)              │  ← 埋め込み生成
│  ├─ データI/O (data_io.py)                      │  ← ファイル処理
│  └─ データモデル (models.py)                     │  ← Pydantic検証
├────────────────────────────────────────────────┤
│  External Dependencies                         │
│  ├─ Google Gemini API (LLM)                    │  ← Q/A生成、埋め込み
│  ├─ Celery (タスクキュー)                        │  ← 並列処理
│  └─ MeCab (形態素解析)                           │  ← キーワード抽出（オプション）
└────────────────────────────────────────────────┘
```

### 依存関係

```
QAPipeline (pipeline.py)
├── データ読み込み
│   └── data_io.py
│       ├── load_uploaded_file()
│       └── load_preprocessed_data()
│
├── チャンク作成
│   └── structure.py
│       └── create_document_chunks()
│           └── semantic.py::SemanticCoverage.create_semantic_chunks()
│
├── Q/A生成
│   └── generation.py
│       └── QAGenerator.generate_for_chunk()
│           └── helper.helper_llm::LLMClient
│
├── カバレージ分析
│   └── evaluation.py
│       └── analyze_coverage()
│           └── semantic.py::SemanticCoverage
│
└── 結果保存
    └── data_io.py::save_results()
```

---

## 🔄 データ処理フロー

### 全体フロー図

```
入力データ
   │
   ├─ Dataset (config.py定義)
   ├─ ローカルファイル (.txt, .csv, .json)
   └─ チャンクCSV (事前作成済み)
   │
   v
┌──────────────────────────────────────────────┐
│ Phase 1: データ読み込み (data_io.py)          │
├──────────────────────────────────────────────┤
│ load_uploaded_file() または                  │
│ load_preprocessed_data()                     │
│   ↓                                          │
│ DataFrame形式に変換                          │
│ - Combined_Text カラム確保                   │
│ - 空白行除去                                 │
└──────────────┬───────────────────────────────┘
               │
               v
┌──────────────────────────────────────────────┐
│ Phase 2: チャンク作成 (structure.py)          │
├──────────────────────────────────────────────┤
│                                              │
│ チャンクCSV読み込み？                         │
│   YES → load_chunks_from_csv()              │
│   NO  → create_document_chunks()            │
│         ↓                                   │
│         並列処理（ThreadPoolExecutor）        │
│         ├─ 文書1 → セマンティック分割         │
│         ├─ 文書2 → セマンティック分割         │
│         ├─ 文書3 → セマンティック分割         │
│         └─ ...                              │
│                                              │
│ セマンティック分割プロセス:                   │
│ ┌────────────────────────────┐              │
│ │ semantic.py                │              │
│ │ SemanticCoverage           │              │
│ │                            │              │
│ │ 1. 段落/文への分割          │              │
│ │ 2. トークン数計算           │              │
│ │ 3. ベクトル類似度分割（オプション） │       │
│ │ 4. オーバーラップ追加（オプション） │       │
│ │ 5. チャンク構築             │              │
│ └────────────────────────────┘              │
│                                              │
│ merge_chunks=True の場合:                    │
│   ↓                                          │
│   merge_small_chunks()                       │
│   - 150トークン未満のチャンクを統合           │
│   - 最大400トークンまで                       │
└──────────────┬───────────────────────────────┘
               │
               v
       チャンクリスト
       [
         {id, text, tokens, type, ...},
         {id, text, tokens, type, ...},
         ...
       ]
               │
               v
┌──────────────────────────────────────────────┐
│ Phase 3: Q/A生成 (generation.py)              │
├──────────────────────────────────────────────┤
│                                              │
│ Celery並列処理？                              │
│   YES → Celeryタスクキューで実行              │
│   NO  → 逐次実行                              │
│                                              │
│ 各チャンクで:                                 │
│ ┌────────────────────────────┐              │
│ │ QAGenerator                │              │
│ │                            │              │
│ │ 1. チャンク特性分析         │              │
│ │    - トークン数             │              │
│ │    - 位置（前半/後半）      │              │
│ │    ↓                       │              │
│ │ 2. Q/A数決定               │              │
│ │    - 50tk未満: 2個         │              │
│ │    - 50-100tk: 3個         │              │
│ │    - 100-200tk: 4-5個      │              │
│ │    - 200tk以上: 5-8個      │              │
│ │    ↓                       │              │
│ │ 3. プロンプト構築           │              │
│ │    - システムプロンプト     │              │
│ │    - チャンクテキスト       │              │
│ │    - Q/A生成指示           │              │
│ │    ↓                       │              │
│ │ 4. Gemini API呼び出し      │              │
│ │    - JSON形式で応答取得    │              │
│ │    ↓                       │              │
│ │ 5. パース＆検証             │              │
│ │    - Pydanticモデル検証    │              │
│ │    - エラーハンドリング     │              │
│ └────────────────────────────┘              │
│                                              │
└──────────────┬───────────────────────────────┘
               │
               v
       Q/Aペアリスト
       [
         {question, answer, chunk_id, ...},
         {question, answer, chunk_id, ...},
         ...
       ]
               │
               v
┌──────────────────────────────────────────────┐
│ Phase 4: カバレージ分析 (evaluation.py)       │
├──────────────────────────────────────────────┤
│                                              │
│ analyze_coverage()                           │
│   ↓                                          │
│ 1. 埋め込みベクトル生成                       │
│    ├─ チャンクの埋め込み（semantic.py）       │
│    └─ Q/Aペアの埋め込み（semantic.py）        │
│                                              │
│ 2. カバレージ行列計算                         │
│    - コサイン類似度行列（NumPy最適化）         │
│    - [チャンク数 × Q/A数] の行列              │
│                                              │
│ 3. 多段階閾値評価                             │
│    - Strict: 0.80                            │
│    - Standard: 0.70                          │
│    - Lenient: 0.60                           │
│                                              │
│ 4. チャンク特性別分析                         │
│    ┌─ 長さ別（Short/Medium/Long）            │
│    └─ 位置別（Beginning/Middle/End）         │
│                                              │
│ 5. 未カバーチャンク特定                       │
│    - 類似度がしきい値未満のチャンク           │
│    - ギャップ計算                             │
│                                              │
└──────────────┬───────────────────────────────┘
               │
               v
       カバレージ結果
       {
         coverage_rate: 0.85,
         multi_threshold: {...},
         chunk_analysis: {...},
         uncovered_chunks: [...]
       }
               │
               v
┌──────────────────────────────────────────────┐
│ Phase 5: 結果保存 (data_io.py)                │
├──────────────────────────────────────────────┤
│                                              │
│ save_results()                               │
│   ↓                                          │
│   ├─ qa_pairs_{dataset}_{timestamp}.json    │
│   ├─ qa_pairs_{dataset}_{timestamp}.csv     │
│   ├─ coverage_{dataset}_{timestamp}.json    │
│   └─ summary_{dataset}_{timestamp}.json     │
│                                              │
└──────────────────────────────────────────────┘
```

### チャンク作成の詳細フロー

```
┌─────────────────────────────────────────────┐
│ セマンティック分割プロセス                    │
│ (semantic.py::create_semantic_chunks)        │
└─────────────────────────────────────────────┘

入力: document (文書テキスト)

Step 1: 段落/文への分割
  ┌──────────────────────────┐
  │ prefer_paragraphs=True   │
  │   ↓                      │
  │ 段落で分割               │
  │ - "\n\n" で区切り        │
  │ - 各段落をチャンク候補   │
  │                          │
  │ prefer_paragraphs=False  │
  │   ↓                      │
  │ 文単位で分割             │
  │ - 句点（。.）で区切り    │
  │ - MeCab使用（オプション）│
  └──────────┬───────────────┘
             │
             v
Step 2: トークン数計算
  ┌──────────────────────────┐
  │ 各セグメントで:           │
  │   token_count = tiktoken │
  │                          │
  │ max_tokens を超える？    │
  │   YES → 再分割           │
  │   NO  → そのまま         │
  └──────────┬───────────────┘
             │
             v
Step 3: ベクトル類似度分割（オプション）
  ┌──────────────────────────┐
  │ use_similarity=True      │
  │   ↓                      │
  │ 1. 各文の埋め込み生成    │
  │ 2. 連続文間の類似度計算  │
  │ 3. similarity_threshold  │
  │    未満の箇所で分割      │
  │                          │
  │ → トピック境界を自動検出 │
  └──────────┬───────────────┘
             │
             v
Step 4: チャンクの構築
  ┌──────────────────────────┐
  │ グループ化した文を結合   │
  │                          │
  │ min_tokens 以上？        │
  │   NO  → 次と統合         │
  │   YES → チャンク確定     │
  └──────────┬───────────────┘
             │
             v
Step 5: オーバーラップ追加（オプション）
  ┌──────────────────────────┐
  │ overlap_tokens > 0       │
  │   ↓                      │
  │ 前チャンクの末尾N tokens │
  │ を次チャンクの先頭に追加 │
  │                          │
  │ → 文脈の連続性を保持    │
  └──────────────────────────┘
             │
             v
出力: チャンクリスト
  [
    {
      id: "chunk_0",
      text: "...",
      type: "paragraph/sentence_group",
      sentences: [...],
      ...
    },
    ...
  ]
```

### Q/A生成の詳細フロー

```
┌─────────────────────────────────────────────┐
│ Q/A生成プロセス                               │
│ (generation.py::QAGenerator)                │
└─────────────────────────────────────────────┘

入力: chunk (チャンク辞書)

Step 1: チャンク特性分析
  ┌──────────────────────────┐
  │ トークン数カウント          │
  │   ↓                      │
  │ チャンク位置取得         　 │
  │ (前半/後半)               │
  └──────────┬───────────────┘
             │
             v
Step 2: Q/A数決定
  ┌──────────────────────────┐
  │ トークン数に基づく:         │
  │                          │
  │ < 50tk   → 2個           │
  │ 50-100tk → 3個           │
  │ 100-200tk→ 4-5個         │
  │ 200-300tk→ 5-6個         │
  │ > 300tk  → 6-8個         │
  │                          │
  │ 文書後半 (+5以降) → +1     │
  └──────────┬───────────────┘
             │
             v
Step 3: プロンプト構築
  ┌──────────────────────────┐
  │ システムプロンプト:         │
  │ """                      │
  │ あなたは教育コンテンツ       │
  │ 作成の専門家です。          │
  │ ...                      │
  │ """                      │
  │                          │
  │ ユーザープロンプト:         │
  │ """                      │
  │ 以下のテキストから{N}個の    │
  │ Q&Aペアを生成:             │
  │                          │
  │ [チャンクテキスト]          │
  │ """                      │
  └──────────┬───────────────┘
             │
             v
Step 4: LLM API呼び出し
  ┌──────────────────────────┐
  │ Gemini API               │
  │ - モデル: gemini-2.0-flash│
  │ - Temperature: 0.7       │
  │ - JSON出力指定            │
  │                          │
  │ 応答形式:                 │
  │ {                        │
  │   "qa_pairs": [          │
  │     {                    │
  │       "question": "...", │
  │       "answer": "...",   │
  │       ...                │
  │     }                    │
  │   ]                      │
  │ }                        │
  └──────────┬───────────────┘
             │
             v
Step 5: パース＆検証
  ┌──────────────────────────┐
  │ 1. JSON文字列をパース      │
  │ 2. Pydanticモデル検証      │
  │    - QAPairsList         │
  │    - 必須フィールド確認     │
  │                          │
  │ 3. メタデータ追加          │
  │    - chunk_id            │
  │    - dataset_type        │
  │    - generation_model    │
  └──────────┬───────────────┘
             │
             v
出力: Q/Aペアリスト
  [
    {
      question: "...",
      answer: "...",
      chunk_id: "chunk_0",
      question_type: "fact",
      difficulty: "medium",
      ...
    },
    ...
  ]
```

### カバレージ分析の詳細フロー

```
┌─────────────────────────────────────────────┐
│ カバレージ分析プロセス                         │
│ (evaluation.py::analyze_coverage)           │
└─────────────────────────────────────────────┘

入力: chunks, qa_pairs

Step 1: 埋め込みベクトル生成
  ┌──────────────────────────┐
  │ チャンク埋め込み:           │
  │   semantic.py            │
  │   generate_embeddings()  │
  │   - Gemini Embedding API │
  │   - バッチ処理             │
  │                          │
  │ Q/Aペア埋め込み:           │
  │   question + answer      │
  │   - バッチサイズ: 2048     │
  └──────────┬───────────────┘
             │
             v
Step 2: カバレージ行列計算
  ┌──────────────────────────┐
  │ NumPy行列演算:            │
  │                          │
  │ coverage_matrix =        │
  │   np.dot(                │
  │     chunk_embeddings,    │
  │     qa_embeddings.T      │
  │   )                      │
  │                          │
  │ → コサイン類似度行列        │
  │   [N_chunks × N_qa]      │
  └──────────┬───────────────┘
             │
             v
Step 3: 多段階閾値評価
  ┌──────────────────────────┐
  │ データセット別閾値:         │
  │                          │
  │ Wikipedia:               │
  │ - Strict: 0.85           │
  │ - Standard: 0.75         │
  │ - Lenient: 0.65          │
  │                          │
  │ News:                    │
  │ - Strict: 0.80           │
  │ - Standard: 0.70         │
  │ - Lenient: 0.60          │
  │                          │
  │ 各閾値で:                 │
  │ - カバー率計算             │
  │ - 未カバーチャンク特定      │
  └──────────┬───────────────┘
             │
             v
Step 4: チャンク特性別分析
  ┌──────────────────────────┐
  │ 長さ別分析:                │
  │ - Short (<100tk)         │
  │ - Medium (100-200tk)     │
  │ - Long (>200tk)          │
  │                          │
  │ 位置別分析:                │
  │ - Beginning (0-33%)      │
  │ - Middle (33-67%)        │
  │ - End (67-100%)          │
  │                          │
  │ 各カテゴリで:              │
  │ - カバー率                │
  │ - 平均類似度               │
  └──────────┬───────────────┘
             │
             v
Step 5: インサイト生成
  ┌──────────────────────────┐
  │ 低カバレージ箇所特定:        │
  │                          │
  │ "Shortチャンクの           │
  │  カバレージが低い (65%)"    │
  │                          │
  │ "文書後半部分の            │
  │  カバレージが低い (70%)"    │
  └──────────────────────────┘
             │
             v
出力: カバレージ結果
  {
    coverage_rate: 0.85,
    multi_threshold: {
      strict: {...},
      standard: {...},
      lenient: {...}
    },
    chunk_analysis: {
      by_length: {...},
      by_position: {...}
    },
    uncovered_chunks: [...],
    ...
  }
```

---

## 📦 モジュール詳細

### 1. models.py - データモデル

**役割**: Pydanticによるデータ検証とスキーマ定義

**主要クラス**:

```python
# 基本Q/Aペアモデル
class QAPair(BaseModel):
    question: str
    answer: str
    question_type: str = "fact"  # fact/reason/comparison/application
    difficulty: str = "medium"   # easy/medium/hard
    source_span: str = ""

class QAPairsList(BaseModel):
    qa_pairs: List[QAPair]

# 拡張モデル
class EnhancedQAPair(BaseModel):
    question: str
    answer: str

# Chain-of-Thoughtモデル
class ChainOfThoughtQAPair(BaseModel):
    question: str
    answer: str
    reasoning: str
    confidence: float

# 生成要件モデル
class QAGenerationConsiderations(BaseModel):
    document_characteristics: Dict
    extraction_requirements: Dict
    quality_standards: Dict
    qa_characteristics: Dict
```

**使用例**:

```python
from qa_generation.models import QAPair, QAPairsList

# Q/Aペアの作成
qa_pair = QAPair(
    question="機械学習とは何ですか？",
    answer="機械学習はデータからパターンを学習するAI技術です。",
    question_type="fact",
    difficulty="easy"
)

# リスト化
qa_list = QAPairsList(qa_pairs=[qa_pair])

# JSON出力
print(qa_list.model_dump_json(indent=2))
```

---

### 2. pipeline.py - パイプライン制御

**役割**: データ読み込みからQ/A生成、カバレージ分析までの全体制御

**主要クラス**:

```python
class QAPipeline:
    def __init__(
        dataset_name: Optional[str] = None,
        input_file: Optional[str] = None,
        input_chunks: Optional[str] = None,  # チャンクCSV対応
        model: str = "gemini-2.0-flash",
        output_dir: str = "qa_output/pipeline",
        max_docs: Optional[int] = None
    )

    def run(
        use_celery: bool = False,
        celery_workers: int = 8,
        batch_chunks: int = 3,
        merge_chunks: bool = True,
        min_tokens: int = 150,
        max_tokens: int = 400,
        analyze_coverage: bool = False,
        overlap_tokens: int = 0,
        use_similarity: bool = False,
        similarity_threshold: float = 0.7
    ) -> Dict
```

**実行フェーズ**:

1. **データ読み込み** (`load_data()` or `load_chunks_from_csv()`)
2. **チャンク作成** (`create_document_chunks()`)
3. **Q/A生成** (`generate_qa_dataset()`)
4. **カバレージ分析** (`analyze_coverage()`) ※オプション
5. **結果保存** (`save_results()`)

**使用例**:

```python
from qa_generation.pipeline import QAPipeline

# パイプライン初期化
pipeline = QAPipeline(
    dataset_name="wikipedia_ja",
    model="gemini-2.0-flash",
    max_docs=100
)

# 実行
result = pipeline.run(
    use_celery=True,
    celery_workers=16,
    analyze_coverage=True,
    merge_chunks=True
)

print(f"生成Q/A数: {result['qa_count']}")
print(f"カバレージ率: {result['coverage_results']['coverage_rate']:.1%}")
```

---

### 3. structure.py - チャンク作成・統合

**役割**: セマンティック分割によるチャンク作成、小チャンクの統合

**主要関数**:

```python
def create_semantic_chunks(
    text: str,
    lang: str = "ja",
    max_tokens: int = 200,
    chunk_id_prefix: str = "chunk",
    overlap_tokens: int = 0,
    use_similarity: bool = False,
    similarity_threshold: float = 0.7
) -> List[Dict]

def create_document_chunks(
    df: pd.DataFrame,
    dataset_type: str,
    max_docs: Optional[int] = None,
    overlap_tokens: int = 0,
    use_similarity: bool = False,
    similarity_threshold: float = 0.7,
    max_workers: int = 8
) -> List[Dict]

def merge_small_chunks(
    chunks: List[Dict],
    min_tokens: int = 150,
    max_tokens: int = 400
) -> List[Dict]
```

**チャンク作成オプション**:


| オプション             | 説明                         | デフォルト |
| ---------------------- | ---------------------------- | ---------- |
| `max_tokens`           | チャンクの最大トークン数     | 200        |
| `min_tokens`           | チャンクの最小トークン数     | 50         |
| `overlap_tokens`       | 前チャンクとの重複トークン数 | 0          |
| `use_similarity`       | ベクトル類似度分割を使用     | False      |
| `similarity_threshold` | 分割判定の類似度閾値         | 0.7        |
| `prefer_paragraphs`    | 段落優先分割                 | True       |

**使用例**:

```python
from qa_generation.structure import create_semantic_chunks

text = """
機械学習は人工知能の一分野です。
データからパターンを学習します。
深層学習は機械学習の一種です。
"""

chunks = create_semantic_chunks(
    text=text,
    max_tokens=100,
    overlap_tokens=20,
    use_similarity=True,
    similarity_threshold=0.7
)

for chunk in chunks:
    print(f"ID: {chunk['id']}, Tokens: {chunk['tokens']}")
    print(f"Text: {chunk['text'][:50]}...")
```

---

### 4. generation.py - Q/A生成

**役割**: LLMを使用したQ/Aペアの生成

**主要クラス**:

```python
class QAGenerator:
    def __init__(
        client: Optional[LLMClient] = None,
        model: str = "gemini-2.0-flash"
    )

    def determine_qa_count(
        chunk: Dict,
        config: Dict
    ) -> int

    def generate_for_chunk(
        chunk: Dict,
        config: Dict
    ) -> List[Dict]

def generate_qa_dataset(
    chunks: List[Dict],
    config: Dict,
    generator: QAGenerator,
    use_celery: bool = False,
    celery_workers: int = 8,
    batch_chunks: int = 3
) -> List[Dict]
```

**Q/A数決定ロジック**:

```python
トークン数 < 50    → 2個
50 ≤ トークン数 < 100  → 3個
100 ≤ トークン数 < 200 → 4-5個
200 ≤ トークン数 < 300 → 5-6個
トークン数 ≥ 300   → 6-8個

# 文書後半の位置補正
if chunk_idx >= 5:
    qa_count += 1
```

**使用例**:

```python
from qa_generation.generation import QAGenerator

generator = QAGenerator(model="gemini-2.0-flash")

chunk = {
    'id': 'chunk_0',
    'text': '機械学習について...',
    'tokens': 150,
    'chunk_idx': 0
}

config = {
    'lang': 'ja',
    'qa_per_chunk': 3
}

qa_pairs = generator.generate_for_chunk(chunk, config)

for qa in qa_pairs:
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}")
```

---

### 5. semantic.py - セマンティック分析

**役割**: 埋め込みベクトル生成、セマンティック分割、類似度計算

**主要クラス**:

```python
class SemanticCoverage:
    def __init__(
        embedding_model="gemini-embedding-001"
    )

    def create_semantic_chunks(
        document: str,
        max_tokens: int = 200,
        min_tokens: int = 50,
        overlap_tokens: int = 0,
        use_similarity: bool = False,
        similarity_threshold: float = 0.7,
        prefer_paragraphs: bool = True
    ) -> List[Dict]

    def generate_embeddings(
        chunks: List[Dict]
    ) -> np.ndarray

    def generate_embeddings_batch(
        texts: List[str],
        batch_size: int = 2048
    ) -> np.ndarray

    def cosine_similarity(
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float
```

**使用例**:

```python
from qa_generation.semantic import SemanticCoverage

analyzer = SemanticCoverage()

# セマンティック分割
chunks = analyzer.create_semantic_chunks(
    document="長文テキスト...",
    max_tokens=200,
    use_similarity=True,
    similarity_threshold=0.7
)

# 埋め込み生成
embeddings = analyzer.generate_embeddings(chunks)

# 類似度計算
similarity = analyzer.cosine_similarity(
    embeddings[0],
    embeddings[1]
)
```

---

### 6. evaluation.py - カバレージ分析

**役割**: Q/Aペアのカバレージ評価、多段階閾値分析

**主要関数**:

```python
def analyze_coverage(
    chunks: List[Dict],
    qa_pairs: List[Dict],
    dataset_type: str = "wikipedia_ja",
    custom_threshold: Optional[float] = None
) -> Dict

def multi_threshold_coverage(
    coverage_matrix: np.ndarray,
    chunks: List[Dict],
    qa_pairs: List[Dict],
    thresholds: Dict[str, float]
) -> Dict

def analyze_chunk_characteristics_coverage(
    chunks: List[Dict],
    coverage_matrix: np.ndarray,
    qa_pairs: List[Dict],
    threshold: float = 0.7
) -> Dict
```

**閾値設定** (config.py::OPTIMAL_THRESHOLDS):

```python
OPTIMAL_THRESHOLDS = {
    "wikipedia_ja": {
        "strict": 0.85,
        "standard": 0.75,
        "lenient": 0.65
    },
    "cc_news": {
        "strict": 0.80,
        "standard": 0.70,
        "lenient": 0.60
    },
    ...
}
```

**使用例**:

```python
from qa_generation.evaluation import analyze_coverage

coverage = analyze_coverage(
    chunks=chunks,
    qa_pairs=qa_pairs,
    dataset_type="wikipedia_ja"
)

print(f"カバレージ率: {coverage['coverage_rate']:.1%}")
print(f"Strict: {coverage['multi_threshold']['strict']['coverage_rate']:.1%}")
print(f"Standard: {coverage['multi_threshold']['standard']['coverage_rate']:.1%}")
print(f"Lenient: {coverage['multi_threshold']['lenient']['coverage_rate']:.1%}")

# チャンク特性別
chunk_analysis = coverage['chunk_analysis']
print(f"Shortチャンク: {chunk_analysis['by_length']['short']['coverage_rate']:.1%}")
```

---

### 7. content.py - コンテンツ分析

**役割**: キーワード抽出、チャンク複雑度分析

**主要クラス・関数**:

```python
class KeywordExtractor:
    def __init__(
        prefer_mecab: bool = True
    )

    def extract(
        text: str,
        top_n: int = 5,
        use_scoring: bool = True
    ) -> List[str]

def analyze_chunk_complexity(
    chunk_text: str,
    lang: str = "ja"
) -> Dict

def extract_key_concepts(
    chunk_text: str,
    lang: str = "ja",
    top_n: int = 5
) -> List[str]
```

**使用例**:

```python
from qa_generation.content import KeywordExtractor, analyze_chunk_complexity

extractor = KeywordExtractor()

text = "機械学習は人工知能の一分野で、データから..."

# キーワード抽出
keywords = extractor.extract(text, top_n=5)
print(f"キーワード: {keywords}")

# 複雑度分析
complexity = analyze_chunk_complexity(text, lang="ja")
print(f"複雑度レベル: {complexity['complexity_level']}")
print(f"専門用語: {complexity['technical_terms']}")
```

---

### 8. data_io.py - データ入出力

**役割**: ファイル読み込み、結果保存

**主要関数**:

```python
def load_uploaded_file(
    file_path: str
) -> pd.DataFrame

def load_preprocessed_data(
    dataset_type: str
) -> pd.DataFrame

def save_results(
    qa_pairs: List[Dict],
    coverage_results: Dict,
    dataset_type: str,
    output_dir: str = "qa_output/a02"
) -> Dict[str, str]
```

**対応ファイル形式**:

- CSV (.csv)
- テキスト (.txt, .text)
- JSON (.json)
- JSON Lines (.jsonl)

**使用例**:

```python
from qa_generation.data_io import load_uploaded_file, save_results

# ファイル読み込み
df = load_uploaded_file("data.csv")

# 結果保存
saved_files = save_results(
    qa_pairs=qa_pairs,
    coverage_results=coverage_results,
    dataset_type="wikipedia_ja"
)

print(f"Q/A CSV: {saved_files['qa_csv']}")
print(f"サマリー: {saved_files['summary']}")
```

---

### 9. config.py - 設定管理

**役割**: データセット別の最適閾値、拡張設定

**主要定義**:

```python
# データセット別最適閾値
OPTIMAL_THRESHOLDS = {
    "cc_news": {
        "strict": 0.80,
        "standard": 0.70,
        "lenient": 0.60
    },
    "wikipedia_ja": {
        "strict": 0.85,
        "standard": 0.75,
        "lenient": 0.65
    },
    ...
}

# ローカル拡張設定
LOCAL_DATASET_EXTENSIONS = {
    "cc_news": {
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en",
    },
    ...
}
```

---

## 🚀 使用方法

### 基本的な使用方法

#### 1. データセットからQ/A生成

```python
from qa_generation.pipeline import QAPipeline

# パイプライン初期化
pipeline = QAPipeline(
    dataset_name="wikipedia_ja",
    model="gemini-2.0-flash",
    max_docs=100
)

# 実行
result = pipeline.run(
    analyze_coverage=True
)

print(f"生成Q/A数: {result['qa_count']}")
print(f"カバレージ率: {result['coverage_results']['coverage_rate']:.1%}")
```

#### 2. ローカルファイルからQ/A生成

```python
pipeline = QAPipeline(
    input_file="data.csv",
    model="gemini-2.0-flash"
)

result = pipeline.run()
```

#### 3. チャンクCSVからQ/A生成

```python
pipeline = QAPipeline(
    input_chunks="chunks.csv",
    model="gemini-2.0-flash"
)

result = pipeline.run(
    merge_chunks=True,
    min_tokens=150,
    max_tokens=400
)
```

### 高度な使用方法

#### Celery並列処理

```python
pipeline = QAPipeline(
    dataset_name="wikipedia_ja",
    max_docs=1000
)

result = pipeline.run(
    use_celery=True,
    celery_workers=24,
    batch_chunks=3
)
```

#### セマンティック分割のカスタマイズ

```python
result = pipeline.run(
    overlap_tokens=50,           # 50トークンのオーバーラップ
    use_similarity=True,         # ベクトル類似度分割
    similarity_threshold=0.6,    # 類似度閾値
    merge_chunks=True            # 小チャンク統合
)
```

---

## 📖 パラメータリファレンス

### QAPipeline.run() パラメータ


| パラメータ             | 型    | デフォルト | 説明                                 |
| ---------------------- | ----- | ---------- | ------------------------------------ |
| `use_celery`           | bool  | False      | Celery並列処理を使用                 |
| `celery_workers`       | int   | 8          | Celeryワーカー数                     |
| `batch_chunks`         | int   | 3          | 1回のAPI呼び出しで処理するチャンク数 |
| `merge_chunks`         | bool  | True       | 小チャンクを統合                     |
| `min_tokens`           | int   | 150        | 統合対象の最小トークン数             |
| `max_tokens`           | int   | 400        | 統合後の最大トークン数               |
| `analyze_coverage`     | bool  | False      | カバレージ分析を実行                 |
| `overlap_tokens`       | int   | 0          | チャンク間のオーバーラップトークン数 |
| `use_similarity`       | bool  | False      | ベクトル類似度分割を使用             |
| `similarity_threshold` | float | 0.7        | 類似度分割の閾値                     |

### チャンク作成パラメータ


| パラメータ             | 説明                     | 推奨値     |
| ---------------------- | ------------------------ | ---------- |
| `max_tokens`           | チャンクの最大トークン数 | 200-400    |
| `min_tokens`           | チャンクの最小トークン数 | 50-150     |
| `overlap_tokens`       | 前チャンクとの重複       | 0-100      |
| `use_similarity`       | 類似度分割               | True/False |
| `similarity_threshold` | 分割閾値                 | 0.6-0.8    |
| `prefer_paragraphs`    | 段落優先                 | True       |

### Q/A生成パラメータ


| パラメータ     | 説明                      | 推奨値           |
| -------------- | ------------------------- | ---------------- |
| `model`        | 使用するLLMモデル         | gemini-2.0-flash |
| `qa_per_chunk` | チャンクあたりの基本Q/A数 | 3-5              |
| `batch_chunks` | バッチ処理するチャンク数  | 3                |

---

## 🔄 実行例とワークフロー

### ワークフロー1: データセットからQ/A生成

```bash
cd /path/to/project

python -c "
from qa_generation.pipeline import QAPipeline

pipeline = QAPipeline(
    dataset_name='wikipedia_ja',
    model='gemini-2.0-flash',
    max_docs=100
)

result = pipeline.run(
    use_celery=True,
    celery_workers=16,
    analyze_coverage=True,
    merge_chunks=True
)

print(f'生成Q/A数: {result[\"qa_count\"]}')
print(f'カバレージ率: {result[\"coverage_results\"][\"coverage_rate\"]:.1%}')
"
```

### ワークフロー2: CSVファイルからQ/A生成

```bash
# Step 1: チャンク作成
python -m chunking.csv_to_chunks_text_para \
  -i data.csv \
  -o chunks.csv \
  --text-column "Combined_Text"

# Step 2: Q/A生成
python -c "
from qa_generation.pipeline import QAPipeline

pipeline = QAPipeline(
    input_chunks='chunks.csv',
    model='gemini-2.0-flash'
)

result = pipeline.run(
    merge_chunks=True,
    analyze_coverage=True
)

print(f'保存先: {result[\"saved_files\"][\"qa_csv\"]}')
"
```

### ワークフロー3: カスタマイズしたQ/A生成

```python
from qa_generation.pipeline import QAPipeline

# 高度な設定でパイプライン実行
pipeline = QAPipeline(
    dataset_name="wikipedia_ja",
    model="gemini-2.0-flash",
    output_dir="qa_output/custom",
    max_docs=500
)

result = pipeline.run(
    # チャンク作成設定
    overlap_tokens=50,           # 50トークンのオーバーラップ
    use_similarity=True,         # ベクトル類似度分割
    similarity_threshold=0.6,    # 低めの閾値（より細かく分割）
    merge_chunks=True,           # 小チャンク統合
    min_tokens=150,              # 統合対象の最小トークン数
    max_tokens=400,              # 統合後の最大トークン数

    # Q/A生成設定
    use_celery=True,             # Celery並列処理
    celery_workers=24,           # 24ワーカー
    batch_chunks=3,              # 3チャンクずつバッチ処理

    # カバレージ分析
    analyze_coverage=True
)

# 結果確認
print("\n" + "=" * 60)
print("実行結果")
print("=" * 60)
print(f"生成Q/A数: {result['qa_count']}")
print(f"保存先: {result['saved_files']['qa_csv']}")

if 'coverage_results' in result:
    cov = result['coverage_results']
    print(f"\nカバレージ分析:")
    print(f"  Standard: {cov['coverage_rate']:.1%}")
    print(f"  Strict:   {cov['multi_threshold']['strict']['coverage_rate']:.1%}")
    print(f"  Lenient:  {cov['multi_threshold']['lenient']['coverage_rate']:.1%}")
```

---

## ⚠️ トラブルシューティング

### 問題1: MeCabが利用できない

**症状**:

```
⚠️ MeCabが利用できません（正規表現モード）
```

**対処法**:

MeCabなしでも動作しますが、インストールすると品質が向上します。

```bash
# Ubuntu/Debian
sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8
pip install mecab-python3

# macOS
brew install mecab mecab-ipadic
pip install mecab-python3
```

---

### 問題2: 埋め込み生成エラー

**症状**:

```
埋め込みベクトル生成エラー: API rate limit exceeded
```

**対処法**:

```python
# バッチサイズを小さくする
from qa_generation.semantic import SemanticCoverage

analyzer = SemanticCoverage()
embeddings = analyzer.generate_embeddings_batch(
    texts,
    batch_size=100  # デフォルト2048から減らす
)
```

---

### 問題3: メモリ不足

**症状**:

```
MemoryError: Unable to allocate array
```

**対処法**:

```python
# 処理する文書数を制限
pipeline = QAPipeline(
    dataset_name="wikipedia_ja",
    max_docs=100  # 制限を設定
)

# またはバッチサイズを小さく
result = pipeline.run(
    batch_chunks=1,  # 1チャンクずつ処理
    celery_workers=4  # ワーカー数を減らす
)
```

---

### 問題4: Celeryワーカーが起動していない

**症状**:

```
RuntimeError: Celery workers are not running
```

**対処法**:

```bash
# Celeryワーカーを起動
celery -A celery_tasks worker --loglevel=info -c 16

# または、Celeryなしで実行
```

```python
result = pipeline.run(
    use_celery=False  # Celeryを使用しない
)
```

---

### 問題5: チャンクCSVの形式エラー

**症状**:

```
ValueError: 必須カラムが不足しています: ['chunk_id', 'text']
```

**対処法**:

チャンクCSVに以下のカラムが必要です：

```csv
chunk_id,text,tokens,chunk_idx,dataset_type
chunk_0,"テキスト...",150,0,dataset_name
```

正しい形式で作成し直してください：

```bash
python -m chunking.csv_to_chunks_text_para \
  -i input.txt \
  -o chunks.csv
```

---

### 問題6: Q/A生成の品質が低い

**対処法**:

```python
# より大きなチャンクを使用
result = pipeline.run(
    merge_chunks=True,
    min_tokens=200,  # より大きなチャンク
    max_tokens=500
)

# オーバーラップを追加
result = pipeline.run(
    overlap_tokens=100  # 文脈を保持
)
```

---

## 💡 ベストプラクティス

### 1. チャンクサイズの最適化

```python
# データセット別推奨設定

# Wikipedia（専門的な内容）
pipeline.run(
    merge_chunks=True,
    min_tokens=200,  # 大きめのチャンク
    max_tokens=500,
    overlap_tokens=50
)

# News（短い記事）
pipeline.run(
    merge_chunks=True,
    min_tokens=150,  # 標準サイズ
    max_tokens=400,
    overlap_tokens=30
)

# 対話データ（短いやり取り）
pipeline.run(
    merge_chunks=True,
    min_tokens=100,  # 小さめのチャンク
    max_tokens=300,
    overlap_tokens=20
)
```

---

### 2. 並列処理の最適化

```python
import multiprocessing

# CPUコア数を取得
cpu_count = multiprocessing.cpu_count()

# 推奨: コア数の75%
workers = int(cpu_count * 0.75)

pipeline.run(
    use_celery=True,
    celery_workers=workers
)
```

---

### 3. カバレージ分析の活用

```python
result = pipeline.run(analyze_coverage=True)

coverage = result['coverage_results']

# 未カバーチャンクを確認
uncovered = coverage['uncovered_chunks']
if uncovered:
    print(f"未カバーチャンク数: {len(uncovered)}")

    # 最も低いカバレージのチャンクを表示
    sorted_uncovered = sorted(
        uncovered,
        key=lambda x: x['similarity']
    )

    for chunk in sorted_uncovered[:5]:
        print(f"類似度: {chunk['similarity']:.2f}")
        print(f"テキスト: {chunk['chunk']['text'][:100]}...")
```

---

### 4. 段階的な実験

```python
# Step 1: 小規模テスト（10文書）
pipeline = QAPipeline(
    dataset_name="wikipedia_ja",
    max_docs=10
)
result = pipeline.run()

# Step 2: 中規模テスト（100文書）
pipeline = QAPipeline(
    dataset_name="wikipedia_ja",
    max_docs=100
)
result = pipeline.run(use_celery=True)

# Step 3: 本番実行（全データ）
pipeline = QAPipeline(
    dataset_name="wikipedia_ja"
)
result = pipeline.run(
    use_celery=True,
    celery_workers=24
)
```

---

### 5. セマンティック分割の活用

```python
# トピック境界を自動検出
result = pipeline.run(
    use_similarity=True,
    similarity_threshold=0.6,  # 低め（より細かく分割）
    overlap_tokens=50          # 文脈保持
)

# 専門文書の場合
result = pipeline.run(
    use_similarity=True,
    similarity_threshold=0.7,  # 高め（トピック境界を明確に）
    overlap_tokens=100         # より多くの文脈
)
```

---

### 6. カスタム閾値の設定

```python
from qa_generation.evaluation import analyze_coverage

# データセットに最適な閾値を実験的に決定
thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]

for threshold in thresholds:
    coverage = analyze_coverage(
        chunks=chunks,
        qa_pairs=qa_pairs,
        dataset_type="custom",
        custom_threshold=threshold
    )
    print(f"閾値 {threshold}: カバレージ {coverage['coverage_rate']:.1%}")
```

---

### 7. ログの活用

```python
import logging

# 詳細ログを有効化
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_generation.log'),
        logging.StreamHandler()
    ]
)

pipeline = QAPipeline(...)
result = pipeline.run(...)

# ログファイルで処理の詳細を確認
# grep "チャンク作成完了" qa_generation.log
# grep "Q/A生成完了" qa_generation.log
```

---

## 📈 パフォーマンス指標

### チャンク作成の処理時間


| 文書数 | ワーカー数 | 処理時間 | チャンク数 |
| ------ | ---------- | -------- | ---------- |
| 100    | 8          | ~2分     | ~300       |
| 1,000  | 16         | ~15分    | ~3,000     |
| 10,000 | 24         | ~2時間   | ~30,000    |

### Q/A生成の処理時間


| チャンク数 | Celery | ワーカー数 | 処理時間 | Q/A数   |
| ---------- | ------ | ---------- | -------- | ------- |
| 100        | No     | -          | ~15分    | ~300    |
| 100        | Yes    | 8          | ~5分     | ~300    |
| 1,000      | Yes    | 16         | ~30分    | ~3,000  |
| 10,000     | Yes    | 24         | ~4時間   | ~30,000 |

### カバレージ分析の処理時間


| チャンク数 | Q/A数  | 処理時間 |
| ---------- | ------ | -------- |
| 100        | 300    | ~30秒    |
| 1,000      | 3,000  | ~3分     |
| 10,000     | 30,000 | ~20分    |

---

## 🎓 Tips & Tricks

### Tip 1: チャンク品質の確認

```python
from qa_generation.structure import create_document_chunks
import pandas as pd

chunks = create_document_chunks(df, "wikipedia_ja")

# トークン分布を確認
import matplotlib.pyplot as plt

tokens = [c['tokens'] for c in chunks]
plt.hist(tokens, bins=30)
plt.xlabel('Tokens')
plt.ylabel('Frequency')
plt.title('Chunk Token Distribution')
plt.savefig('token_distribution.png')
```

---

### Tip 2: Q/A多様性の確認

```python
from collections import Counter

# 質問タイプの分布
question_types = [qa['question_type'] for qa in qa_pairs]
type_counts = Counter(question_types)
print(f"質問タイプ分布: {type_counts}")

# 難易度分布
difficulties = [qa['difficulty'] for qa in qa_pairs]
diff_counts = Counter(difficulties)
print(f"難易度分布: {diff_counts}")
```

---

### Tip 3: バッチサイズの最適化

```python
# メモリ使用量とのトレードオフ

# メモリ潤沢（32GB以上）
batch_chunks=5

# 標準（16GB）
batch_chunks=3

# メモリ制限あり（8GB以下）
batch_chunks=1
```

---

## 📞 サポート

### 問題が発生した場合

1. **ログを確認**

   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
2. **小規模テストを実行**

   ```python
   pipeline = QAPipeline(max_docs=10)
   ```
3. **パラメータを調整**

   - チャンクサイズ
   - バッチサイズ
   - ワーカー数
4. **Githubでissueを作成**

---

## 📝 バージョン履歴


| バージョン | 日付       | 変更内容                    |
| ---------- | ---------- | --------------------------- |
| 1.0.0      | 2025-01-11 | 初版リリース                |
| 1.1.0      | 2025-01-11 | セマンティック分割機能追加  |
| 1.2.0      | 2025-01-11 | 多段階カバレージ分析追加    |
| 1.3.0      | 2025-01-17 | チャンクCSV読み込み機能追加 |

---

## 📄 関連ドキュメント

- `qa_qdrant/doc/make_qa_register_qdrant.md` - Q/A生成からQdrant登録までの統合ツール
- `chunking/SKILL.md` - チャンク処理の技術詳細
- `helper/helper_llm.md` - LLMクライアントの使用方法
- `helper/helper_embedding.md` - Embeddingクライアントの使用方法

---

**作成日**: 2025-01-17
**最終更新**: 2025-01-17
**ドキュメントバージョン**: 1.3.0
