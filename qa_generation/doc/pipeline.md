# pipeline.py 完全ガイド

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

`qa_generation/pipeline.py` は、**データ読み込みからQ/A生成、カバレージ分析、結果保存までを一貫して実行するパイプライン制御モジュール**です。3つの入力モードに対応し、Celeryによる並列処理とセマンティック分割による高品質なチャンク作成を実現します。

### 主な特徴

✅ **3つの入力モード**
- **データセットモード**: 事前定義されたデータセット（DATASET_CONFIGS）
- **ファイルモード**: ローカルCSV/テキストファイル
- **チャンクCSVモード**: 事前作成されたチャンクCSV（✨新機能）

✅ **柔軟なチャンク処理**
- セマンティック分割による高品質チャンク
- オーバーラップ機能（文脈保持）
- ベクトル類似度分割オプション
- 並列チャンク作成（ThreadPoolExecutor）

✅ **Celery並列処理対応**
- 大規模データセットの高速処理
- 最大24ワーカーでの並列実行
- バッチサイズの柔軟な調整

✅ **包括的なカバレージ分析**
- Q/Aペアによるチャンクのカバー率測定
- 未カバーチャンクの特定
- 多段階閾値評価オプション

✅ **統合されたワークフロー**
- 1つのメソッド（run）で全処理実行
- エラーハンドリングとログ出力
- 結果の自動保存（JSON/CSV）

---

## 🏗️ システムアーキテクチャ

### クラス構成

```
QAPipeline
├── __init__()                    # 初期化、入力検証
├── _validate_inputs()            # 入力の排他制御
├── _load_config()                # 設定ロード
│
├── load_data()                   # データ読み込み（データセット/ファイル）
├── load_chunks_from_csv()        # チャンクCSV読み込み ✨新機能
├── _split_sentences()            # 文分割ユーティリティ
│
├── create_chunks()               # チャンク作成（並列対応）
│
├── generate_qa()                 # Q/A生成（Celery/逐次）
│   ├── _generate_with_celery()   # Celery並列処理
│   └── _generate_sync()          # 逐次処理
│
├── evaluate_coverage()           # カバレージ分析
│
├── save()                        # 結果保存（JSON/CSV）
│
└── run()                         # パイプライン実行 ⭐メイン
```

### 依存関係

```
QAPipeline
├── qa_generation.data_io
│   ├── load_uploaded_file()         # ローカルファイル読み込み
│   ├── load_preprocessed_data()     # 前処理済みデータ読み込み
│   └── save_results()               # 結果保存
│
├── qa_generation.structure
│   ├── create_document_chunks()     # チャンク作成（並列）
│   └── merge_small_chunks()         # 小チャンク統合
│
├── qa_generation.generation
│   ├── QAGenerator                  # Q/A生成クラス
│   └── generate_qa_dataset()        # データセット全体のQ/A生成
│
├── qa_generation.evaluation
│   └── analyze_coverage()           # カバレージ分析
│
├── celery_tasks
│   ├── submit_unified_qa_generation() # Celeryタスク投入
│   ├── collect_results()              # 結果収集
│   └── check_celery_workers()         # ワーカー確認
│
├── helper.helper_llm
│   └── LLMClient                    # LLM操作
│
└── config
    ├── DATASET_CONFIGS              # データセット定義
    └── LOCAL_DATASET_EXTENSIONS     # ローカル拡張設定
```

### レイヤー構成

```
┌──────────────────────────────────────────┐
│  Application Layer (QAPipeline)          │  ← パイプライン制御
│  - run()                                 │
├──────────────────────────────────────────┤
│  Business Logic Layer                    │
│  ├─ データ読み込み (data_io)             │
│  ├─ チャンク作成 (structure)             │
│  ├─ Q/A生成 (generation)                 │
│  ├─ カバレージ分析 (evaluation)          │
│  └─ 結果保存 (data_io)                   │
├──────────────────────────────────────────┤
│  Infrastructure Layer                    │
│  ├─ Celery (並列処理)                    │
│  ├─ LLMClient (Gemini API)               │
│  └─ SemanticCoverage (埋め込み生成)      │
└──────────────────────────────────────────┘
```

---

## 🔄 データ処理フロー

### 全体フロー

```
入力
├─ dataset_name → DATASET_CONFIGS
├─ input_file   → CSV/テキストファイル
└─ input_chunks → チャンクCSV ✨新機能
   │
   v
┌──────────────────────────────────────────┐
│ QAPipeline.run()                         │
├──────────────────────────────────────────┤
│                                          │
│ [1/4] データ読み込み                     │
│   ├─ load_data() (dataset/file)         │
│   └─ load_chunks_from_csv() (chunks)    │
│                                          │
│ [2/4] チャンク作成/読み込み              │
│   ├─ create_chunks()                     │
│   │   ├─ セマンティック分割              │
│   │   ├─ オーバーラップ                  │
│   │   └─ 並列処理                        │
│   └─ load_chunks_from_csv() (スキップ)  │
│                                          │
│ [3/4] Q/A生成                            │
│   ├─ generate_qa()                       │
│   │   ├─ Celery並列 (_generate_with_celery) │
│   │   └─ 逐次処理 (_generate_sync)      │
│   ├─ チャンクマージ                      │
│   └─ バッチ処理                          │
│                                          │
│ [4/4] カバレージ分析 & 結果保存          │
│   ├─ evaluate_coverage()                 │
│   └─ save()                              │
│       ├─ JSON保存                        │
│       └─ CSV保存                         │
│                                          │
└──────────────────────────────────────────┘
   │
   v
結果
├─ qa_pairs_{dataset}_{timestamp}.json
├─ qa_pairs_{dataset}_{timestamp}.csv
└─ coverage_{dataset}_{timestamp}.json
```

### 入力モード別フロー

#### モード1: データセットモード

```python
QAPipeline(dataset_name="wikipedia_ja")
   │
   v
_load_config()
   ├─ DATASET_CONFIGS["wikipedia_ja"]
   └─ LOCAL_DATASET_EXTENSIONS（マージ）
   │
   v
load_data()
   └─ load_preprocessed_data()
   │
   v
create_chunks()
   └─ セマンティック分割
   │
   v
generate_qa()
   │
   v
結果保存
```

#### モード2: ファイルモード

```python
QAPipeline(input_file="data.csv")
   │
   v
_load_config()
   └─ 動的設定生成（ファイル名から推測）
   │
   v
load_data()
   └─ load_uploaded_file()
   │
   v
create_chunks()
   │
   v
generate_qa()
   │
   v
結果保存
```

#### モード3: チャンクCSVモード ✨新機能

```python
QAPipeline(input_chunks="chunks.csv")
   │
   v
_load_config()
   └─ 動的設定生成（チャンクCSVファイル名から）
   │
   v
load_chunks_from_csv()
   ├─ CSV読み込み
   ├─ 必須カラム検証（chunk_id, text, tokens, chunk_idx）
   ├─ 既存形式に変換
   └─ センテンス情報の再生成
   │
   v
generate_qa() ← チャンク作成をスキップ
   │
   v
結果保存
```

---

## 📚 モジュール詳細

### `__init__(dataset_name, input_file, input_chunks, model, output_dir, max_docs, client)`

**シグネチャ**:
```python
def __init__(self,
             dataset_name: Optional[str] = None,
             input_file: Optional[str] = None,
             input_chunks: Optional[str] = None,  # ✨新規
             model: str = "gemini-2.0-flash",
             output_dir: str = "qa_output/pipeline",
             max_docs: Optional[int] = None,
             client: Optional[LLMClient] = None)
```

**パラメータ**:

| パラメータ | 型 | デフォルト | 説明 |
|----------|---|----------|------|
| `dataset_name` | Optional[str] | None | 事前定義データセット名 |
| `input_file` | Optional[str] | None | ローカルファイルパス |
| `input_chunks` | Optional[str] | None | チャンクCSVパス ✨ |
| `model` | str | "gemini-2.0-flash" | 使用モデル |
| `output_dir` | str | "qa_output/pipeline" | 出力ディレクトリ |
| `max_docs` | Optional[int] | None | 最大処理文書数 |
| `client` | Optional[LLMClient] | None | LLMクライアント |

**処理**:
```python
1. パラメータ保存
2. 入力検証（_validate_inputs）
   - 3つの入力モードは排他的
   - いずれか1つのみ指定可能
3. 設定ロード（_load_config）
```

**使用例**:
```python
# モード1: データセット
pipeline = QAPipeline(dataset_name="wikipedia_ja")

# モード2: ファイル
pipeline = QAPipeline(input_file="data.csv")

# モード3: チャンクCSV
pipeline = QAPipeline(input_chunks="chunks.csv")

# カスタム設定
pipeline = QAPipeline(
    dataset_name="wikipedia_ja",
    model="gemini-2.0-flash",
    output_dir="my_output",
    max_docs=100
)
```

**エラーハンドリング**:
```python
try:
    pipeline = QAPipeline(dataset_name="wikipedia_ja", input_file="data.csv")
except ValueError as e:
    # "dataset_name, input_file, input_chunks は同時に指定できません"
    print(f"入力エラー: {e}")
```

---

### `_validate_inputs()`

**目的**: 3つの入力モードの排他制御

**処理**:
```python
inputs = [self.dataset_name, self.input_file, self.input_chunks]
non_none_count = sum(1 for x in inputs if x is not None)

if non_none_count == 0:
    raise ValueError(
        "dataset_name, input_file, input_chunks のいずれか1つを指定してください"
    )

if non_none_count > 1:
    raise ValueError(
        "dataset_name, input_file, input_chunks は同時に指定できません"
    )
```

**エラー例**:
```python
# ❌ エラー: 何も指定していない
pipeline = QAPipeline()
# → ValueError: "...のいずれか1つを指定してください"

# ❌ エラー: 複数指定
pipeline = QAPipeline(dataset_name="wikipedia_ja", input_file="data.csv")
# → ValueError: "...は同時に指定できません"

# ✅ 正しい
pipeline = QAPipeline(dataset_name="wikipedia_ja")
```

---

### `_load_config()`

**目的**: 入力モードに応じた設定ロード

**処理フロー**:
```python
if self.input_chunks:
    # チャンクCSVモード
    chunk_path = Path(self.input_chunks)
    dataset_type = chunk_path.stem.replace('_chunks', '')

    return {
        "name": f"チャンクCSV ({chunk_path.name})",
        "text_column": "text",
        "title_column": None,
        "lang": "ja",
        "chunk_size": 300,
        "qa_per_chunk": 3,
        "type": dataset_type
    }

elif self.input_file:
    # ファイルモード
    file_basename = Path(self.input_file).stem
    return {
        "name": f"ローカルファイル ({file_basename})",
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja",
        "chunk_size": 300,
        "qa_per_chunk": 3,
        "type": "custom_upload"
    }

elif self.dataset_name:
    # データセットモード
    config = DATASET_CONFIGS[self.dataset_name].copy()
    if self.dataset_name in LOCAL_DATASET_EXTENSIONS:
        config.update(LOCAL_DATASET_EXTENSIONS[self.dataset_name])
    config["type"] = self.dataset_name
    return config
```

**設定例**:
```python
# チャンクCSVモード
# input_chunks="wikipedia_chunks.csv"
{
    "name": "チャンクCSV (wikipedia_chunks.csv)",
    "text_column": "text",
    "lang": "ja",
    "chunk_size": 300,
    "qa_per_chunk": 3,
    "type": "wikipedia"
}

# データセットモード
# dataset_name="wikipedia_ja"
{
    "name": "Wikipedia日本語",
    "file": "data/wikipedia_ja.csv",
    "text_column": "Combined_Text",
    "lang": "ja",
    "chunk_size": 300,
    "qa_per_chunk": 3,
    "type": "wikipedia_ja"
}
```

---

### `load_data()`

**シグネチャ**:
```python
def load_data(self) -> pd.DataFrame
```

**目的**: データセットまたはファイルからデータを読み込む

**処理フロー**:
```python
if self.input_file:
    # ローカルファイル読み込み
    df = load_uploaded_file(self.input_file)

    if self.max_docs and len(df) > self.max_docs:
        df = df.head(self.max_docs)
        logger.info(f"最大文書数制限: {self.max_docs}件")

    return df
else:
    # データセット読み込み
    df = load_preprocessed_data(self.config)

    if self.max_docs and len(df) > self.max_docs:
        df = df.head(self.max_docs)
        logger.info(f"最大文書数制限: {self.max_docs}件")

    return df
```

**使用例**:
```python
pipeline = QAPipeline(input_file="data.csv", max_docs=100)
df = pipeline.load_data()
print(f"読み込み件数: {len(df)}")
```

---

### `load_chunks_from_csv()` ✨新機能

**シグネチャ**:
```python
def load_chunks_from_csv(self) -> List[Dict]
```

**目的**: 事前作成されたチャンクCSVを読み込んで既存形式に変換

**必須カラム**:
- `chunk_id`: チャンクID
- `text`: テキスト内容
- `tokens`: トークン数
- `chunk_idx`: チャンク位置

**オプショナルカラム**:
- `type`: チャンク種別
- `dataset_type`: データセット種別
- `sentence_count`: 文数
- `source_file`: ソースファイル名
- `doc_id`: 文書ID
- `doc_idx`: 文書位置

**処理フロー**:
```python
1. ファイル存在確認
   if not Path(self.input_chunks).exists():
       raise FileNotFoundError()

2. CSV読み込み
   df = pd.read_csv(self.input_chunks)

3. 必須カラム検証
   required_cols = ['chunk_id', 'text', 'tokens', 'chunk_idx']
   missing_cols = [c for c in required_cols if c not in df.columns]
   if missing_cols:
       raise ValueError(f"必須カラムが不足: {missing_cols}")

4. 既存形式に変換
   for idx, row in df.iterrows():
       sentences = self._split_sentences(str(row['text']))
       chunk = {
           'id': str(row['chunk_id']),
           'text': str(row['text']),
           'tokens': int(row['tokens']),
           'chunk_idx': int(row['chunk_idx']),
           'type': str(row.get('type', 'llm_chunk')),
           'dataset_type': str(row.get('dataset_type', 'custom')),
           'sentences': sentences,
           'sentence_count': int(row.get('sentence_count', len(sentences))),
           'source_file': str(row.get('source_file', '')),
           'doc_id': str(row.get('doc_id', f"doc_{idx}")),
           'doc_idx': int(row.get('doc_idx', 0))
       }
       chunks.append(chunk)

5. 統計情報ログ出力
```

**使用例**:
```python
pipeline = QAPipeline(input_chunks="chunks.csv")
chunks = pipeline.load_chunks_from_csv()

print(f"チャンク数: {len(chunks)}")
print(f"最初のチャンク: {chunks[0]}")
```

**チャンクCSV例**:
```csv
chunk_id,text,tokens,chunk_idx,type,dataset_type
chunk_0,"機械学習は...",150,0,llm_chunk,wikipedia_ja
chunk_1,"深層学習は...",200,1,llm_chunk,wikipedia_ja
```

**エラーハンドリング**:
```python
try:
    chunks = pipeline.load_chunks_from_csv()
except FileNotFoundError:
    print("チャンクファイルが見つかりません")
except ValueError as e:
    print(f"必須カラム不足: {e}")
```

---

### `_split_sentences(text)`

**シグネチャ**:
```python
def _split_sentences(self, text: str) -> List[str]
```

**目的**: テキストを文に分割（簡易版）

**処理**:
```python
import re

# 句点・疑問符・感嘆符で分割
sentences = re.findall(r'[^。．.！？!?]+[。．.！？!?]\s*', text)

# マッチしなかった残りのテキストを追加
if sentences:
    covered_text = ''.join(sentences)
    remaining = text[len(covered_text):].strip()
    if remaining:
        sentences.append(remaining)
else:
    sentences = [text]

return sentences
```

**使用例**:
```python
text = "これは文1です。これは文2です。残りのテキスト"
sentences = pipeline._split_sentences(text)
# → ["これは文1です。", "これは文2です。", "残りのテキスト"]
```

---

### `create_chunks(df, overlap_tokens, use_similarity, similarity_threshold, max_workers)`

**シグネチャ**:
```python
def create_chunks(self, df: pd.DataFrame,
                  overlap_tokens: int = 0,
                  use_similarity: bool = False,
                  similarity_threshold: float = 0.7,
                  max_workers: int = 8) -> List[Dict]
```

**目的**: データフレームからチャンクを作成（並列処理対応）

**パラメータ**:

| パラメータ | 型 | デフォルト | 説明 |
|----------|---|----------|------|
| `df` | pd.DataFrame | - | 入力データ（必須） |
| `overlap_tokens` | int | 0 | チャンク間のオーバーラップトークン数 |
| `use_similarity` | bool | False | ベクトル類似度分割を使用 |
| `similarity_threshold` | float | 0.7 | 類似度分割の閾値 |
| `max_workers` | int | 8 | 並列ワーカー数 |

**処理**:
```python
chunks = create_document_chunks(
    df,
    dataset_type=self.config["type"],
    chunk_size=self.config["chunk_size"],
    text_column=self.config["text_column"],
    title_column=self.config.get("title_column"),
    overlap_tokens=overlap_tokens,
    use_similarity=use_similarity,
    similarity_threshold=similarity_threshold,
    max_workers=max_workers
)
```

**使用例**:
```python
df = pipeline.load_data()

# 基本的なチャンク作成
chunks = pipeline.create_chunks(df)

# オーバーラップ付き
chunks = pipeline.create_chunks(df, overlap_tokens=50)

# セマンティック分割
chunks = pipeline.create_chunks(
    df,
    use_similarity=True,
    similarity_threshold=0.8
)

# 並列処理（16ワーカー）
chunks = pipeline.create_chunks(df, max_workers=16)
```

---

### `generate_qa(chunks, use_celery, celery_workers, batch_chunks, merge_chunks, min_tokens, max_tokens)`

**シグネチャ**:
```python
def generate_qa(self, chunks: List[Dict],
                use_celery: bool = False,
                celery_workers: int = 8,
                batch_chunks: int = 3,
                merge_chunks: bool = True,
                min_tokens: int = 150,
                max_tokens: int = 400) -> List[Dict]
```

**目的**: チャンクからQ/Aペアを生成（Celery並列 or 逐次）

**パラメータ**:

| パラメータ | 型 | デフォルト | 説明 |
|----------|---|----------|------|
| `chunks` | List[Dict] | - | チャンクリスト（必須） |
| `use_celery` | bool | False | Celery並列処理を使用 |
| `celery_workers` | int | 8 | Celeryワーカー数 |
| `batch_chunks` | int | 3 | バッチサイズ |
| `merge_chunks` | bool | True | 小チャンク統合フラグ |
| `min_tokens` | int | 150 | マージ最小トークン数 |
| `max_tokens` | int | 400 | マージ最大トークン数 |

**処理フロー**:
```python
if use_celery:
    # Celery並列処理
    qa_pairs = self._generate_with_celery(
        chunks, celery_workers, batch_chunks, merge_chunks, min_tokens, max_tokens
    )
else:
    # 逐次処理
    qa_pairs = self._generate_sync(
        chunks, batch_chunks, merge_chunks, min_tokens, max_tokens
    )

return qa_pairs
```

**使用例**:
```python
# 逐次処理（小規模）
qa_pairs = pipeline.generate_qa(chunks)

# Celery並列処理（大規模）
qa_pairs = pipeline.generate_qa(
    chunks,
    use_celery=True,
    celery_workers=16,
    batch_chunks=5
)

# チャンクマージなし
qa_pairs = pipeline.generate_qa(
    chunks,
    merge_chunks=False
)
```

---

### `_generate_with_celery(chunks, workers, batch_size, merge_chunks, min_tokens, max_tokens)`

**目的**: Celeryを使用した並列Q/A生成

**処理フロー**:
```python
1. Celeryワーカー確認
   check_celery_workers(required_workers=workers)

2. タスク投入
   task_id = submit_unified_qa_generation(
       chunks=chunks,
       dataset_type=self.config["type"],
       model=self.model,
       batch_size=batch_size,
       workers=workers,
       merge_chunks=merge_chunks,
       min_tokens=min_tokens,
       max_tokens=max_tokens,
       config=self.config
   )

3. 結果収集
   qa_pairs = collect_results(task_id)

4. 返却
```

**使用例**:
```python
# 24ワーカーで並列処理
qa_pairs = pipeline._generate_with_celery(
    chunks,
    workers=24,
    batch_size=5,
    merge_chunks=True,
    min_tokens=150,
    max_tokens=400
)
```

---

### `_generate_sync(chunks, batch_size, merge_chunks, min_tokens, max_tokens)`

**目的**: 逐次Q/A生成（Celeryなし）

**処理**:
```python
qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type=self.config["type"],
    model=self.model,
    chunk_batch_size=batch_size,
    merge_chunks=merge_chunks,
    min_tokens=min_tokens,
    max_tokens=max_tokens,
    config=self.config,
    client=self.client
)
```

---

### `evaluate_coverage(chunks, qa_pairs, threshold)`

**シグネチャ**:
```python
def evaluate_coverage(self, chunks: List[Dict], qa_pairs: List[Dict],
                     threshold: Optional[float] = None) -> Dict
```

**目的**: Q/Aペアによるチャンクカバレージを分析

**処理**:
```python
coverage_results = analyze_coverage(
    chunks=chunks,
    qa_pairs=qa_pairs,
    dataset_type=self.config["type"],
    threshold=threshold
)
```

**返却値例**:
```python
{
    "coverage_rate": 0.85,
    "covered_chunks": 85,
    "total_chunks": 100,
    "uncovered_chunks": [
        {"chunk_id": "chunk_10", "reason": "類似度不足"},
        # ...
    ]
}
```

---

### `save(qa_pairs, coverage_results)`

**シグネチャ**:
```python
def save(self, qa_pairs: List[Dict], coverage_results: Dict) -> Dict[str, str]
```

**目的**: Q/Aペアとカバレージ結果を保存

**処理**:
```python
saved_files = save_results(
    qa_pairs=qa_pairs,
    coverage_results=coverage_results,
    dataset_type=self.config["type"],
    output_dir=self.output_dir
)

return saved_files
```

**返却値例**:
```python
{
    "qa_json": "qa_output/pipeline/qa_pairs_wikipedia_ja_20250117_143025.json",
    "qa_csv": "qa_output/pipeline/qa_pairs_wikipedia_ja_20250117_143025.csv",
    "coverage_json": "qa_output/pipeline/coverage_wikipedia_ja_20250117_143025.json"
}
```

---

### `run()` ⭐メイン

**シグネチャ**:
```python
def run(self,
        use_celery: bool = False,
        celery_workers: int = 8,
        batch_chunks: int = 3,
        merge_chunks: bool = True,
        min_tokens: int = 150,
        max_tokens: int = 400,
        analyze_coverage: bool = True,
        coverage_threshold: Optional[float] = None,
        overlap_tokens: int = 0,
        use_similarity: bool = False,
        similarity_threshold: float = 0.7)
```

**目的**: パイプライン全体を実行

**パラメータ**:

| パラメータ | 型 | デフォルト | 説明 |
|----------|---|----------|------|
| `use_celery` | bool | False | Celery並列処理 |
| `celery_workers` | int | 8 | Celeryワーカー数 |
| `batch_chunks` | int | 3 | Q/Aバッチサイズ |
| `merge_chunks` | bool | True | 小チャンク統合 |
| `min_tokens` | int | 150 | マージ最小トークン数 |
| `max_tokens` | int | 400 | マージ最大トークン数 |
| `analyze_coverage` | bool | True | カバレージ分析実施 |
| `coverage_threshold` | Optional[float] | None | カバレージ閾値 |
| `overlap_tokens` | int | 0 | チャンク間オーバーラップ |
| `use_similarity` | bool | False | セマンティック分割 |
| `similarity_threshold` | float | 0.7 | 類似度閾値 |

**処理フロー**:
```python
1. チャンク取得
   if self.input_chunks:
       chunks = self.load_chunks_from_csv()  # ✨新機能
   else:
       df = self.load_data()
       chunks = self.create_chunks(df, ...)

2. Q/A生成
   qa_pairs = self.generate_qa(chunks, ...)

3. カバレージ分析
   if analyze_coverage:
       coverage_results = self.evaluate_coverage(chunks, qa_pairs, ...)

4. 結果保存
   saved_files = self.save(qa_pairs, coverage_results)

5. 返却
   return {
       "saved_files": saved_files,
       "qa_count": len(qa_pairs),
       "coverage_results": coverage_results,
       "success": True
   }
```

**使用例**:
```python
# 基本的な実行
pipeline = QAPipeline(dataset_name="wikipedia_ja")
result = pipeline.run()

print(f"Q/A数: {result['qa_count']}")
print(f"カバレージ: {result['coverage_results']['coverage_rate']:.2%}")
print(f"保存ファイル: {result['saved_files']}")

# Celery並列処理
result = pipeline.run(
    use_celery=True,
    celery_workers=24
)

# カスタム設定
result = pipeline.run(
    use_celery=True,
    celery_workers=16,
    batch_chunks=5,
    merge_chunks=True,
    min_tokens=200,
    max_tokens=500,
    overlap_tokens=50,
    use_similarity=True,
    similarity_threshold=0.8
)
```

**返却値例**:
```python
{
    "saved_files": {
        "qa_json": "qa_output/pipeline/qa_pairs_wikipedia_ja_20250117.json",
        "qa_csv": "qa_output/pipeline/qa_pairs_wikipedia_ja_20250117.csv",
        "coverage_json": "qa_output/pipeline/coverage_wikipedia_ja_20250117.json"
    },
    "qa_count": 280,
    "coverage_results": {
        "coverage_rate": 0.87,
        "covered_chunks": 87,
        "total_chunks": 100,
        "uncovered_chunks": [...]
    },
    "success": True
}
```

---

## 🚀 使用方法

### パターン1: データセットモード（基本）

```python
from qa_generation.pipeline import QAPipeline

# パイプライン初期化
pipeline = QAPipeline(dataset_name="wikipedia_ja")

# 実行
result = pipeline.run()

print(f"生成Q/A数: {result['qa_count']}")
print(f"カバレージ率: {result['coverage_results']['coverage_rate']:.2%}")
```

---

### パターン2: ファイルモード

```python
# ローカルCSVからQ/A生成
pipeline = QAPipeline(input_file="my_data.csv")

result = pipeline.run(
    use_celery=False,  # 逐次処理
    batch_chunks=3
)
```

---

### パターン3: チャンクCSVモード ✨新機能

```python
# 事前作成されたチャンクCSVからQ/A生成
pipeline = QAPipeline(input_chunks="chunks.csv")

result = pipeline.run(
    use_celery=True,
    celery_workers=16,
    merge_chunks=True
)

# チャンク作成をスキップして効率的に処理
```

---

### パターン4: Celery並列処理（大規模）

```python
# 大規模データセットを並列処理
pipeline = QAPipeline(
    dataset_name="fineweb",
    max_docs=5000
)

result = pipeline.run(
    use_celery=True,
    celery_workers=24,
    batch_chunks=5,
    merge_chunks=True,
    min_tokens=200,
    max_tokens=500
)
```

---

### パターン5: カスタムチャンク作成

```python
# セマンティック分割 + オーバーラップ
pipeline = QAPipeline(dataset_name="wikipedia_ja")

result = pipeline.run(
    overlap_tokens=50,
    use_similarity=True,
    similarity_threshold=0.8
)
```

---

## 📊 パラメータリファレンス

### `__init__()` パラメータ

| パラメータ | 型 | デフォルト | 必須 | 説明 |
|----------|---|----------|-----|------|
| `dataset_name` | Optional[str] | None | ※ | データセット名 |
| `input_file` | Optional[str] | None | ※ | ファイルパス |
| `input_chunks` | Optional[str] | None | ※ | チャンクCSVパス |
| `model` | str | "gemini-2.0-flash" | - | 使用モデル |
| `output_dir` | str | "qa_output/pipeline" | - | 出力ディレクトリ |
| `max_docs` | Optional[int] | None | - | 最大文書数 |
| `client` | Optional[LLMClient] | None | - | LLMクライアント |

※ いずれか1つのみ必須

---

### `run()` パラメータ

| パラメータ | 型 | デフォルト | 説明 | 推奨値 |
|----------|---|----------|------|-------|
| `use_celery` | bool | False | Celery並列処理 | 1,000+チャンクでTrue |
| `celery_workers` | int | 8 | ワーカー数 | 8-24 |
| `batch_chunks` | int | 3 | Q/Aバッチサイズ | 3-5 |
| `merge_chunks` | bool | True | 小チャンク統合 | True推奨 |
| `min_tokens` | int | 150 | マージ最小トークン | 100-200 |
| `max_tokens` | int | 400 | マージ最大トークン | 300-500 |
| `analyze_coverage` | bool | True | カバレージ分析 | True推奨 |
| `coverage_threshold` | Optional[float] | None | カバレージ閾値 | 0.7-0.9 |
| `overlap_tokens` | int | 0 | チャンク間オーバーラップ | 0-100 |
| `use_similarity` | bool | False | セマンティック分割 | Falseで十分 |
| `similarity_threshold` | float | 0.7 | 類似度閾値 | 0.7-0.9 |

---

## 💡 実行例とワークフロー

### ワークフロー1: 小規模データセット（< 100チャンク）

```python
from qa_generation.pipeline import QAPipeline

# パイプライン初期化
pipeline = QAPipeline(
    dataset_name="wikipedia_ja",
    max_docs=50  # デバッグ用
)

# 実行（Celeryなし）
result = pipeline.run(
    use_celery=False,
    batch_chunks=3,
    merge_chunks=True
)

# 結果確認
print(f"Q/A数: {result['qa_count']}")
print(f"ファイル: {result['saved_files']['qa_csv']}")
```

**所要時間**: 約5-15分

---

### ワークフロー2: 中規模データセット（100-1,000チャンク）

```python
# Celery起動（別ターミナル）
# ./start_celery.sh restart -w 16

pipeline = QAPipeline(dataset_name="wikipedia_ja")

result = pipeline.run(
    use_celery=True,
    celery_workers=16,
    batch_chunks=5,
    merge_chunks=True,
    overlap_tokens=50
)

# 統計情報
print(f"Q/A数: {result['qa_count']}")
print(f"カバレージ: {result['coverage_results']['coverage_rate']:.2%}")
```

**所要時間**: 約30-90分

---

### ワークフロー3: 大規模データセット（1,000+チャンク）

```python
# Celery起動（最大ワーカー）
# ./start_celery.sh restart -w 24

pipeline = QAPipeline(
    dataset_name="fineweb",
    max_docs=5000
)

result = pipeline.run(
    use_celery=True,
    celery_workers=24,
    batch_chunks=5,
    merge_chunks=True,
    min_tokens=200,
    max_tokens=500
)

# 進捗モニタリング
# Flower: http://localhost:5555
```

**所要時間**: 数時間～数日

---

### ワークフロー4: チャンクCSVからの効率的処理 ✨

```python
# Step 1: チャンク作成（別ツール）
# python -m chunking.csv_to_chunks_text_para -i data.txt -o chunks.csv

# Step 2: Q/A生成（チャンク作成スキップ）
pipeline = QAPipeline(input_chunks="chunks.csv")

result = pipeline.run(
    use_celery=True,
    celery_workers=16,
    merge_chunks=True
)

# メリット:
# - チャンク品質を事前確認可能
# - チャンク作成時間の節約
# - デバッグしやすい
```

---

## 🔧 トラブルシューティング

### 問題1: 入力検証エラー

**症状**:
```
ValueError: dataset_name, input_file, input_chunks のいずれか1つを指定してください
```

**対処法**:
```python
# ❌ 誤り
pipeline = QAPipeline()

# ✅ 正しい
pipeline = QAPipeline(dataset_name="wikipedia_ja")
# または
pipeline = QAPipeline(input_file="data.csv")
# または
pipeline = QAPipeline(input_chunks="chunks.csv")
```

---

### 問題2: チャンクCSVの必須カラム不足

**症状**:
```
ValueError: 必須カラムが不足しています: ['chunk_id', 'text']
```

**対処法**:
```python
# CSVファイルを確認
import pandas as pd
df = pd.read_csv("chunks.csv")
print(df.columns)

# 必須カラムを追加
df['chunk_id'] = [f"chunk_{i}" for i in range(len(df))]
df['tokens'] = df['text'].apply(lambda x: len(x.split()))
df['chunk_idx'] = range(len(df))
df.to_csv("chunks_fixed.csv", index=False)
```

---

### 問題3: Celeryワーカー未起動

**症状**:
```
RuntimeError: Celery workers are not running
```

**対処法**:
```bash
# ワーカー起動
./start_celery.sh restart -w 16

# ステータス確認
./start_celery.sh status

# または、Celeryなしで実行
result = pipeline.run(use_celery=False)
```

---

### 問題4: メモリ不足

**症状**:
```
MemoryError: Unable to allocate array
```

**対処法**:
```python
# 1. max_docsで制限
pipeline = QAPipeline(dataset_name="wikipedia_ja", max_docs=100)

# 2. バッチサイズを小さく
result = pipeline.run(batch_chunks=1)

# 3. Celeryで分散処理
result = pipeline.run(use_celery=True, celery_workers=24)
```

---

### 問題5: カバレージ率が低い

**症状**:
```
coverage_rate: 0.45  # 期待: 0.80以上
```

**対処法**:
```python
# 1. batch_chunksを増やす（Q/A数増加）
result = pipeline.run(batch_chunks=5)

# 2. qa_per_chunkを増やす（設定変更が必要）
# config.py を編集

# 3. チャンクサイズを調整
# より小さいチャンクで密度向上
```

---

## 🎯 ベストプラクティス

### 1. 入力モードの選択

**推奨フロー**:
```
開発/デバッグ
  ↓
チャンクCSVモード
  ├─ チャンク品質確認
  ├─ Q/A生成のみテスト
  └─ 高速イテレーション
  ↓
本番実行
  ↓
データセットモード
  └─ 一貫した処理
```

---

### 2. Celeryの活用

**推奨**:
```python
# チャンク数に応じた選択
if len(chunks) < 100:
    use_celery = False  # 逐次処理
elif len(chunks) < 1000:
    use_celery = True
    celery_workers = 16
else:
    use_celery = True
    celery_workers = 24
```

---

### 3. チャンクマージの最適化

**推奨**:
```python
# データセット特性に応じた調整
MERGE_SETTINGS = {
    'wikipedia_ja': {
        'merge_chunks': True,
        'min_tokens': 200,
        'max_tokens': 500
    },
    'cc_news': {
        'merge_chunks': True,
        'min_tokens': 150,
        'max_tokens': 400
    },
    'fineweb': {
        'merge_chunks': False  # 既に最適化済み
    }
}

dataset = 'wikipedia_ja'
settings = MERGE_SETTINGS[dataset]
result = pipeline.run(**settings)
```

---

### 4. エラーハンドリング

**推奨**:
```python
import logging

logging.basicConfig(level=logging.INFO)

try:
    pipeline = QAPipeline(dataset_name="wikipedia_ja")
    result = pipeline.run()

    # 成功後の処理
    save_to_database(result)

except ValueError as e:
    logging.error(f"設定エラー: {e}")
except RuntimeError as e:
    logging.error(f"実行時エラー: {e}")
except Exception as e:
    logging.error(f"予期しないエラー: {e}")
    # 部分結果の保存など
```

---

### 5. 段階的なテスト

**推奨フロー**:
```python
# Step 1: 小規模テスト（10文書）
pipeline = QAPipeline(dataset_name="wikipedia_ja", max_docs=10)
result = pipeline.run(use_celery=False)

# Step 2: 中規模テスト（100文書）
pipeline = QAPipeline(dataset_name="wikipedia_ja", max_docs=100)
result = pipeline.run(use_celery=True, celery_workers=8)

# Step 3: 本番実行（全文書）
pipeline = QAPipeline(dataset_name="wikipedia_ja")
result = pipeline.run(use_celery=True, celery_workers=24)
```

---

### 6. ログの活用

**推奨**:
```python
# ログレベルを詳細に
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

result = pipeline.run()

# ログから確認すべき項目:
# - [1/4] データ読み込み
# - [2/4] チャンク作成/読み込み
# - [3/4] Q/A生成
# - [4/4] カバレージ分析 & 結果保存
```

---

### 7. 結果の検証

**推奨**:
```python
import pandas as pd

result = pipeline.run()

# CSV読み込み
df = pd.read_csv(result['saved_files']['qa_csv'])

# 統計確認
print(f"Q/A数: {len(df)}")
print(f"質問タイプ分布:\n{df['question_type'].value_counts()}")
print(f"データセット分布:\n{df['dataset_type'].value_counts()}")

# サンプル確認
print("\n【サンプルQ/A】")
for i in range(min(5, len(df))):
    print(f"\nQ: {df.iloc[i]['question']}")
    print(f"A: {df.iloc[i]['answer']}")

# 品質チェック
empty_qa = df[(df['question'].str.strip() == '') | (df['answer'].str.strip() == '')]
print(f"\n空のQ/A: {len(empty_qa)}個")
```

---

### 8. パフォーマンス最適化

**推奨**:
```python
# LLMクライアントの再利用
from helper.helper_llm import create_llm_client

client = create_llm_client(provider="gemini")

# 複数データセットを処理
for dataset_name in ['dataset1', 'dataset2', 'dataset3']:
    pipeline = QAPipeline(
        dataset_name=dataset_name,
        client=client  # クライアント再利用
    )
    result = pipeline.run()
```

---

### 9. チャンクCSVの活用 ✨

**推奨フロー**:
```bash
# Step 1: チャンク作成と確認
python -m chunking.csv_to_chunks_text_para -i data.txt -o chunks.csv

# チャンクの品質確認
head -20 chunks.csv

# Step 2: Q/A生成
python -c "
from qa_generation.pipeline import QAPipeline
pipeline = QAPipeline(input_chunks='chunks.csv')
result = pipeline.run()
print(f'Q/A数: {result[\"qa_count\"]}')
"
```

---

### 10. 進捗モニタリング

**推奨**:
```bash
# Flowerでモニタリング
celery -A celery_config flower --port=5555

# ブラウザでアクセス
# http://localhost:5555

# タスク数、成功率、失敗数などを確認
```

---

## 📚 関連ドキュメント

- `qa_generation/generation.md` - Q/A生成モジュールの詳細
- `qa_generation/doc/qa_generation.md` - qa_generationモジュール全体ガイド
- `qa_generation/structure.py` - チャンク作成の実装
- `qa_generation/evaluation.py` - カバレージ分析の実装
- `celery_tasks.py` - Celeryタスクの定義

---

**作成日**: 2025-01-17
**対象ファイル**: `qa_generation/pipeline.py`
**バージョン**: 1.0.0
**総行数**: 430行
