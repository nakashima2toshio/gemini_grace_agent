## Q/A生成 & Qdrant登録システム 完全設計書
3フェーズ処理: CSV行結合 → Q/A生成 → Qdrant登録
- Q/A Smart_Generation
- Q/A QAPipeline
- SemanticCoverage

- qdrant Services
- Qdrant Client Wrapper

## 1. システム概要

テキスト/CSVデータからQ/Aペアを自動生成し、Qdrantベクトルデータベースに登録するRAGパイプラインシステム。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        make_qa_register_qdrant.py                           │
│                          (CLIエントリーポイント)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  Phase 0      │         │    Phase 1      │         │    Phase 2      │
│  CSV行結合     │         │   Q/A生成        │         │  Qdrant登録     │
│ (オプション)    │         │                 │         │                 │
└───────────────┘         └─────────────────┘         └─────────────────┘
        │                         │                           │
        ▼                         ▼                           ▼
  combine_rows_to_chunks    QAPipeline                 run_registration
        │                         │                           │
        │                    ┌────┴────┐                      │
        │                    ▼         ▼                      ▼
        │              Smart生成   Legacy生成           Qdrant Service
        │                    │         │                      │
        │                    ▼         ▼                      ▼
        │               SmartQA    QAGenerator          Embedding生成
        │               Generator                             │
        │                    │         │                      ▼
        │                    └────┬────┘                 ベクトル登録
        │                         │
        │                         ▼
        │                  カバレージ分析
        │                  (SemanticCoverage)
        │                         │
        └─────────────────────────┼─────────────────────────────
                                  ▼
                           出力ファイル生成
                        (CSV/JSON/UI CSV)
```

---

## 2. ファイル構成と依存関係

### 2.1 メインプログラム

| ファイル | 行数 | 役割 |
|---------|------|------|
| `make_qa_register_qdrant.py` | 780 | CLIエントリーポイント、全体オーケストレーション |

### 2.2 Q/A生成モジュール (`qa_generation/`)

| ファイル | 行数 | 役割 |
|---------|------|------|
| `pipeline.py` | - | Q/A生成パイプライン制御 (`QAPipeline`) |
| `generation.py` | - | Q/A生成ロジック (`QAGenerator`) |
| `smart_qa_generator.py` | - | スマート生成 (`SmartQAGenerator`) |
| `structure.py` | - | チャンク分割・構造化 |
| `semantic.py` | - | セマンティック分析 (`SemanticCoverage`) |
| `data_io.py` | 156 | データ入出力 |
| `evaluation.py` | 252 | カバレージ分析 |
| `models.py` | 183 | Pydanticモデル定義 |

### 2.3 Qdrantモジュール

| ファイル | 行数 | 役割 |
|---------|------|------|
| `qdrant_service.py` | 1072 | Qdrant操作サービス（高レベルAPI） |
| `qdrant_client_wrapper.py` | 1184 | Qdrantクライアントラッパー（低レベルAPI） |
| `register_to_qdrant.py` | - | Qdrant登録専用モジュール |

### 2.4 ヘルパーモジュール (`helper/`)

| ファイル | 行数 | 役割 |
|---------|------|------|
| `helper_llm.py` | 246 | LLMクライアント抽象化 |
| `helper_embedding.py` | 318 | Embeddingクライアント抽象化 |
| `helper_rag.py` | - | RAG関連ユーティリティ |

### 2.5 設定・タスク

| ファイル | 行数 | 役割 |
|---------|------|------|
| `config.py` | 552 | 全設定の一元管理 |
| `celery_tasks.py` | 588 | Celery分散タスク定義 |
| `celery_config.py` | - | Celery設定 |

---

## 3. 処理フロー詳細

### 3.1 Phase 0: CSV行結合（オプション）

```python
# トリガー: --combine-rows フラグ
def combine_rows_to_chunks(input_file, text_column, block_size=400):
    """
    複数のCSV行を結合してチャンクを作成

    処理:
    1. CSVを読み込み
    2. text_columnからテキストを抽出
    3. block_size行ごとに結合
    4. combined_chunks_YYYYMMDD_HHMMSS.csv として出力
    """
```

### 3.2 Phase 1: Q/A生成

#### 3.2.1 パイプライン制御 (`QAPipeline`)

```python
class QAPipeline:
    def __init__(self, dataset_name, model, output_dir):
        self.dataset_name = dataset_name
        self.model = model
        self.output_dir = output_dir

    def run(self,
            use_smart_generation=True,
            batch_chunks=3,
            use_celery=False,
            concurrency=8,
            min_tokens=150,
            max_tokens=400):
        """
        メインパイプライン実行

        1. load_data()      - データ読み込み
        2. create_chunks()  - チャンク分割
        3. generate_qa()    - Q/A生成
        4. evaluate_coverage() - カバレージ分析
        5. save_results()   - 結果保存
        """
```

#### 3.2.2 チャンク分割 (`structure.py`)

```python
def create_document_chunks(df, text_column, dataset_type,
                          min_tokens=150, max_tokens=400,
                          use_semantic=False, concurrency=8):
    """
    DataFrameからチャンクを作成

    処理:
    1. ThreadPoolExecutorで並列処理
    2. セマンティック分割 or 固定長分割
    3. 小さすぎるチャンクのマージ

    出力:
    [
        {
            'id': 'chunk_0',
            'text': '...',
            'tokens': 250,
            'type': 'paragraph',
            'doc_id': 'doc_0',
            'chunk_idx': 0,
            'dataset_type': 'local_file'
        },
        ...
    ]
    """

def create_semantic_chunks(text, min_tokens, max_tokens, overlap_tokens=50):
    """
    セマンティック境界でチャンク分割

    1. 段落分割 → 文分割
    2. MeCab（日本語）/ regex（英語）でセンテンス分割
    3. 類似度ベースの境界検出
    4. オーバーラップ適用
    """
```

#### 3.2.3 Q/A生成 (`generation.py`)

```python
class QAGenerator:
    def __init__(self, use_smart_generation=True):
        self.use_smart_generation = use_smart_generation
        self.smart_generator = SmartQAGenerator() if use_smart_generation else None

    def generate_for_chunk(self, chunk, config, model):
        """
        単一チャンクのQ/A生成

        Smart mode:
        1. analyze_chunk() でチャンク分析
        2. 0-5個のQ/Aを動的決定
        3. topic, importance_score, complexity付与

        Legacy mode:
        1. トークン数ベースで2-8個のQ/A固定生成
        """

    def generate_for_batch(self, chunks, config, model):
        """バッチ処理（複数チャンクを1API呼び出しで処理）"""

def generate_qa_dataset(chunks, dataset_type, model,
                       chunk_batch_size=3,
                       use_smart_generation=True,
                       config=None, client=None):
    """
    データセット全体のQ/A生成（トップレベル関数）

    Celeryタスクから呼び出される
    """
```

#### 3.2.4 スマート生成 (`smart_qa_generator.py`)

```python
class SmartQAGenerator:
    def __init__(self, model="gemini-2.0-flash"):
        # google.genai (新API) を使用
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def analyze_chunk(self, chunk_text):
        """
        チャンク分析（LLM使用）

        出力:
        {
            'qa_count': 0-5,           # 生成すべきQ/A数
            'key_topics': [...],       # 主要トピック
            'importance_score': 0.85,  # 重要度スコア
            'complexity': 'high',      # low/medium/high
            'reasoning': '...'         # 分析理由
        }
        """

    def generate_qa_pairs(self, chunk_text, analysis):
        """
        分析結果に基づくQ/A生成

        出力:
        [
            {
                'question': '...',
                'answer': '...',
                'question_type': 'fact',
                'topic': 'トピック名',
                'importance_score': 0.85,
                'complexity': 'high'
            },
            ...
        ]
        """

    def process_chunk(self, chunk):
        """analyze + generate の統合処理"""
```

### 3.3 Phase 2: Qdrant登録

#### 3.3.1 登録メイン処理

```python
def run_registration(qa_csv_path, collection_name, recreate=False,
                    batch_size=100, provider="gemini", ui_output_dir="qa_output"):
    """
    Q/A CSVをQdrantに登録

    処理:
    1. CSVロード & テキスト列検出
    2. コレクション作成/再作成
    3. バッチ埋め込み生成
    4. ポイント構築 & アップサート
    5. UI用CSV生成（ファイル名正規化）
    """
```

#### 3.3.2 Qdrantサービス (`qdrant_service.py`)

```python
# ===== ヘルスチェック =====
class QdrantHealthChecker:
    def check_port(host, port, timeout=2.0) -> bool
    def check_qdrant() -> Tuple[bool, str, Optional[Dict]]

# ===== データフェッチ =====
class QdrantDataFetcher:
    def __init__(self, client: QdrantClient)
    def fetch_collections() -> pd.DataFrame
    def fetch_collection_points(collection_name, limit=50) -> pd.DataFrame
    def fetch_collection_info(collection_name) -> Dict
    def fetch_collection_source_info(collection_name, sample_size=200) -> Dict

# ===== コレクション管理 =====
def get_collection_stats(client, collection_name) -> Optional[Dict]
def get_all_collections(client) -> List[Dict]
def get_all_collections_simple(client) -> List[Dict]
def delete_all_collections(client, excluded=[]) -> int
def create_or_recreate_collection_for_qdrant(client, name, recreate, vector_size=3072, use_sparse=False)

# ===== データ処理・登録 =====
def load_csv_for_qdrant(path, required=("question", "answer"), limit=0) -> pd.DataFrame
def build_inputs_for_embedding(df, include_answer) -> List[str]
def embed_texts_for_qdrant(texts, model="gemini-embedding-001", batch_size=100) -> List[List[float]]
def build_points_for_qdrant(df, vectors, domain, source_file, sparse_vectors=None, start_index=0) -> List[PointStruct]
def upsert_points_to_qdrant(client, collection, points, batch_size=128) -> int

# ===== 検索 =====
def embed_query_for_search(query, model="gemini-embedding-001", dims=None) -> List[float]

# ===== コレクション統合 =====
def scroll_all_points_with_vectors(client, collection_name, batch_size=100, progress_callback=None) -> List[Record]
def merge_collections(client, source_collections, target_collection, recreate=True, vector_size=3072, progress_callback=None) -> Dict

# ===== マッピング =====
def map_collection_to_csv(collection_name, qa_output_dir="qa_output") -> Optional[str]
def get_dynamic_collection_mapping(client, qa_output_dir="qa_output") -> Dict[str, str]
def get_collection_embedding_params(client, collection_name) -> Dict[str, Any]
```

#### 3.3.3 Qdrantクライアントラッパー (`qdrant_client_wrapper.py`)

```python
# ===== クライアント生成 =====
def create_qdrant_client(url=None, timeout=30) -> QdrantClient

# ===== コレクション管理 =====
def create_or_recreate_collection(client, name, recreate=False, vector_size=1536, use_sparse=False)

# ===== 埋め込み生成（プロバイダー抽象化） =====
def embed_texts_unified(texts, provider=None, batch_size=100) -> List[List[float]]
def embed_query_unified(text, provider=None) -> List[float]
def embed_sparse_texts_unified(texts, batch_size=100) -> List[SparseVector]

# ===== ポイント操作 =====
def build_points(df, vectors, domain, source_file, sparse_vectors=None) -> List[PointStruct]
def upsert_points(client, collection, points, batch_size=128) -> int

# ===== 検索 =====
def search_collection(client, collection_name, query_vector, sparse_vector=None, limit=5, hybrid_alpha=0.5) -> List[Dict]

# ===== 後方互換エイリアス =====
embed_texts_for_qdrant = embed_texts
create_or_recreate_collection_for_qdrant = create_or_recreate_collection
build_points_for_qdrant = build_points
upsert_points_to_qdrant = upsert_points
embed_query_for_search = embed_query
```

---

## 4. ヘルパーモジュール詳細

### 4.1 LLMクライアント (`helper_llm.py`)

```python
# ===== 抽象基底クラス =====
class LLMClient(ABC):
    @abstractmethod
    def generate_content(self, prompt, model=None, **kwargs) -> str

    @abstractmethod
    def generate_structured(self, prompt, response_schema: Type[BaseModel], model=None, **kwargs) -> BaseModel

    @abstractmethod
    def count_tokens(self, text, model=None) -> int

# ===== OpenAI実装 =====
class OpenAIClient(LLMClient):
    def __init__(self, api_key=None, default_model="gpt-4o-mini")

# ===== Gemini実装 =====
class GeminiClient(LLMClient):
    def __init__(self, api_key=None, default_model="gemini-2.0-flash")
    # google.genai パッケージ使用

# ===== ファクトリ =====
def create_llm_client(provider="gemini", **kwargs) -> LLMClient

# ===== モデル設定 =====
LLM_MODELS = ["gemini-2.5-flash", "gemini-3-pro-preview", ...]
LLM_PRICING = {"gemini-2.5-flash": {"input": 0.0001, "output": 0.0004}, ...}
LLM_LIMITS = {"gemini-2.5-flash": {"max_tokens": 1000000, "max_output": 8192}, ...}
```

### 4.2 Embeddingクライアント (`helper_embedding.py`)

```python
# ===== 定数 =====
DEFAULT_GEMINI_EMBEDDING_DIMS = 3072
DEFAULT_OPENAI_EMBEDDING_DIMS = 1536

# ===== 抽象基底クラス =====
class EmbeddingClient(ABC):
    @property
    @abstractmethod
    def dimensions(self) -> int

    @abstractmethod
    def embed_text(self, text, task_type=None) -> List[float]

    @abstractmethod
    def embed_texts(self, texts, batch_size=100) -> List[List[float]]

# ===== OpenAI実装 =====
class OpenAIEmbedding(EmbeddingClient):
    def __init__(self, api_key=None, model="text-embedding-3-small", dims=1536)

# ===== Gemini実装 =====
class GeminiEmbedding(EmbeddingClient):
    def __init__(self, api_key=None, model="gemini-embedding-001", dims=3072)
    # google.genai パッケージ使用
    # task_type: "retrieval_document", "retrieval_query" 対応

# ===== ファクトリ =====
def create_embedding_client(provider="gemini", **kwargs) -> EmbeddingClient

# ===== ヘルパー =====
def get_embedding_dimensions(provider="gemini") -> int
def get_default_embedding_client(**kwargs) -> EmbeddingClient
```

---

## 5. 設定クラス (`config.py`)

```python
# ===== モデル設定 =====
class ModelConfig:
    AVAILABLE_MODELS: List[str]
    DEFAULT_MODEL: str = "gemini-2.5-flash"
    MODEL_PRICING: Dict[str, Dict[str, float]]
    MODEL_LIMITS: Dict[str, Dict[str, int]]

    @classmethod
    def supports_temperature(cls, model) -> bool
    @classmethod
    def get_model_limits(cls, model) -> Dict[str, int]

# ===== データセット設定 =====
@dataclass
class DatasetInfo:
    name: str
    icon: str
    description: str
    file: Optional[str]
    hf_dataset: Optional[str]
    text_field: str = "text"
    text_column: Optional[str]
    sample_size: int = 1000
    min_text_length: int = 100
    chunk_size: int = 300
    qa_per_chunk: int = 3
    lang: str = "ja"

class DatasetConfig:
    DATASETS: Dict[str, DatasetInfo] = {
        "wikipedia_ja": DatasetInfo(...),
        "wikipedia_ja_5per": DatasetInfo(...),
        "japanese_text": DatasetInfo(...),
        "fineweb_edu_ja": DatasetInfo(...),
        "cc_news": DatasetInfo(...),
        "livedoor": DatasetInfo(...),
    }

    RAG_DATASETS: Dict[str, Dict[str, Any]] = {...}

    @classmethod
    def get_dataset_dict(cls, name) -> Dict[str, Any]
    @classmethod
    def get_all_dataset_names(cls) -> List[str]

# ===== Q/A生成設定 =====
class QAConfig:
    QUESTION_TYPES_HIERARCHY: Dict[str, Dict[str, str]]  # basic/understanding/application
    DEFAULT_COVERAGE_THRESHOLD: float = 0.58
    DEFAULT_BATCH_CHUNKS: int = 3
    DEFAULT_MIN_TOKENS: int = 150
    DEFAULT_MAX_TOKENS: int = 400

# ===== Qdrant設定 =====
class QdrantConfig:
    HOST: str = "localhost"
    PORT: int = 6333
    URL: str = "http://localhost:6333"
    DEFAULT_VECTOR_SIZE: int = 1536
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"

# ===== Celery設定 =====
class CeleryConfig:
    BROKER_URL: str = "redis://localhost:6379/0"
    RESULT_BACKEND: str = "redis://localhost:6379/0"
    WORKER_CONCURRENCY: int = 8
    TASK_TIME_LIMIT: int = 300

# ===== Gemini API設定 =====
class GeminiConfig:
    AVAILABLE_MODELS: List[str]
    DEFAULT_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMS: int = 3072
    DEFAULT_TEMPERATURE: float = 1.0

# ===== パス設定 =====
class PathConfig:
    BASE_DIR: Path
    OUTPUT_DIR: Path
    QA_OUTPUT_DIR: Path
    DATASETS_DIR: Path
```

---

## 6. Celeryタスク (`celery_tasks.py`)

```python
# ===== タスク投入 =====
def submit_unified_qa_generation(chunks, config, model,
                                 provider="gemini",
                                 use_smart_generation=True) -> List[AsyncResult]:
    """チャンクのQ/A生成タスクを並列投入"""

# ===== Celeryタスク定義 =====
@app.task(name='generate_qa_for_chunk', bind=True, max_retries=3)
def generate_qa_for_chunk_task(self, chunk, config, model,
                               use_smart_generation=True) -> List[Dict]:
    """
    単一チャンクのQ/A生成タスク

    - 自動リトライ（最大3回、60秒間隔）
    - バックオフ + ジッター
    - qa_generation.generation.generate_qa_dataset() を呼び出し
    """

# ===== 結果収集 =====
def collect_results(tasks, timeout=600) -> List[Dict]:
    """タスク結果を収集（タイムアウト対応）"""

# ===== ワーカー状態確認 =====
def get_worker_info() -> Dict:
    """
    ワーカー詳細情報を取得

    返却:
    {
        'available': bool,
        'worker_count': int,
        'total_concurrency': int,
        'workers': Dict[str, Dict],
        'error': Optional[str]
    }
    """

def get_total_concurrency() -> int:
    """総並列処理能力を取得"""

def check_celery_workers(min_workers=1, required_concurrency=None) -> Union[bool, Dict]:
    """
    ワーカー状態確認

    後方互換モード（required_concurrency=None）: bool返却
    詳細モード（required_concurrency指定）: Dict返却
    """

def validate_concurrency(requested) -> Dict:
    """要求された並列数が実行可能かを検証"""

# ===== ユーティリティ =====
def get_active_tasks() -> Dict
def purge_queue(queue_name='celery') -> int
```

---

## 7. データモデル (`models.py`)

```python
# ===== Q/A関連 =====
class QAPair(BaseModel):
    question: str
    answer: str
    question_type: str = "fact"  # fact/reason/comparison/application/definition/process/evaluation
    difficulty_level: Optional[str] = "medium"  # easy/medium/hard
    question_category: Optional[str] = "understanding"  # basic/understanding/application
    source_chunk_id: Optional[str]
    dataset_type: Optional[str]
    auto_generated: bool = False
    confidence_score: Optional[float]
    quality_score: Optional[float]

class QAPairsResponse(BaseModel):
    qa_pairs: List[QAPair] = []

# ===== チャンク関連 =====
class ChunkData(BaseModel):
    id: str
    text: str
    tokens: int = 0
    doc_id: Optional[str]
    dataset_type: Optional[str]
    chunk_idx: int = 0
    position: Optional[str]  # start/middle/end

class ChunkComplexity(BaseModel):
    complexity_level: str = "medium"  # low/medium/high
    technical_terms: List[str] = []
    avg_sentence_length: float = 0.0
    concept_density: float = 0.0

# ===== Qdrant関連 =====
class QdrantPointPayload(BaseModel):
    domain: str
    question: str
    answer: str
    source: str
    created_at: str
    schema_version: str = "qa:v1"
    generation_method: Optional[str]

class QdrantCollectionStats(BaseModel):
    total_points: int = 0
    vector_config: Dict[str, Any] = {}
    status: str = "unknown"

# ===== カバレージ分析 =====
class CoverageResult(BaseModel):
    coverage_rate: float = 0.0
    covered_chunks: int = 0
    total_chunks: int = 0
    uncovered_chunks: List[str] = []
```

---

## 8. データ入出力 (`data_io.py`)

```python
def load_uploaded_file(file_path) -> pd.DataFrame:
    """
    ローカルファイル読み込み

    対応形式: csv, txt, json, jsonl
    自動処理:
    - Combined_Text列の生成
    - 空行の除去
    """

def load_preprocessed_data(dataset_type) -> pd.DataFrame:
    """
    前処理済みデータセット読み込み

    - config.DATASET_CONFIGSから設定取得
    - タイムスタンプ付きファイルの自動選択
    """

def save_results(qa_pairs, coverage_results, dataset_type,
                output_dir="qa_output/a02") -> Dict[str, str]:
    """
    結果保存

    出力ファイル:
    - qa_pairs_{dataset_type}_{timestamp}.json
    - qa_pairs_{dataset_type}_{timestamp}.csv
    - coverage_{dataset_type}_{timestamp}.json
    - summary_{dataset_type}_{timestamp}.json
    """
```

---

## 9. カバレージ分析 (`evaluation.py`)

```python
def get_optimal_thresholds(dataset_type) -> Dict[str, float]:
    """
    データセット別の最適閾値を取得

    返却: {strict: 0.8, standard: 0.7, lenient: 0.6}
    """

def multi_threshold_coverage(coverage_matrix, chunks, qa_pairs, thresholds) -> Dict:
    """
    複数閾値でカバレージを評価

    各閾値レベル（strict/standard/lenient）ごとに:
    - covered_chunks
    - coverage_rate
    - uncovered_chunks
    """

def analyze_chunk_characteristics_coverage(chunks, coverage_matrix, qa_pairs, threshold=0.7) -> Dict:
    """
    チャンク特性別のカバレージ分析

    分析軸:
    - by_length: short/medium/long
    - by_position: beginning/middle/end
    - summary + insights
    """

def analyze_coverage(chunks, qa_pairs, dataset_type="wikipedia_ja",
                    custom_threshold=None) -> Dict:
    """
    包括的カバレージ分析

    処理:
    1. チャンク埋め込み生成
    2. Q/Aペア埋め込み生成（バッチAPI）
    3. カバレージ行列計算（NumPy行列積）
    4. 多段階カバレージ分析
    5. チャンク特性別分析

    返却:
    {
        'coverage_rate': float,
        'covered_chunks': int,
        'total_chunks': int,
        'uncovered_chunks': List,
        'multi_threshold': Dict,
        'chunk_analysis': Dict,
        'optimal_thresholds': Dict
    }
    """
```

---

## 10. コマンドラインオプション

```bash
python make_qa_register_qdrant.py [OPTIONS]

# === 入力オプション ===
--dataset NAME           # 定義済みデータセット名
--input-file PATH        # ローカルファイルパス (.txt, .csv)
--text-column NAME       # CSVのテキスト列名 (default: text)
--combine-rows           # CSV行結合モード有効化
--block-size N           # 結合する行数 (default: 400)

# === Q/A生成オプション ===
--model NAME             # LLMモデル (default: gemini-2.0-flash)
--use-smart-generation   # スマート生成有効 (default: True)
--no-smart-generation    # 従来方式（トークンベース固定生成）
--batch-chunks N         # バッチあたりのチャンク数 (default: 3)
--use-celery             # Celery並列処理有効化
--concurrency N          # 並列タスク数 (default: 8)

# === Qdrantオプション ===
--collection NAME        # コレクション名（必須）
--recreate               # コレクション再作成
--batch-size N           # 埋め込みバッチサイズ (default: 100)
--provider NAME          # gemini or openai (default: gemini)

# === 出力オプション ===
--output DIR             # Q/A出力ディレクトリ (default: qa_output/pipeline)
--ui-output DIR          # UI用CSV出力ディレクトリ (default: qa_output)
```

---

## 11. 使用例

### 11.1 基本的な使用

```bash
# ローカルCSVからQ/A生成 & Qdrant登録
python make_qa_register_qdrant.py \
  --input-file data/articles.csv \
  --text-column content \
  --collection articles_qa \
  --recreate
```

### 11.2 CSV行結合モード

```bash
# 多数の短い行を結合してからQ/A生成
python make_qa_register_qdrant.py \
  --input-file data/news.csv \
  --text-column text \
  --combine-rows \
  --block-size 400 \
  --collection news_qa \
  --recreate
```

### 11.3 Celery並列処理

```bash
# 24並列でQ/A生成（大規模データ向け）
python make_qa_register_qdrant.py \
  --dataset wikipedia_ja \
  --collection wiki_qa \
  --use-celery \
  --concurrency 24 \
  --recreate
```

### 11.4 従来方式（Legacy）

```bash
# スマート生成を使わない高速処理
python make_qa_register_qdrant.py \
  --input-file data/faq.csv \
  --collection faq_qa \
  --no-smart-generation \
  --recreate
```

---

## 12. 出力ファイル

### 12.1 Q/A生成フェーズ出力

```
qa_output/pipeline/
├── qa_pairs_local_file_20250126_123456.csv   # Q/Aペア（CSV）
├── qa_pairs_local_file_20250126_123456.json  # Q/Aペア（JSON）
├── coverage_local_file_20250126_123456.json  # カバレージ分析
└── summary_local_file_20250126_123456.json   # サマリー
```

### 12.2 Qdrant登録フェーズ出力

```
qa_output/
└── {collection_name}.csv   # UI用CSV（タイムスタンプなし、正規化済み）
```

### 12.3 CSV行結合フェーズ出力

```
{input_dir}/
└── combined_chunks_20250126_123456.csv
```

---

## 13. 技術的特徴

### 13.1 Gemini 3 Migration

- **google.genai** パッケージ使用（新API）
- 埋め込み: 3072次元（Gemini最大精度）
- task_type: `retrieval_document`, `retrieval_query` 対応
- フォールバック: google.generativeai（旧API）

### 13.2 プロバイダー抽象化

- LLMClient / EmbeddingClient 抽象基底クラス
- OpenAI / Gemini 切り替え可能
- ファクトリ関数によるインスタンス生成

### 13.3 並列処理

- チャンク分割: ThreadPoolExecutor
- Q/A生成: Celery分散タスク
- 埋め込み生成: バッチAPI（最大100件/リクエスト）

### 13.4 Hybrid Search対応

- Dense Vector: Gemini/OpenAI埋め込み
- Sparse Vector: Splade（オプション）
- RRF (Reciprocal Rank Fusion) による統合

---

## 14. エラーハンドリング

### 14.1 リトライ戦略

```python
# Celeryタスク
@app.task(max_retries=3, default_retry_delay=60,
          retry_backoff=True, retry_jitter=True)

# 処理内リトライ
for attempt in range(max_retries):
    try:
        result = api_call()
        break
    except RateLimitError:
        time.sleep(2 ** attempt)
```

### 14.2 フォールバック

- バッチ処理失敗 → 個別処理
- Hybrid Search失敗 → Dense Searchのみ
- 新API失敗 → 旧API

---

## 15. 依存関係図

```
make_qa_register_qdrant.py
├── qa_generation/
│   ├── pipeline.py
│   │   ├── data_io.py
│   │   ├── structure.py
│   │   │   └── semantic.py
│   │   ├── generation.py
│   │   │   └── smart_qa_generator.py
│   │   └── evaluation.py
│   │       └── semantic.py
│   └── models.py
├── qdrant_service.py
│   ├── qdrant_client_wrapper.py
│   └── helper/helper_embedding.py
├── helper/
│   ├── helper_llm.py
│   ├── helper_embedding.py
│   └── helper_rag.py
├── config.py
└── celery_tasks.py
    └── celery_config.py
```

---

## 16. 環境変数

```bash
# 必須
GOOGLE_API_KEY=your_gemini_api_key

# オプション（OpenAI使用時）
OPENAI_API_KEY=your_openai_api_key

# オプション（設定変更）
EMBEDDING_PROVIDER=gemini  # or openai
LLM_PROVIDER=gemini        # or openai

# Celery（並列処理時）
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

---

*Last Updated: 2025-01-26*
