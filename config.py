#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config.py - 設定・定数の一元管理
================================
プロジェクト全体の設定と定数を一元管理

使用箇所:
- rag_qa_pair_qdrant.py
- celery_tasks.py
- a02_make_qa_para.py
- helper_rag.py
- helper_api.py
"""

from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass
from pathlib import Path


# ===================================================================
# モデル設定
# ===================================================================

class ModelConfig:
    """Gemini モデル設定（Gemini API学習用）"""

    # 利用可能なモデル一覧
    AVAILABLE_MODELS: List[str] = [
        "gemini-3-pro-preview",       # 最新Pro（思考モード対応）
        "gemini-2.5-flash-preview",   # 高速・思考対応
        "gemini-2.0-flash",           # 安定版（推奨デフォルト）
        "gemini-1.5-pro",             # レガシー
        "gemini-1.5-flash",           # レガシー高速
    ]

    # デフォルトモデル
    DEFAULT_MODEL: str = "gemini-2.0-flash"

    # temperatureパラメータをサポートしないモデル
    # Geminiでは全モデルでtemperatureがサポートされる
    NO_TEMPERATURE_MODELS: List[str] = []

    # モデル料金（$/1M tokens）
    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        "gemini-3-pro-preview": {"input": 0.00125, "output": 0.010},
        "gemini-2.5-flash-preview": {"input": 0.00015, "output": 0.0035},
        "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    }

    # モデル制限
    MODEL_LIMITS: Dict[str, Dict[str, int]] = {
        "gemini-3-pro-preview": {"max_tokens": 1000000, "max_output": 64000},
        "gemini-2.5-flash-preview": {"max_tokens": 1000000, "max_output": 64000},
        "gemini-2.0-flash": {"max_tokens": 1000000, "max_output": 8192},
        "gemini-1.5-pro": {"max_tokens": 1000000, "max_output": 8192},
        "gemini-1.5-flash": {"max_tokens": 1000000, "max_output": 8192},
    }

    @classmethod
    def supports_temperature(cls, model: str) -> bool:
        """モデルがtemperatureパラメータをサポートするかチェック"""
        return model not in cls.NO_TEMPERATURE_MODELS

    @classmethod
    def get_model_limits(cls, model: str) -> Dict[str, int]:
        """モデルの制限を取得"""
        return cls.MODEL_LIMITS.get(model, {"max_tokens": 128000, "max_output": 4096})

    @classmethod
    def get_model_pricing(cls, model: str) -> Dict[str, float]:
        """モデルの料金を取得"""
        return cls.MODEL_PRICING.get(model, {"input": 0.00015, "output": 0.0006})

    @classmethod
    def uses_max_completion_tokens(cls, model: str) -> bool:
        """max_completion_tokensを使用するモデルかどうか"""
        # Geminiでは全モデルでmax_output_tokensを使用
        return False


# ===================================================================
# データセット設定
# ===================================================================

@dataclass
class DatasetInfo:
    """データセット情報"""
    name: str
    icon: str
    description: str
    file: Optional[str] = None
    hf_dataset: Optional[str] = None
    hf_config: Optional[str] = None
    download_url: Optional[str] = None
    split: Optional[str] = "train"
    text_field: str = "text"
    title_field: Optional[str] = None
    text_column: Optional[str] = None  # a02_make_qa_para用
    sample_size: int = 1000
    min_text_length: int = 100
    chunk_size: int = 300
    qa_per_chunk: int = 3
    lang: str = "ja"


class DatasetConfig:
    """データセット設定"""

    # HuggingFace/ローカルデータセット設定
    DATASETS: Dict[str, DatasetInfo] = {
        "wikipedia_ja": DatasetInfo(
            name="Wikipedia日本語版",
            icon="📚",
            description="Wikipedia日本語版の記事データ（百科事典的知識）",
            hf_dataset="wikimedia/wikipedia",
            hf_config="20231101.ja",
            text_field="text",
            title_field="title",
            text_column="Combined_Text",
            file="OUTPUT/preprocessed_wikipedia_ja.csv",
            sample_size=1000,
            min_text_length=200,
            chunk_size=250,
            qa_per_chunk=3,
            lang="ja",
        ),
        "japanese_text": DatasetInfo(
            name="日本語Webテキスト（CC100）",
            icon="📰",
            description="日本語Webテキストコーパス",
            hf_dataset="range3/cc100-ja",
            text_field="text",
            text_column="Combined_Text",
            file="OUTPUT/preprocessed_japanese_text.csv",
            sample_size=1000,
            min_text_length=10,
            chunk_size=200,
            qa_per_chunk=2,
            lang="ja",
        ),
        "cc_news": DatasetInfo(
            name="CC-News（英語ニュース）",
            icon="🌐",
            description="Common Crawl英語ニュース記事",
            hf_dataset="cc_news",
            text_field="text",
            title_field="title",
            text_column="Combined_Text",
            file="OUTPUT/preprocessed_cc_news.csv",
            sample_size=500,
            min_text_length=100,
            chunk_size=300,
            qa_per_chunk=5,
            lang="en",
        ),
        "livedoor": DatasetInfo(
            name="Livedoorニュースコーパス",
            icon="📰",
            description="Livedoorニュース日本語記事（9カテゴリ、全7,376件）",
            download_url="https://www.rondhuit.com/download/ldcc-20140209.tar.gz",
            text_field="content",
            title_field="title",
            text_column="Combined_Text",
            file="OUTPUT/preprocessed_livedoor.csv",
            split=None,
            sample_size=7376,
            min_text_length=100,
            chunk_size=200,
            qa_per_chunk=4,
            lang="ja",
        ),
    }

    # RAG用データセット設定 (helper_rag.py互換)
    RAG_DATASETS: Dict[str, Dict[str, Any]] = {
        "customer_support_faq": {
            "name": "カスタマーサポート・FAQ",
            "icon": "💬",
            "required_columns": ["question", "answer"],
            "description": "カスタマーサポートFAQデータセット",
            "combine_template": "{question} {answer}",
            "port": 8501
        },
        "medical_qa": {
            "name": "医療QAデータ",
            "icon": "🏥",
            "required_columns": ["Question", "Complex_CoT", "Response"],
            "description": "医療質問回答データセット",
            "combine_template": "{question} {complex_cot} {response}",
            "port": 8503
        },
        "sciq_qa": {
            "name": "科学・技術QA（SciQ）",
            "icon": "🔬",
            "required_columns": ["question", "correct_answer"],
            "description": "科学・技術質問回答データセット",
            "combine_template": "{question} {correct_answer}",
            "port": 8504
        },
        "legal_qa": {
            "name": "法律・判例QA",
            "icon": "⚖️",
            "required_columns": ["question", "answer"],
            "description": "法律・判例質問回答データセット",
            "combine_template": "{question} {answer}",
            "port": 8505
        },
        "trivia_qa": {
            "name": "雑学QA（TriviaQA）",
            "icon": "🎯",
            "required_columns": ["question", "answer"],
            "description": "雑学質問回答データセット",
            "combine_template": "{question} {answer} {entity_pages} {search_results}",
            "port": 8506
        }
    }

    @classmethod
    def get_dataset(cls, dataset_type: str) -> Optional[DatasetInfo]:
        """データセット情報を取得"""
        return cls.DATASETS.get(dataset_type)

    @classmethod
    def get_dataset_dict(cls, dataset_type: str) -> Dict[str, Any]:
        """データセット情報を辞書形式で取得（後方互換性用）"""
        info = cls.DATASETS.get(dataset_type)
        if info is None:
            return {}

        return {
            "name": info.name,
            "icon": info.icon,
            "description": info.description,
            "file": info.file,
            "hf_dataset": info.hf_dataset,
            "hf_config": info.hf_config,
            "download_url": info.download_url,
            "split": info.split,
            "text_field": info.text_field,
            "title_field": info.title_field,
            "text_column": info.text_column,
            "sample_size": info.sample_size,
            "min_text_length": info.min_text_length,
            "chunk_size": info.chunk_size,
            "qa_per_chunk": info.qa_per_chunk,
            "lang": info.lang,
        }

    @classmethod
    def get_all_dataset_names(cls) -> List[str]:
        """全データセット名のリストを取得"""
        return list(cls.DATASETS.keys())

    @classmethod
    def get_rag_config(cls, dataset_type: str) -> Dict[str, Any]:
        """RAGデータセット設定を取得"""
        return cls.RAG_DATASETS.get(dataset_type, {
            "name": "未知のデータセット",
            "icon": "❓",
            "required_columns": [],
            "description": "未知のデータセット",
            "combine_template": "{}",
            "port": 8500
        })


# ===================================================================
# Q/A生成設定
# ===================================================================

class QAGenerationConfig:
    """Q/A生成設定"""

    # 質問タイプ階層構造
    QUESTION_TYPES_HIERARCHY: Dict[str, Dict[str, str]] = {
        "basic": {
            "definition": "定義型（〜とは何ですか？）",
            "identification": "識別型（〜の例を挙げてください）",
            "enumeration": "列挙型（〜の種類/要素は？）"
        },
        "understanding": {
            "cause_effect": "因果関係型（〜の結果/影響は？）",
            "process": "プロセス型（〜はどのように行われますか？）",
            "mechanism": "メカニズム型（〜の仕組みは？）",
            "comparison": "比較型（〜と〜の違いは？）"
        },
        "application": {
            "synthesis": "統合型（〜を組み合わせるとどうなりますか？）",
            "evaluation": "評価型（〜の長所と短所は？）",
            "prediction": "予測型（〜の場合どうなりますか？）",
            "practical": "実践型（〜はどのように活用されますか？）"
        }
    }

    # デフォルトのカバレージ閾値
    DEFAULT_COVERAGE_THRESHOLD: float = 0.58

    # デフォルトのバッチサイズ
    DEFAULT_BATCH_CHUNKS: int = 3

    # デフォルトのトークン制限
    DEFAULT_MIN_TOKENS: int = 150
    DEFAULT_MAX_TOKENS: int = 400


# ===================================================================
# Qdrant設定
# ===================================================================

class QdrantConfig:
    """Qdrant設定"""

    HOST: str = "localhost"
    PORT: int = 6333
    URL: str = f"http://{HOST}:{PORT}"
    DOCKER_IMAGE: str = "qdrant/qdrant"
    HEALTH_CHECK_ENDPOINT: str = "/collections"
    DEFAULT_TIMEOUT: int = 30
    DEFAULT_VECTOR_SIZE: int = 1536  # text-embedding-3-small
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"


# ===================================================================
# パス設定
# ===================================================================

class PathConfig:
    """パス設定"""

    BASE_DIR: Path = Path(__file__).parent
    OUTPUT_DIR: Path = BASE_DIR / "OUTPUT"
    QA_OUTPUT_DIR: Path = BASE_DIR / "qa_output"
    DATASETS_DIR: Path = BASE_DIR / "datasets"
    TEMP_DIR: Path = BASE_DIR / "temp_uploads"
    LOG_DIR: Path = BASE_DIR / "logs"

    @classmethod
    def ensure_dirs(cls) -> None:
        """必要なディレクトリを作成"""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.QA_OUTPUT_DIR.mkdir(exist_ok=True)
        cls.DATASETS_DIR.mkdir(exist_ok=True)
        cls.TEMP_DIR.mkdir(exist_ok=True)
        cls.LOG_DIR.mkdir(exist_ok=True)


# ===================================================================
# Celery設定
# ===================================================================

class CeleryConfig:
    """Celery設定"""

    BROKER_URL: str = "redis://localhost:6379/0"
    RESULT_BACKEND: str = "redis://localhost:6379/0"
    TASK_SERIALIZER: str = "json"
    ACCEPT_CONTENT: List[str] = ["json"]
    RESULT_SERIALIZER: str = "json"
    TIMEZONE: str = "Asia/Tokyo"
    ENABLE_UTC: bool = True
    TASK_TIME_LIMIT: int = 300  # 5分
    TASK_SOFT_TIME_LIMIT: int = 240  # 4分
    WORKER_CONCURRENCY: int = 8  # Gemini APIレート制限対策のためデフォルトを8に設定
    WORKER_PREFETCH_MULTIPLIER: int = 1


# ===================================================================
# Gemini API設定
# ===================================================================

class GeminiConfig:
    """Gemini 3 API設定"""

    # 利用可能なモデル一覧
    AVAILABLE_MODELS: List[str] = [
        "gemini-3-pro-preview",
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-preview",
        "gemini-2.5-pro-preview",
        "gemini-2.0-flash",
    ]

    # デフォルトモデル
    DEFAULT_MODEL: str = "gemini-2.0-flash"

    # Embeddingモデル
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # Embedding次元数（3072: Gemini 3最大精度）
    EMBEDDING_DIMS: int = 3072

    # 思考レベル
    DEFAULT_THINKING_LEVEL: str = "low"  # "low" or "high"

    # 温度設定（Gemini 3推奨: 1.0）
    DEFAULT_TEMPERATURE: float = 1.0

    # モデル料金（1000トークンあたりのドル）
    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        "gemini-3-pro-preview": {"input": 0.002, "output": 0.012},
        "gemini-3-pro-image-preview": {"input": 0.002, "output": 0.012},
        "gemini-2.5-flash-preview": {"input": 0.00015, "output": 0.0006},
        "gemini-2.5-pro-preview": {"input": 0.00125, "output": 0.005},
        "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
        "gemini-embedding-001": {"input": 0.0, "output": 0.0},  # 無料枠あり
    }

    # モデル制限
    MODEL_LIMITS: Dict[str, Dict[str, int]] = {
        "gemini-3-pro-preview": {"max_input_tokens": 1000000, "max_output_tokens": 64000},
        "gemini-3-pro-image-preview": {"max_input_tokens": 65000, "max_output_tokens": 32000},
        "gemini-2.5-flash-preview": {"max_input_tokens": 1000000, "max_output_tokens": 64000},
        "gemini-2.5-pro-preview": {"max_input_tokens": 1000000, "max_output_tokens": 64000},
        "gemini-2.0-flash": {"max_input_tokens": 1000000, "max_output_tokens": 8192},
    }

    @classmethod
    def get_model_limits(cls, model: str) -> Dict[str, int]:
        """モデルの制限を取得"""
        return cls.MODEL_LIMITS.get(
            model,
            {"max_input_tokens": 128000, "max_output_tokens": 8192}
        )

    @classmethod
    def get_model_pricing(cls, model: str) -> Dict[str, float]:
        """モデルの料金を取得"""
        return cls.MODEL_PRICING.get(model, {"input": 0.001, "output": 0.004})

    @classmethod
    def supports_thinking_level(cls, model: str) -> bool:
        """モデルがthinking_levelをサポートするかチェック"""
        return model.startswith("gemini-3")


# ===================================================================
# RAG Agent設定
# ===================================================================

class AgentConfig:
    """エージェント設定"""

    # RAG検索設定
    RAG_DEFAULT_COLLECTION: str = "qa_a02_qa_pairs_wikipedia_ja" # Default collection to search
    RAG_AVAILABLE_COLLECTIONS: List[str] = [ # List of available collections
        "qa_a02_qa_pairs_wikipedia_ja",
        "qa_a02_qa_pairs_livedoor",
        # Add other relevant collection names here
    ]
    RAG_SEARCH_LIMIT: int = 3
    RAG_SCORE_THRESHOLD: float = 0.50  # 検索結果として採用する最小スコア (0.7 -> 0.5に緩和)

    # エージェントモデル設定
    MODEL_NAME: str = GeminiConfig.DEFAULT_MODEL

    # ロギング設定
    CHAT_LOG_FILE_NAME: str = "agent_chat.log"
    CHAT_LOG_LEVEL: str = "INFO"


# ===================================================================
# LLMプロバイダー設定
# ===================================================================

class LLMProviderConfig:
    """LLMプロバイダー設定"""

    # デフォルトプロバイダー
    DEFAULT_LLM_PROVIDER: str = "gemini"  # "openai" or "gemini"
    DEFAULT_EMBEDDING_PROVIDER: str = "gemini"  # "openai" or "gemini"

    @classmethod
    def get_embedding_dims(cls, provider: Optional[str] = None) -> int:
        """プロバイダーに応じたEmbedding次元数を取得"""
        provider = provider or cls.DEFAULT_EMBEDDING_PROVIDER
        if provider.lower() == "gemini":
            return GeminiConfig.EMBEDDING_DIMS
        else:
            return QdrantConfig.DEFAULT_VECTOR_SIZE


# ===================================================================
# 後方互換性のためのエイリアス
# ===================================================================

# helper_rag.py の AppConfig 互換
class AppConfig(ModelConfig):
    """AppConfig互換クラス (後方互換性用)"""
    pass


# DATASET_CONFIGS 辞書形式（後方互換性用）
def get_dataset_configs() -> Dict[str, Dict[str, Any]]:
    """DATASET_CONFIGS辞書を取得（後方互換性用）"""
    return {
        name: DatasetConfig.get_dataset_dict(name)
        for name in DatasetConfig.get_all_dataset_names()
    }


# グローバル変数として公開（後方互換性用）
DATASET_CONFIGS = get_dataset_configs()
NO_TEMPERATURE_MODELS = ModelConfig.NO_TEMPERATURE_MODELS


def supports_temperature(model: str) -> bool:
    """temperatureサポートチェック（後方互換性用）"""
    return ModelConfig.supports_temperature(model)