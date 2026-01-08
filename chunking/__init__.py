# __init__.py
"""
chunking パッケージ

テキストを意味的なチャンクに分割するためのツール。
非同期・並列処理により高速化。
"""

from .models import (
    SentenceUnit,
    ParagraphUnit,
    StructuralResult,
    ContinuityResult
)
from .prompts import (
    PARAGRAPH_SEPARATION_PROMPT,
    SEMANTIC_CHUNKING_PROMPT,
    CONTINUITY_CHECK_PROMPT
)
from .async_api_client import (
    AsyncAPIClient,
    AdaptiveSemaphore
)
from .checkpoint_manager import CheckpointManager
from .csv_to_chunks_text_para import (
    LargeTextProcessorPara,
    chunk_overlap_para,
    chunks_all,
    chunks_all_async
)
from .utils import (
    show_paragraphs,
    setup_logging,
    format_time,
    format_size,
    estimate_api_calls
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "SentenceUnit",
    "ParagraphUnit",
    "StructuralResult",
    "ContinuityResult",
    # Prompts
    "PARAGRAPH_SEPARATION_PROMPT",
    "SEMANTIC_CHUNKING_PROMPT",
    "CONTINUITY_CHECK_PROMPT",
    # API Client
    "AsyncAPIClient",
    "AdaptiveSemaphore",
    # Checkpoint
    "CheckpointManager",
    # Main Processor
    "LargeTextProcessorPara",
    "chunk_overlap_para",
    "chunks_all",
    "chunks_all_async",
    # Utils
    "show_paragraphs",
    "setup_logging",
    "format_time",
    "format_size",
    "estimate_api_calls",
]
