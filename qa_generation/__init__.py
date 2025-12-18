#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation - Q/A generation package
======================================
Core functionality for RAG Q&A system

Module structure:
- models.py: Pydantic data models (QAPair, QAPairsList, etc.)
- keyword_extraction.py: Keyword extraction (BestKeywordSelector, SmartKeywordSelector)
- semantic.py: Semantic coverage (SemanticCoverage)
- generation.py: Q/A generation (QAGenerator, generate_qa_dataset)
- structure.py: Chunk structure processing
- evaluation.py: Q/A evaluation
- pipeline.py: Generation pipeline
- content.py: Content processing
- data_io.py: Data I/O
- config.py: Configuration
"""

# ===================================================================
# Models
# ===================================================================
from qa_generation.models import (
    QAPair,
    QAPairsList,
    ChainOfThoughtAnalysis,
    ChainOfThoughtQAPair,
    ChainOfThoughtResponse,
    EnhancedQAPair,
    EnhancedQAPairsList,
    QAGenerationConsiderations,
)

# ===================================================================
# Keyword Extraction
# ===================================================================
from qa_generation.keyword_extraction import (
    BestKeywordSelector,
    SmartKeywordSelector,
    get_best_keywords,
    get_smart_keywords,
)

# ===================================================================
# Semantic Coverage
# ===================================================================
from qa_generation.semantic import SemanticCoverage

# ===================================================================
# Export
# ===================================================================
__all__ = [
    # Models
    "QAPair",
    "QAPairsList",
    "ChainOfThoughtAnalysis",
    "ChainOfThoughtQAPair",
    "ChainOfThoughtResponse",
    "EnhancedQAPair",
    "EnhancedQAPairsList",
    "QAGenerationConsiderations",
    # Keyword extraction
    "BestKeywordSelector",
    "SmartKeywordSelector",
    "get_best_keywords",
    "get_smart_keywords",
    # Semantic coverage
    "SemanticCoverage",
]
