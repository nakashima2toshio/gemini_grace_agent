#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
conftest.py - テスト共通フィクスチャ
====================================
pytest用のフィクスチャ定義
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir():
    """一時ディレクトリを作成"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_qa_df():
    """サンプルQ/AデータのDataFrame"""
    return pd.DataFrame({
        "question": [
            "Pythonとは何ですか？",
            "機械学習の基本は？",
            "RAGとは？"
        ],
        "answer": [
            "Pythonは汎用プログラミング言語です。",
            "機械学習は大量のデータからパターンを学習する技術です。",
            "RAGは検索拡張生成の略で、LLMに外部知識を組み込む手法です。"
        ]
    })


@pytest.fixture
def sample_text_df():
    """サンプルテキストデータのDataFrame"""
    return pd.DataFrame({
        "Combined_Text": [
            "これはテストテキスト1です。日本語の文章です。",
            "これはテストテキスト2です。もう少し長い文章です。",
            "これはテストテキスト3です。さらに別の内容の文章です。"
        ]
    })


@pytest.fixture
def mock_openai_client():
    """OpenAIクライアントのモック"""
    with patch("services.qdrant_service.OpenAI") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client

        # embeddings.createのモック
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
            MagicMock(embedding=[0.3] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response

        yield mock_client


@pytest.fixture
def mock_qdrant_client():
    """Qdrantクライアントのモック"""
    with patch("services.qdrant_service.QdrantClient") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client

        # get_collectionsのモック
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client.get_collections.return_value.collections = [mock_collection]

        # get_collectionのモック
        mock_info = MagicMock()
        mock_info.points_count = 100
        mock_info.vectors_count = 100
        mock_info.indexed_vectors_count = 100
        mock_info.status = "green"
        mock_info.config.params.vectors.size = 1536
        mock_info.config.params.vectors.distance = "Cosine"
        mock_client.get_collection.return_value = mock_info

        yield mock_client


@pytest.fixture
def qa_output_dir(temp_dir):
    """qa_output/ディレクトリを作成してCSVファイルを配置"""
    qa_dir = temp_dir / "qa_output"
    qa_dir.mkdir()

    # サンプルCSVを作成
    sample_df = pd.DataFrame({
        "question": ["質問1", "質問2"],
        "answer": ["回答1", "回答2"]
    })
    csv_path = qa_dir / "test_qa.csv"
    sample_df.to_csv(csv_path, index=False)

    return qa_dir


@pytest.fixture
def output_dir(temp_dir):
    """OUTPUT/ディレクトリを作成"""
    out_dir = temp_dir / "OUTPUT"
    out_dir.mkdir()

    # サンプルCSVを作成
    sample_df = pd.DataFrame({
        "Combined_Text": ["テキスト1", "テキスト2"]
    })
    csv_path = out_dir / "preprocessed_test_20241126_120000.csv"
    sample_df.to_csv(csv_path, index=False)

    return out_dir