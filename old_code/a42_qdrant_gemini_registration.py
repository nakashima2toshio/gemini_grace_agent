#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a42_qdrant_gemini_registration.py - Gemini Embedding用Qdrant登録スクリプト
==========================================================================
Gemini 3の3072次元Embeddingを使用してQdrantにQ/Aデータを登録

使用例:
    # テスト（10件のみ）
    python a42_qdrant_gemini_registration.py --limit 10

    # 本番登録（全件）
    python a42_qdrant_gemini_registration.py --recreate

    # 既存コレクションを再作成して登録
    python a42_qdrant_gemini_registration.py --recreate --limit 100
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client_wrapper import (
    create_qdrant_client,
    create_collection_for_provider,
    load_csv_for_qdrant,
    build_inputs_for_embedding,
    embed_texts_unified,
    build_points,
    upsert_points,
    get_collection_stats,
    get_provider_vector_size,
    PROVIDER_DEFAULTS,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =====================================================
# 設定
# =====================================================

# Gemini用コレクション設定（3072次元）
GEMINI_COLLECTIONS = {
    "qa_livedoor_gemini": {
        "csv_path": "qa_output/a02_qa_pairs_livedoor.csv",
        "domain": "livedoor",
        "provider": "gemini",
    },
    "qa_wikipedia_gemini": {
        "csv_path": "qa_output/a02_qa_pairs_wikipedia_ja.csv",
        "domain": "wikipedia_ja",
        "provider": "gemini",
    },
    "qa_cc_news_gemini": {
        "csv_path": "qa_output/a02_qa_pairs_cc_news.csv",
        "domain": "cc_news",
        "provider": "gemini",
    },
}


def register_collection(
    client,
    collection_name: str,
    config: dict,
    recreate: bool = False,
    limit: int = 0,
    include_answer: bool = True
) -> dict:
    """
    単一コレクションの登録処理

    Args:
        client: Qdrantクライアント
        collection_name: コレクション名
        config: コレクション設定
        recreate: 再作成フラグ
        limit: 行数制限
        include_answer: 回答をEmbeddingに含めるか

    Returns:
        処理結果の辞書
    """
    csv_path = config["csv_path"]
    domain = config["domain"]
    provider = config.get("provider", "gemini")

    # 絶対パスに変換
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_csv_path = os.path.join(base_dir, csv_path)

    if not os.path.exists(full_csv_path):
        return {
            "collection": collection_name,
            "status": "error",
            "message": f"CSV not found: {full_csv_path}",
            "points": 0,
        }

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {collection_name}")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  Provider: {provider}")
    logger.info(f"  Vector Dims: {get_provider_vector_size(provider)}")
    logger.info(f"{'='*60}")

    try:
        # 1. CSVロード
        logger.info("Loading CSV...")
        df = load_csv_for_qdrant(full_csv_path, limit=limit)
        logger.info(f"  Loaded {len(df)} rows")

        if len(df) == 0:
            return {
                "collection": collection_name,
                "status": "error",
                "message": "CSV is empty",
                "points": 0,
            }

        # 2. コレクション作成
        logger.info("Creating collection...")
        create_collection_for_provider(
            client=client,
            name=collection_name,
            provider=provider,
            recreate=recreate
        )

        # 3. Embedding生成（Gemini: 3072次元）
        logger.info("Generating embeddings (Gemini 3072 dims)...")
        texts = build_inputs_for_embedding(df, include_answer=include_answer)
        vectors = embed_texts_unified(texts, provider=provider)
        logger.info(f"  Generated {len(vectors)} embeddings")
        logger.info(f"  Vector dims: {len(vectors[0]) if vectors else 0}")

        # 4. ポイント構築
        logger.info("Building points...")
        points = build_points(
            df=df,
            vectors=vectors,
            domain=domain,
            source_file=csv_path
        )

        # 5. Upsert
        logger.info("Upserting to Qdrant...")
        count = upsert_points(client, collection_name, points)
        logger.info(f"  Upserted {count} points")

        # 6. 統計確認
        stats = get_collection_stats(client, collection_name)
        logger.info(f"  Total points: {stats.get('total_points', 0) if stats else 0}")

        return {
            "collection": collection_name,
            "status": "success",
            "points": count,
            "vector_dims": len(vectors[0]) if vectors else 0,
            "provider": provider,
        }

    except Exception as e:
        logger.error(f"Error processing {collection_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "collection": collection_name,
            "status": "error",
            "message": str(e),
            "points": 0,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Gemini Embedding用Qdrant登録（3072次元）"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="既存コレクションを再作成"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="行数制限（0=無制限）"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="特定のコレクションのみ処理"
    )
    parser.add_argument(
        "--include-answer",
        action="store_true",
        default=True,
        help="回答をEmbeddingに含める"
    )

    args = parser.parse_args()

    # Qdrant接続
    logger.info("Connecting to Qdrant...")
    client = create_qdrant_client()

    # 処理対象コレクション
    if args.collection:
        if args.collection not in GEMINI_COLLECTIONS:
            logger.error(f"Unknown collection: {args.collection}")
            logger.info(f"Available: {list(GEMINI_COLLECTIONS.keys())}")
            sys.exit(1)
        collections = {args.collection: GEMINI_COLLECTIONS[args.collection]}
    else:
        collections = GEMINI_COLLECTIONS

    # 処理開始
    logger.info(f"\n{'='*60}")
    logger.info("Gemini 3 Embedding Registration")
    logger.info(f"  Provider: gemini")
    logger.info(f"  Vector Dims: 3072 (Gemini 3 Max Precision)")
    logger.info(f"  Recreate: {args.recreate}")
    logger.info(f"  Limit: {args.limit if args.limit > 0 else 'No limit'}")
    logger.info(f"  Collections: {list(collections.keys())}")
    logger.info(f"{'='*60}")

    results = []
    for name, config in collections.items():
        result = register_collection(
            client=client,
            collection_name=name,
            config=config,
            recreate=args.recreate,
            limit=args.limit,
            include_answer=args.include_answer
        )
        results.append(result)

    # サマリー
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    total_points = 0
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        points = r.get("points", 0)
        dims = r.get("vector_dims", 0)
        msg = r.get("message", "")

        logger.info(f"  [{status_icon}] {r['collection']}: {points} points ({dims} dims)")
        if msg:
            logger.info(f"      {msg}")

        if r["status"] == "success":
            total_points += points

    logger.info(f"\nTotal: {total_points} points registered")
    logger.info(f"Vector dimensions: 3072 (Gemini 3)")


if __name__ == "__main__":
    main()