# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
register_csv_to_qdrant.py - 既存CSVをQdrantに登録する汎用スクリプト

Usage:
    python register_csv_to_qdrant.py <csv_path> <collection_name> [--recreate] [--batch-size N]

Examples:
    python register_csv_to_qdrant.py qa_output/qa_pairs_fineweb_edu_ja.csv qa_fineweb_edu_ja --recreate
    python register_csv_to_qdrant.py qa_output/qa_pairs_wikipedia_ja.csv qa_wikipedia_ja --recreate --batch-size 50
"""
import sys
import os
import argparse
import logging
import pandas as pd
from services.qdrant_service import (
    create_or_recreate_collection_for_qdrant,
    embed_texts_for_qdrant,
    upsert_points_to_qdrant,
    build_points_for_qdrant
)
from qdrant_client_wrapper import create_qdrant_client

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def register_csv_to_qdrant(
        csv_path: str,
        collection_name: str,
        recreate: bool = True,
        batch_size: int = 100
):
    """
    CSVファイルをQdrantコレクションに登録

    Args:
        csv_path: CSVファイルパス
        collection_name: 登録先コレクション名
        recreate: コレクションを再作成するか
        batch_size: バッチサイズ
    """
    # ファイル存在確認
    if not os.path.exists(csv_path):
        logger.error(f"❌ ファイルが見つかりません: {csv_path}")
        return False

    # CSV読み込み
    logger.info(f"📁 ファイル読み込み中: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"   -> CSV件数: {len(df)}")

    # 必須カラム確認
    if 'question' not in df.columns or 'answer' not in df.columns:
        logger.error("❌ CSVに 'question' と 'answer' カラムが必要です")
        logger.error(f"   現在のカラム: {list(df.columns)}")
        return False

    # ベクトル化対象テキストの準備
    texts = (df['question'].astype(str) + "\n" + df['answer'].astype(str)).tolist()
    source_file = os.path.basename(csv_path)

    # Qdrantクライアント作成
    logger.info(f"🔌 Qdrant接続中...")
    try:
        client = create_qdrant_client()
    except Exception as e:
        logger.error(f"❌ Qdrant接続エラー: {e}")
        return False

    # コレクション作成
    if recreate:
        logger.info(f"🗑️ コレクション '{collection_name}' を再作成します...")
    else:
        logger.info(f"📦 コレクション '{collection_name}' を作成/確認します...")

    create_or_recreate_collection_for_qdrant(client, collection_name, recreate=recreate)

    # バッチ処理
    logger.info(f"🚀 登録処理開始 (全 {len(df)} 件, バッチサイズ: {batch_size})")
    total = 0

    try:
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]

            # ベクトル化
            vectors = embed_texts_for_qdrant(batch_texts)

            # ポイント構築（start_indexでグローバルインデックスを使用）
            points = build_points_for_qdrant(
                batch_df,
                vectors,
                domain=collection_name,
                source_file=source_file,
                start_index=i
            )

            # Qdrantへアップサート
            upsert_points_to_qdrant(client, collection_name, points)

            total += len(points)
            logger.info(f"   ✅ 進捗: {total}/{len(df)} 件完了")

    except Exception as e:
        logger.error(f"❌ 登録中にエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info(f"")
    logger.info(f"=" * 60)
    logger.info(f"🎉 登録完了！")
    logger.info(f"   コレクション: {collection_name}")
    logger.info(f"   登録件数: {total} 件")
    logger.info(f"=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="既存CSVをQdrantに登録する汎用スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python register_csv_to_qdrant.py qa_output/qa_pairs_fineweb_edu_ja.csv qa_fineweb_edu_ja --recreate
    python register_csv_to_qdrant.py qa_output/qa_pairs_wikipedia_ja.csv qa_wikipedia_ja --batch-size 50
        """
    )

    parser.add_argument("csv_path", type=str, help="登録するCSVファイルのパス")
    parser.add_argument("collection_name", type=str, help="Qdrantコレクション名")
    parser.add_argument("--recreate", action="store_true", default=True,
                        help="コレクションを再作成 (デフォルト: True)")
    parser.add_argument("--no-recreate", action="store_false", dest="recreate",
                        help="既存コレクションに追加")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="バッチサイズ (デフォルト: 100)")

    args = parser.parse_args()

    # APIキー確認
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("❌ GOOGLE_API_KEYが設定されていません")
        sys.exit(1)

    # 実行
    success = register_csv_to_qdrant(
        csv_path=args.csv_path,
        collection_name=args.collection_name,
        recreate=args.recreate,
        batch_size=args.batch_size
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

