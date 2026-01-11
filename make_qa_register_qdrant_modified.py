#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_qa_register_qdrant.py - Q/A生成からQdrant登録までを完結する統合ツール（改修版）
チャンクCSV読み込み機能を追加
"""

import sys
import os
import argparse
import logging
import re
import pandas as pd
from typing import List, Dict

# QA生成関連
from qa_generation.pipeline import QAPipeline
from config import DATASET_CONFIGS

# Qdrant登録関連
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


def normalize_source_filename(filename: str) -> str:
    """
    ファイル名から日時サフィックス（例: _20251230_232641）を除去して正規化する。
    UI(agent_rag.py)での参照を安定させるための処理。
    """
    normalized = re.sub(r'_\d{8}_\d{6}', '', filename)
    return normalized


def run_registration(csv_path: str, collection_name: str, recreate: bool,
                     batch_size: int, provider: str):
    """
    Qdrant登録ロジックの実行
    """
    logger.info(f"\n" + "=" * 60)
    logger.info(f"Phase 2: Qdrant Registration")
    logger.info(f"=" * 60)

    if not os.path.exists(csv_path):
        logger.error(f"入力ファイルが見つかりません: {csv_path}")
        return False

    logger.info(f"📁 ファイル読み込み中: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"   -> 読み込み完了: {len(df)} 行")
    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {e}")
        return False

    # ベクトル化対象テキストの準備 (question + answer)
    if 'question' in df.columns and 'answer' in df.columns:
        texts = (df['question'].astype(str) + "\n" + df['answer'].astype(str)).tolist()
        logger.info("📝 ベクトル化対象: 'question' と 'answer' を結合")
    else:
        logger.error("Q/Aカラムが見つかりません。")
        return False

    # Qdrant準備
    try:
        client = create_qdrant_client()
        if recreate:
            logger.info(f"🗑️ コレクション '{collection_name}' を再作成します...")
            create_or_recreate_collection_for_qdrant(client, collection_name, recreate=True)
        else:
            create_or_recreate_collection_for_qdrant(client, collection_name, recreate=False)
    except Exception as e:
        logger.error(f"Qdrant接続エラー: {e}")
        return False

    # バッチ処理によるEmbedding生成と登録
    total_processed = 0
    source_filename = os.path.basename(csv_path)
    normalized_filename = normalize_source_filename(source_filename)

    logger.info(f"🚀 登録処理開始 (全 {len(df)} 件, バッチサイズ: {batch_size})")

    try:
        for i in range(0, len(df), batch_size):
            end_idx = min(i + batch_size, len(df))
            batch_df = df.iloc[i: end_idx]
            batch_texts = texts[i: end_idx]

            # ベクトル化
            vectors = embed_texts_for_qdrant(batch_texts)
            if not vectors:
                logger.warning(f"   Batch {i}-{end_idx}: ベクトル生成失敗（スキップ）")
                continue

            # ポイント構築（グローバルインデックスを使用）
            points = build_points_for_qdrant(
                batch_df,
                vectors,
                domain=collection_name,
                source_file=normalized_filename,
                start_index=i
            )

            # source情報を確実に正規化名で登録
            for point in points:
                point.payload["source"] = normalized_filename

            # Qdrantへアップサート
            upsert_points_to_qdrant(client, collection_name, points)

            total_processed += len(points)
            logger.info(f"   ✅ 進捗: {total_processed} / {len(df)} 件完了")

    except Exception as e:
        logger.error(f"登録中にエラー発生: {e}")
        return False

    # UI用正規化CSVの作成
    try:
        output_dir = "qa_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, normalized_filename)

        logger.info(f"📋 UI用ファイル作成: {output_path}")

        if 'question' in df.columns and 'answer' in df.columns:
            df[['question', 'answer']].to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"   -> 作成完了")
        else:
            logger.warning(f"   -> 必要なカラム(question, answer)が見つからないためスキップ")

    except Exception as e:
        logger.warning(f"UI用ファイル作成失敗: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="統合ツール: Q/Aペア自動生成 & Qdrantデータベース登録（チャンクCSV対応）"
    )

    # ================================================================
    # 入力ソース（排他的）
    # ================================================================
    input_group = parser.add_argument_group("Input Source Options (choose one)")
    input_group.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="事前定義されたデータセット名"
    )
    input_group.add_argument(
        "--input-csv",
        type=str,
        help="入力CSVファイルのパス（テキストまたはQ/Aペア）"
    )
    input_group.add_argument(
        "--input-chunks",  # ✅ 新規追加
        type=str,
        help="事前作成されたチャンクCSVファイルのパス"
    )

    # ================================================================
    # QA生成パラメータ
    # ================================================================
    group_gen = parser.add_argument_group("QA Generation Options")
    group_gen.add_argument("--model", type=str, default="gemini-2.0-flash")
    group_gen.add_argument("--max-docs", type=int, default=None)
    group_gen.add_argument("--use-celery", action="store_true", help="Celery並列処理を使用")
    group_gen.add_argument("--celery-workers", type=int, default=8)
    group_gen.add_argument("--batch-chunks", type=int, default=3)
    group_gen.add_argument("--merge-chunks", action="store_true", default=True)
    group_gen.add_argument("--overlap-tokens", type=int, default=0)
    group_gen.add_argument("--use-similarity", action="store_true")
    group_gen.add_argument("--similarity-threshold", type=float, default=0.7)

    # ================================================================
    # Qdrant登録パラメータ
    # ================================================================
    group_reg = parser.add_argument_group("Qdrant Registration Options")
    group_reg.add_argument("--collection", type=str, required=True, help="登録先コレクション名")
    group_reg.add_argument("--recreate", action="store_true", help="コレクションを再作成")
    group_reg.add_argument("--batch-size", type=int, default=100, help="Embeddingバッチサイズ")
    group_reg.add_argument("--provider", type=str, default="gemini")

    args = parser.parse_args()

    # ================================================================
    # 入力検証
    # ================================================================
    input_count = sum([
        args.dataset is not None,
        args.input_csv is not None,
        args.input_chunks is not None  # ✅ 新規追加
    ])

    if input_count == 0:
        logger.error("--dataset, --input-csv, --input-chunks のいずれか1つを指定してください")
        sys.exit(1)

    if input_count > 1:
        logger.error("--dataset, --input-csv, --input-chunks は同時に指定できません")
        sys.exit(1)

    # APIキー確認
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEYが設定されていません")
        sys.exit(1)

    try:
        # ================================================================
        # Phase 1: Q/A生成
        # ================================================================
        logger.info(f"\n" + "=" * 60)
        logger.info(f"Phase 1: QA Generation Pipeline")
        logger.info(f"=" * 60)

        # ✅ チャンクCSVが指定された場合
        if args.input_chunks:
            logger.info(f"📁 チャンクCSVを使用: {args.input_chunks}")

            # パイプライン初期化
            pipeline = QAPipeline(
                input_chunks=args.input_chunks,  # ✅ 新規
                model=args.model,
                max_docs=args.max_docs
            )

            # Q/A生成実行
            result = pipeline.run(
                use_celery=args.use_celery,
                celery_workers=args.celery_workers,
                batch_chunks=args.batch_chunks,
                merge_chunks=args.merge_chunks,
                analyze_coverage=True
            )

            generated_csv = result['saved_files'].get('qa_csv')
            if not generated_csv or not os.path.exists(generated_csv):
                logger.error("Q/A生成フェーズでCSVファイルが作成されませんでした。")
                sys.exit(1)

            qa_count = result['qa_count']
            logger.info(f"✅ Q/A生成完了: {qa_count} ペア")

        # input-csvが指定された場合（既存ロジック）
        elif args.input_csv:
            if not os.path.exists(args.input_csv):
                logger.error(f"入力ファイルが見つかりません: {args.input_csv}")
                sys.exit(1)

            logger.info(f"📁 既存のCSVファイルを使用: {args.input_csv}")

            try:
                df_check = pd.read_csv(args.input_csv)
                logger.info(f"✅ CSVファイル確認: {len(df_check)} 行")
                logger.info(f"   カラム: {list(df_check.columns)}")
            except Exception as e:
                logger.error(f"CSVファイルの読み込みエラー: {e}")
                sys.exit(1)

            has_qa_columns = 'question' in df_check.columns and 'answer' in df_check.columns
            has_text_columns = 'text' in df_check.columns or 'Combined_Text' in df_check.columns

            if has_qa_columns:
                logger.info("✅ Q/Aカラムが存在します - Q/A生成をスキップして登録へ")
                generated_csv = args.input_csv
                qa_count = len(df_check)

            elif has_text_columns:
                logger.info("📝 テキストカラムのみ検出 - Q/A生成を実行します")

                pipeline = QAPipeline(
                    input_file=args.input_csv,
                    model=args.model,
                    max_docs=args.max_docs
                )

                result = pipeline.run(
                    use_celery=args.use_celery,
                    celery_workers=args.celery_workers,
                    batch_chunks=args.batch_chunks,
                    merge_chunks=args.merge_chunks,
                    analyze_coverage=True,
                    overlap_tokens=args.overlap_tokens,
                    use_similarity=args.use_similarity,
                    similarity_threshold=args.similarity_threshold
                )

                generated_csv = result['saved_files'].get('qa_csv')
                if not generated_csv or not os.path.exists(generated_csv):
                    logger.error("Q/A生成フェーズでCSVファイルが作成されませんでした。")
                    sys.exit(1)

                qa_count = result['qa_count']
                logger.info(f"✅ Q/A生成完了: {qa_count} ペア")

            else:
                logger.error("❌ CSVファイルに必要なカラムが見つかりません")
                logger.error("   必要なカラム: (question + answer) または (text または Combined_Text)")
                sys.exit(1)

        # datasetが指定された場合（既存ロジック）
        else:
            pipeline = QAPipeline(
                dataset_name=args.dataset,
                model=args.model,
                max_docs=args.max_docs
            )

            result = pipeline.run(
                use_celery=args.use_celery,
                celery_workers=args.celery_workers,
                batch_chunks=args.batch_chunks,
                merge_chunks=args.merge_chunks,
                analyze_coverage=True,
                overlap_tokens=args.overlap_tokens,
                use_similarity=args.use_similarity,
                similarity_threshold=args.similarity_threshold
            )

            generated_csv = result['saved_files'].get('qa_csv')
            if not generated_csv or not os.path.exists(generated_csv):
                logger.error("Q/A生成フェーズでCSVファイルが作成されませんでした。")
                sys.exit(1)

            qa_count = result['qa_count']
            logger.info(f"✅ Q/A生成完了: {qa_count} ペア")

        # ================================================================
        # Phase 2: Qdrant登録
        # ================================================================
        success = run_registration(
            csv_path=generated_csv,
            collection_name=args.collection,
            recreate=args.recreate,
            batch_size=args.batch_size,
            provider=args.provider
        )

        if success:
            logger.info(f"\n" + "=" * 60)
            logger.info(f"🎉 統合処理が正常に完了しました！")
            logger.info(f"   コレクション: {args.collection}")
            logger.info(f"   データ件数  : {qa_count} 件")
            logger.info(f"=" * 60)
        else:
            logger.error("\n❌ Qdrant登録フェーズで失敗しました。")

    except Exception as e:
        logger.error(f"致命的なエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()