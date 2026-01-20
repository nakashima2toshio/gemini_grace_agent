#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_qa_register_qdrant.py - Q/A生成からQdrant登録までを完結する統合ツール（リファクタリング版）

python make_qa_register_qdrant.py \
  --input-file test.txt \
  --collection test \
  --use-celery \
  --recreate

改修内容:
- --input-file（テキスト/CSV両対応）
- --outputオプションを追加（出力先の柔軟化）
- --ui-outputオプションを追加（UI用CSV出力先の柔軟化）
- スマートQ/A生成をデフォルトで有効化（SmartQAGenerator使用）
  - LLMによる動的Q/A数決定（0-5個）
  - 内容の重要度・複雑さを考慮
  - --no-smart-generationで従来方式に戻すことも可能
"""

import sys
import os
import argparse
import logging
import re
import pandas as pd
from typing import List, Dict
from pathlib import Path

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


def run_registration(
        csv_path: str,
        collection_name: str,
        recreate: bool,
        batch_size: int,
        provider: str,
        ui_output_dir: str = "qa_output"  # ✅ 新規追加
) -> bool:
    """
    Qdrant登録ロジックの実行

    Args:
        csv_path: Q/AペアCSVのパス
        collection_name: Qdrantコレクション名
        recreate: コレクションを再作成するか
        batch_size: Embeddingバッチサイズ
        provider: Embeddingプロバイダー
        ui_output_dir: UI用正規化CSVの出力ディレクトリ（デフォルト: qa_output）

    Returns:
        bool: 成功時True、失敗時False
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
        output_dir = ui_output_dir  # ✅ パラメータ使用
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
        description="統合ツール: Q/Aペア自動生成 & Qdrantデータベース登録（リファクタリング版）"
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
        "--input-file",  # ✅ --input-csvから変更、テキストとCSV両対応
        type=str,
        help="入力ファイルのパス（.txt, .csv）"
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
    # ✨ スマート生成オプション（デフォルト: True）
    group_gen.add_argument(
        "--use-smart-generation",
        action="store_true",
        default=True,
        help="スマートQ/A生成を使用（LLMによる動的Q/A数決定、デフォルト有効）"
    )
    group_gen.add_argument(
        "--no-smart-generation",
        dest="use_smart_generation",
        action="store_false",
        help="従来方式のQ/A生成を使用（トークン数ベース）"
    )

    # ================================================================
    # Qdrant登録パラメータ
    # ================================================================
    group_reg = parser.add_argument_group("Qdrant Registration Options")
    group_reg.add_argument("--collection", type=str, required=True, help="登録先コレクション名")
    group_reg.add_argument("--recreate", action="store_true", help="コレクションを再作成")
    group_reg.add_argument("--batch-size", type=int, default=100, help="Embeddingバッチサイズ")
    group_reg.add_argument("--provider", type=str, default="gemini")

    # ================================================================
    # 出力パラメータ ✅ 新規追加
    # ================================================================
    group_output = parser.add_argument_group("Output Options")
    group_output.add_argument(
        "--output",
        type=str,
        default="qa_output/pipeline",
        help="Q/AペアCSVの出力ディレクトリ（デフォルト: qa_output/pipeline）"
    )
    group_output.add_argument(
        "--ui-output",
        type=str,
        default="qa_output",
        help="UI用正規化CSVの出力ディレクトリ（デフォルト: qa_output）"
    )

    args = parser.parse_args()

    # ================================================================
    # 入力検証
    # ================================================================
    input_count = sum([
        args.dataset is not None,
        args.input_file is not None,  # ✅ input_csvから変更
    ])

    if input_count == 0:
        logger.error("--dataset, --input-file のいずれか1つを指定してください")
        sys.exit(1)

    if input_count > 1:
        logger.error("--dataset, --input-file は同時に指定できません")
        sys.exit(1)

    # APIキー確認
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEYが設定されていません")
        sys.exit(1)

    # ✨ スマート生成モードのログ表示
    logger.info("")
    logger.info("=" * 60)
    if args.use_smart_generation:
        logger.info("🆕 Q/A生成モード: スマート生成（デフォルト）")
        logger.info("   - LLMによる動的Q/A数決定（0-5個）")
        logger.info("   - 内容の重要度・複雑さを考慮")
        logger.info("   - 主要トピックを明示的にカバー")
        logger.info("   ※ 従来方式に戻す場合: --no-smart-generation")
    else:
        logger.info("🔧 Q/A生成モード: 従来方式（トークン数ベース）")
        logger.info("   - 固定的なQ/A数決定（2-8個）")
        logger.info("   ※ スマート生成に切り替える場合: --use-smart-generation")
    logger.info("=" * 60)

    try:
        # ================================================================
        # Phase 1: Q/A生成
        # ================================================================
        logger.info(f"\n" + "=" * 60)
        logger.info(f"Phase 1: QA Generation Pipeline")
        logger.info(f"=" * 60)

        # ✅ 入力ファイルが指定された場合
        if args.input_file:
            if not os.path.exists(args.input_file):
                logger.error(f"入力ファイルが見つかりません: {args.input_file}")
                sys.exit(1)

            logger.info(f"📁 入力ファイル: {args.input_file}")

            file_path = Path(args.input_file)

            # ✅ ファイル種別判定
            if file_path.suffix == '.txt':
                # テキストファイル → 常にチャンク作成 + Q/A生成
                logger.info("📝 テキストファイル検出 - チャンク作成 + Q/A生成を実行します")

                pipeline = QAPipeline(
                    input_file=args.input_file,
                    model=args.model,
                    output_dir=args.output,  # ✅ 出力先を指定
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
                    similarity_threshold=args.similarity_threshold,
                    use_smart_generation=args.use_smart_generation  # ✨ 追加
                )

                generated_csv = result['saved_files'].get('qa_csv')
                if not generated_csv or not os.path.exists(generated_csv):
                    logger.error("Q/A生成フェーズでCSVファイルが作成されませんでした。")
                    sys.exit(1)

                qa_count = result['qa_count']
                logger.info(f"✅ Q/A生成完了: {qa_count} ペア")

            elif file_path.suffix == '.csv':
                # CSV → カラムで判定
                try:
                    df_check = pd.read_csv(args.input_file)
                    logger.info(f"✅ CSVファイル確認: {len(df_check)} 行")
                    logger.info(f"   カラム: {list(df_check.columns)}")
                except Exception as e:
                    logger.error(f"CSVファイルの読み込みエラー: {e}")
                    sys.exit(1)

                has_qa_columns = 'question' in df_check.columns and 'answer' in df_check.columns
                has_text_columns = 'text' in df_check.columns or 'Combined_Text' in df_check.columns

                if has_qa_columns:
                    # Q/Aペア → Phase 1スキップ
                    logger.info("✅ Q/Aカラムが存在します - Q/A生成をスキップして登録へ")
                    generated_csv = args.input_file
                    qa_count = len(df_check)

                elif has_text_columns:
                    # テキストのみ → チャンク作成 + Q/A生成
                    logger.info("📝 テキストカラムのみ検出 - チャンク作成 + Q/A生成を実行します")

                    pipeline = QAPipeline(
                        input_file=args.input_file,
                        model=args.model,
                        output_dir=args.output,  # ✅ 出力先を指定
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
                        similarity_threshold=args.similarity_threshold,
                        use_smart_generation=args.use_smart_generation  # ✨ 追加
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

            else:
                logger.error(f"❌ 未対応のファイル形式: {file_path.suffix}")
                logger.error("   対応形式: .txt, .csv")
                sys.exit(1)

        # datasetが指定された場合
        else:
            pipeline = QAPipeline(
                dataset_name=args.dataset,
                model=args.model,
                output_dir=args.output,  # ✅ 出力先を指定
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
                similarity_threshold=args.similarity_threshold,
                use_smart_generation=args.use_smart_generation  # ✨ 追加
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
            provider=args.provider,
            ui_output_dir=args.ui_output  # ✅ 出力先を指定
        )

        if success:
            logger.info(f"\n" + "=" * 60)
            logger.info(f"🎉 統合処理が正常に完了しました！")
            logger.info(f"   コレクション: {args.collection}")
            logger.info(f"   データ件数  : {qa_count} 件")
            logger.info(f"   Q/A CSV     : {generated_csv}")
            logger.info(
                f"   UI用CSV     : {os.path.join(args.ui_output, normalize_source_filename(os.path.basename(generated_csv)))}")
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

