#!/usr/bin/env python
# -*- coding: utf-8 -*-"""
"""
make_qa_register_qdrant.py - Q/A生成からQdrant登録までを完結する統合ツール：
make_qa.py + register_csv_to_qdrant.py の両方の処理。

python qa_qdrant/make_qa_register_qdrant.py --input-csv OUTPUT/cc_news_5per_chunked.csv --collection cc_news_5per --use-celery --celery-workers 16 --recreate
python qa_qdrant.mmake_qa_register_qdrant.py --input-csv OUTPUT/fineweb_edu_ja_5per_chunked.csv --collection fineweb_edu_ja_5per --use-celery --celery-workers 16 --recreate
python qa_qdrant.mmake_qa_register_qdrant.py --input-csv OUTPUT/japanese_text_5per_chunked.csv --collection japanese_text_5per --use-celery --celery-workers 16 --recreate
python qa_qdrant.mmake_qa_register_qdrant.py --input-csv OUTPUT/livedoor_5per_chunked.csv --collection livedoor_5per --use-celery --celery-workers 16 --recreate
python qa_qdrant.mmake_qa_register_qdrant.py --input-csv OUTPUT/fineweb_edu_ja_5per_chunked.csv--collection cc_news_5per --use-celery --celery-workers 16 --recreate

使用方法:
1. 事前定義されたデータセットから生成 & 登録:
python make_qa_register_qdrant.py \
 --dataset fineweb_edu_ja \
 --collection qa_fineweb_edu_ja \
 --use-celery \
 --celery-workers 16 \
 --recreate


2. テキストCSVファイルからQ/A生成 & 登録（NEW!）:
python make_qa_register_qdrant.py \
 --input-csv OUTPUT/cc_news_5per.csv \
 --collection cc_news_5per \
 --use-celery \
 --celery-workers 16 \
 --recreate

3. Q/AペアCSVから直接登録（Q/A生成スキップ）:
python make_qa_register_qdrant.py \
 --input-csv qa_output/qa_pairs_fineweb_edu_ja.csv \
 --collection cc_news_5per \
 --batch-size 100 \
 --recreate

処理フロー:
--input-csv の場合:
  1. CSVファイルを読み込み、カラムを確認
  2-A. question/answer カラムがある → Q/A生成スキップ → Qdrant登録
  2-B. text/Combined_Text カラムのみ → Q/A生成実行 → Qdrant登録
  
--dataset の場合:
  1. DATASET_CONFIGSから設定を取得
  2. Q/A生成実行
  3. Qdrant登録

========================================================================
新しいアーキテクチャに基づき、ドキュメントのチャンク化、Q/Aペアの生成、
そしてQdrantベクトルデータベースへの登録を一貫して行います。
# Flower起動
celery -A celery_config flower --port=5555

[Usage: ]
・celery起動：
redis cashe clear:
redis-cli -n 0 DEL qa_generation

# ステータス確認
./start_celery.sh status

# 停止
./start_celery.sh stop

# 再起動
./start_celery.sh restart -w 24

python make_qa_register_qdrant.py \
  --dataset fineweb_edu_ja \
  --collection qa_fineweb_edu_ja \
  --use-celery \
  --celery-workers 24 \
  --recreate

python make_qa_register_qdrant.py \
--dataset wikipedia_ja \
--collection qa_wikipedia_ja \
--use-celery \
--celery-workers 24 \
--recreate

"""

import sys
import os
# 🔧 プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def run_registration(csv_path: str, collection_name: str, recreate: bool, batch_size: int, provider: str):
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

    # 2. ベクトル化対象テキストの準備 (question + answer)
    if 'question' in df.columns and 'answer' in df.columns:
        texts = (df['question'].astype(str) + "\n" + df['answer'].astype(str)).tolist()
        logger.info("📝 ベクトル化対象: 'question' と 'answer' を結合")
    else:
        logger.error("Q/Aカラムが見つかりません。")
        return False

    # 3. Qdrant準備
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

    # 4. バッチ処理によるEmbedding生成と登録
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
                start_index=i  # バッチの開始インデックスを渡す
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

    # =================================================================
    # UI用正規化CSVの作成
    # =================================================================
    try:
        output_dir = "qa_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, normalized_filename)

        logger.info(f"📋 UI用ファイル作成: {output_path}")

        # カラム存在チェック
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
        description="統合ツール: Q/Aペア自動生成 & Qdrantデータベース登録"
    )

    # 生成パラメータ
    group_gen = parser.add_argument_group("QA Generation Options")
    group_gen.add_argument("--dataset", type=str, choices=list(DATASET_CONFIGS.keys()),
                           help="事前定義されたデータセット名（DATASET_CONFIGSから選択）")
    group_gen.add_argument("--input-csv", type=str,
                           help="入力CSVファイルのパス（--datasetの代わりに使用可能）")
    group_gen.add_argument("--model", type=str, default="gemini-2.0-flash")
    group_gen.add_argument("--max-docs", type=int, default=None)
    group_gen.add_argument("--use-celery", action="store_true", help="Celery並列処理を使用")
    group_gen.add_argument("--celery-workers", type=int, default=8)
    group_gen.add_argument("--batch-chunks", type=int, default=3)
    group_gen.add_argument("--merge-chunks", action="store_true", default=True)
    group_gen.add_argument("--overlap-tokens", type=int, default=0, help="チャンク間の重複トークン数")
    group_gen.add_argument("--use-similarity", action="store_true", help="ベクトル類似度によるセマンティック分割を使用")
    group_gen.add_argument("--similarity-threshold", type=float, default=0.7, help="セマンティック分割の類似度閾値")

    # 登録パラメータ
    group_reg = parser.add_argument_group("Qdrant Registration Options")
    group_reg.add_argument("--collection", type=str, required=True, help="登録先コレクション名")
    group_reg.add_argument("--recreate", action="store_true", help="コレクションを再作成")
    group_reg.add_argument("--batch-size", type=int, default=100, help="Embeddingバッチサイズ")
    group_reg.add_argument("--provider", type=str, default="gemini")

    args = parser.parse_args()

    # datasetとinput-csvのどちらか一方が必須
    if not args.dataset and not args.input_csv:
        logger.error("--dataset または --input-csv のいずれかを指定してください")
        sys.exit(1)

    if args.dataset and args.input_csv:
        logger.error("--dataset と --input-csv は同時に指定できません")
        sys.exit(1)

    # APIキー確認
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEYが設定されていません")
        sys.exit(1)

    try:
        # Phase 1: Q/A生成
        logger.info(f"\n" + "=" * 60)
        logger.info(f"Phase 1: QA Generation Pipeline")
        logger.info(f"=" * 60)

        # input-csvが指定された場合は、Q/A生成をスキップして直接登録へ
        if args.input_csv:
            if not os.path.exists(args.input_csv):
                logger.error(f"入力ファイルが見つかりません: {args.input_csv}")
                sys.exit(1)

            logger.info(f"📁 既存のCSVファイルを使用: {args.input_csv}")
            
            # CSVファイルの内容を確認
            try:
                df_check = pd.read_csv(args.input_csv)
                logger.info(f"✅ CSVファイル確認: {len(df_check)} 行")
                logger.info(f"   カラム: {list(df_check.columns)}")
            except Exception as e:
                logger.error(f"CSVファイルの読み込みエラー: {e}")
                sys.exit(1)
            
            # question/answerカラムの有無を確認
            has_qa_columns = 'question' in df_check.columns and 'answer' in df_check.columns
            has_text_columns = 'text' in df_check.columns or 'Combined_Text' in df_check.columns
            
            if has_qa_columns:
                # 既にQ/Aペアが存在する場合は生成をスキップ
                logger.info("✅ Q/Aカラムが存在します - Q/A生成をスキップして登録へ")
                generated_csv = args.input_csv
                qa_count = len(df_check)
                
            elif has_text_columns:
                # テキストカラムしかない場合はQ/A生成を実行
                logger.info("📝 テキストカラムのみ検出 - Q/A生成を実行します")
                logger.info(f"   use_celery: {args.use_celery}, workers: {args.celery_workers}")
                
                pipeline = QAPipeline(
                    dataset_name=None,
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
        else:
            # 従来通りdatasetを使用してQ/A生成
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

        # Phase 2: Qdrant登録
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

