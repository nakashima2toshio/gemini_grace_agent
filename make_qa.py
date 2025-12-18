#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_qa.py - Q/Aペア生成 CLIエントリーポイント
============================================
新しいアーキテクチャに基づくQ/A生成ツール
"""

import sys
import os
import argparse
import logging
from qa_generation.pipeline import QAPipeline
from config import DATASET_CONFIGS

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="make_qa.py - Q/Aペア自動生成システム (New Architecture)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        default=None,
        help="処理するデータセット"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="ローカルQ/A CSVファイルのパス"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="使用するGeminiモデル"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qa_output/pipeline",
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="処理する最大文書数"
    )
    parser.add_argument(
        "--analyze-coverage",
        action="store_true",
        help="カバレージ分析を実行"
    )
    parser.add_argument(
        "--batch-chunks",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="1回のAPIで処理するチャンク数"
    )
    parser.add_argument(
        "--merge-chunks",
        action="store_true",
        default=True,
        help="小さいチャンクを統合する"
    )
    parser.add_argument(
        "--no-merge-chunks",
        dest="merge_chunks",
        action="store_false",
        help="チャンク統合を無効化"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=150,
        help="統合対象の最小トークン数"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="統合後の最大トークン数"
    )
    parser.add_argument(
        "--use-celery",
        action="store_true",
        help="Celeryによる非同期並列処理を使用"
    )
    parser.add_argument(
        "--celery-workers",
        type=int,
        default=8,
        help="Celeryワーカー数"
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=None,
        help="カバレージ判定の類似度閾値"
    )

    args = parser.parse_args()

    # 引数チェック
    if not args.dataset and not args.input_file:
        logger.error("--dataset または --input-file のいずれかを指定してください")
        sys.exit(1)
        
    if args.dataset and args.input_file:
        logger.error("--dataset と --input-file は同時に指定できません")
        sys.exit(1)

    # APIキー確認
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEYが設定されていません")
        sys.exit(1)

    try:
        # パイプラインの初期化
        pipeline = QAPipeline(
            dataset_name=args.dataset,
            input_file=args.input_file,
            model=args.model,
            output_dir=args.output,
            max_docs=args.max_docs
        )

        # パイプラインの実行
        result = pipeline.run(
            use_celery=args.use_celery,
            celery_workers=args.celery_workers,
            batch_chunks=args.batch_chunks,
            merge_chunks=args.merge_chunks,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            analyze_coverage=args.analyze_coverage,
            coverage_threshold=args.coverage_threshold
        )
        
        logger.info(f"Make QA 完了: {result['saved_files']['summary']}")
        logger.info(f"生成Q/A数: {result['qa_count']}")

    except Exception as e:
        logger.error(f"実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
