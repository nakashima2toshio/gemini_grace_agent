#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import traceback
import os
from typing import Optional
from qa_generation.pipeline import QAPipeline

logger = logging.getLogger(__name__)

def run_qa_generator(
    dataset: Optional[str] = None,
    input_file: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    output_dir: str = "qa_output/a02",
    max_docs: Optional[int] = None,
    analyze_coverage: bool = False,
    batch_chunks: int = 3,
    merge_chunks: bool = True,
    min_tokens: int = 150,
    max_tokens: int = 400,
    use_celery: bool = False,
    celery_workers: int = 8,
    coverage_threshold: Optional[float] = None,
    log_callback=None,
    progress_callback=None 
):
    """
    Q/A生成プロセスのエントリーポイント（UIからの呼び出し用）
    """
    # ロガーのハンドラを設定してcallbackに流す
    handler = None
    if log_callback:
        class CallbackHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record)
                log_callback(msg)
        
        handler = CallbackHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        # Root logger to capture everything from pipeline modules
        logging.getLogger().addHandler(handler)

    try:
        logger.info("🚀 Q/A生成プロセスを開始します (New Architecture via Pipeline)")
        
        # APIキー確認
        if not os.getenv("GOOGLE_API_KEY"):
            logger.error("GOOGLE_API_KEYが設定されていません")
            return {"success": False, "error": "GOOGLE_API_KEY missing"}

        pipeline = QAPipeline(
            dataset_name=dataset,
            input_file=input_file,
            model=model,
            output_dir=output_dir,
            max_docs=max_docs
        )

        # パイプライン実行
        result = pipeline.run(
            use_celery=use_celery,
            celery_workers=celery_workers,
            batch_chunks=batch_chunks,
            merge_chunks=merge_chunks,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            analyze_coverage=analyze_coverage,
            coverage_threshold=coverage_threshold
        )

        return result

    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    finally:
        # ハンドラの削除
        if handler:
            logging.getLogger().removeHandler(handler)