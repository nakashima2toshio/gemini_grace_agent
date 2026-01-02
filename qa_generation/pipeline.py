#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/pipeline.py - Q/A生成パイプライン制御モジュール
"""

import sys
import logging
from typing import List, Dict, Optional, Any
import pandas as pd

from config import DATASET_CONFIGS
from helper_llm import LLMClient
from qa_generation.config import LOCAL_DATASET_EXTENSIONS
from qa_generation.structure import create_document_chunks, merge_small_chunks
from qa_generation.generation import QAGenerator, generate_qa_dataset
from qa_generation.evaluation import analyze_coverage
from celery_tasks import submit_unified_qa_generation, collect_results, check_celery_workers

logger = logging.getLogger(__name__)

class QAPipeline:
    """Q/A生成パイプライン"""

    def __init__(self,
                 dataset_name: Optional[str] = None,
                 input_file: Optional[str] = None,
                 model: str = "gemini-2.0-flash",
                 output_dir: str = "qa_output/pipeline",
                 max_docs: Optional[int] = None,
                 client: Optional[LLMClient] = None):
        """
        Args:
            dataset_name: データセット名 (cc_news, wikipedia_ja, etc.)
            input_file: ローカル入力ファイルパス
            model: 使用するモデル
            output_dir: 出力ディレクトリ
            max_docs: 最大処理文書数
            client: LLMクライアント（DI用）
        """
        self.dataset_name = dataset_name
        self.input_file = input_file
        self.model = model
        self.output_dir = output_dir
        self.max_docs = max_docs
        self.client = client  # Can be injected for testing or reuse

        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定をロード"""
        if self.input_file:
            # ローカルファイル用の動的設定
            from pathlib import Path
            file_basename = Path(self.input_file).stem
            lang = "ja" # デフォルト
            return {
                "name": f"ローカルファイル ({file_basename})",
                "text_column": "Combined_Text",
                "title_column": None,
                "lang": lang,
                "chunk_size": 300,
                "qa_per_chunk": 3,
                "type": "custom_upload"
            }
        elif self.dataset_name:
            if self.dataset_name not in DATASET_CONFIGS:
                raise ValueError(f"未対応のデータセット: {self.dataset_name}")
            
            config = DATASET_CONFIGS[self.dataset_name].copy() # config.pyのデータセット情報
            # a02の拡張設定をマージ (qa_generation.config.LOCAL_DATASET_EXTENSIONS)
            if self.dataset_name in LOCAL_DATASET_EXTENSIONS:
                config.update(LOCAL_DATASET_EXTENSIONS[self.dataset_name])
            
            config["type"] = self.dataset_name
            return config
        else:
            raise ValueError("dataset_name または input_file を指定してください")

    def load_data(self) -> pd.DataFrame:
        """データを読み込む"""
        # 循環参照を避けるため、ここでインポート (a02に残っている関数を使用...
        # 将来的には qa_generation/data_io.py に移動すべき)
        from qa_generation.data_io import load_uploaded_file, load_preprocessed_data
        
        logger.info("\n[1/4] データ読み込み...")
        if self.input_file:
            df = load_uploaded_file(self.input_file)
            if self.max_docs and len(df) > self.max_docs:
                df = df.head(self.max_docs)
                logger.info(f"  📊 最大文書数制限: {len(df)} 件に制限")
            return df
        else:
            return load_preprocessed_data(self.dataset_name)

    def create_chunks(self, df: pd.DataFrame, 
                      overlap_tokens: int = 0, 
                      use_similarity: bool = False, 
                      similarity_threshold: float = 0.7) -> List[Dict]:
        """チャンクを作成する"""
        logger.info("\n[2/4] チャンク作成...")
        dataset_type = self.config.get("type", "unknown")
        # ローカルファイルの場合、max_docsは読み込み時に適用済み
        max_docs_for_chunks = None if self.input_file else self.max_docs
        
        chunks = create_document_chunks(
            df, dataset_type, max_docs_for_chunks, config=self.config,
            overlap_tokens=overlap_tokens,
            use_similarity=use_similarity,
            similarity_threshold=similarity_threshold
        )
        
        if not chunks:
            logger.error("チャンクが作成されませんでした")
            # パイプラインとしてはここで例外を投げるべき
            raise RuntimeError("Chunk creation failed")
            
        return chunks

    def generate_qa(self, chunks: List[Dict],
                    use_celery: bool = False,
                    celery_workers: int = 8,
                    batch_chunks: int = 3,
                    merge_chunks: bool = True,
                    min_tokens: int = 150,
                    max_tokens: int = 400) -> List[Dict]:
        """Q/Aペアを生成する"""
        logger.info("\n[3/4] Q/Aペア生成...")
        
        if use_celery:
            return self._generate_with_celery(
                chunks, celery_workers, batch_chunks, merge_chunks, min_tokens, max_tokens
            )
        else:
            return self._generate_sync(
                chunks, batch_chunks, merge_chunks, min_tokens, max_tokens
            )

    def _generate_with_celery(self, chunks: List[Dict], workers: int, batch_size: int,
                              merge: bool, min_tokens: int, max_tokens: int) -> List[Dict]:
        """Celeryを使用した非同期生成"""
        logger.info(f"Celery並列処理モード: ワーカー数={workers}")
        logger.info("Celeryワーカーの状態を確認中...")
        if not check_celery_workers(workers):
            raise RuntimeError("Celery workers are not running")
            
        if merge:
            processed_chunks = merge_small_chunks(chunks, min_tokens, max_tokens)
        else:
            processed_chunks = chunks

        tasks = submit_unified_qa_generation(
            processed_chunks, self.config, self.model, provider="gemini"
        )

        timeout_seconds = min(max(len(tasks) * 10, 600), 1800)
        logger.info(f"結果収集タイムアウト: {timeout_seconds}秒（{len(tasks)}タスク）")
        return collect_results(tasks, timeout=timeout_seconds)

    def _generate_sync(self, chunks: List[Dict], batch_size: int,
                       merge: bool, min_tokens: int, max_tokens: int) -> List[Dict]:
        """同期生成"""
        logger.info("通常処理モード")
        dataset_type = self.config.get("type", "unknown")
        
        return generate_qa_dataset(
            chunks,
            dataset_type,
            self.model,
            chunk_batch_size=batch_size,
            merge_chunks=merge,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            config=self.config,
            client=self.client
        )

    def evaluate_coverage(self, chunks: List[Dict], qa_pairs: List[Dict], threshold: Optional[float] = None) -> Dict:
        """カバレッジを評価する"""
        logger.info("\n[4/4] カバレージ分析...")
        dataset_type = self.config.get("type", "unknown")
        return analyze_coverage(chunks, qa_pairs, dataset_type, custom_threshold=threshold)

    def save(self, qa_pairs: List[Dict], coverage_results: Dict) -> Dict[str, str]:
        """結果を保存する"""
        # 循環参照回避
        from qa_generation.data_io import save_results
        
        logger.info("\n結果を保存中...")
        dataset_type = self.config.get("type", "unknown")
        return save_results(qa_pairs, coverage_results, dataset_type, self.output_dir)

    def run(
            self,
            use_celery: bool = False,
            celery_workers: int = 8,
            batch_chunks: int = 3,
            merge_chunks: bool = True,
            min_tokens: int = 150,
            max_tokens: int = 400,
            analyze_coverage: bool = True,
            coverage_threshold: Optional[float] = None,
            overlap_tokens: int = 0,
            use_similarity: bool = False,
            similarity_threshold: float = 0.7):
        """パイプライン実行のショートカット"""
        try:
            df = self.load_data()
            chunks = self.create_chunks(
                df, 
                overlap_tokens=overlap_tokens, 
                use_similarity=use_similarity, 
                similarity_threshold=similarity_threshold
            )
            qa_pairs = self.generate_qa(
                chunks, use_celery, celery_workers, batch_chunks, merge_chunks, min_tokens, max_tokens
            )
            
            if not qa_pairs:
                logger.warning("Q/Aペアが生成されませんでした")
                
            coverage_results = {}
            if analyze_coverage and qa_pairs:
                coverage_results = self.evaluate_coverage(chunks, qa_pairs, coverage_threshold)
            else:
                coverage_results = {
                    "coverage_rate": 0,
                    "covered_chunks": 0,
                    "total_chunks": len(chunks),
                    "uncovered_chunks": []
                }
                
            saved_files = self.save(qa_pairs, coverage_results)
            
            return {
                "saved_files": saved_files,
                "qa_count": len(qa_pairs),
                "coverage_results": coverage_results,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"パイプライン実行エラー: {e}")
            raise
