#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/pipeline.py - Q/A生成パイプライン制御モジュール（改修版）
CSV形式のチャンク読み込み機能を追加
"""

import sys
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
from pathlib import Path

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
                 input_chunks: Optional[str] = None,  # ✅ 新規追加: チャンクCSV
                 model: str = "gemini-2.0-flash",
                 output_dir: str = "qa_output/pipeline",
                 max_docs: Optional[int] = None,
                 client: Optional[LLMClient] = None):
        """
        Args:
            dataset_name: データセット名 (cc_news, wikipedia_ja, etc.)
            input_file: ローカル入力ファイルパス
            input_chunks: 事前作成されたチャンクCSVファイルパス（✅ 新規）
            model: 使用するモデル
            output_dir: 出力ディレクトリ
            max_docs: 最大処理文書数
            client: LLMクライアント（DI用）
        """
        self.dataset_name = dataset_name
        self.input_file = input_file
        self.input_chunks = input_chunks  # ✅ 新規
        self.model = model
        self.output_dir = output_dir
        self.max_docs = max_docs
        self.client = client

        # 引数の排他制御
        self._validate_inputs()
        
        self.config = self._load_config()

    def _validate_inputs(self):
        """入力パラメータの検証"""
        inputs = [self.dataset_name, self.input_file, self.input_chunks]
        non_none_count = sum(1 for x in inputs if x is not None)
        
        if non_none_count == 0:
            raise ValueError(
                "dataset_name, input_file, input_chunks のいずれか1つを指定してください"
            )
        
        if non_none_count > 1:
            raise ValueError(
                "dataset_name, input_file, input_chunks は同時に指定できません"
            )

    def _load_config(self) -> Dict[str, Any]:
        """設定をロード"""
        if self.input_chunks:
            # ✅ 新規: チャンクCSV用の動的設定
            chunk_path = Path(self.input_chunks)
            dataset_type = chunk_path.stem.replace('_chunks', '')
            
            return {
                "name": f"チャンクCSV ({chunk_path.name})",
                "text_column": "text",
                "title_column": None,
                "lang": "ja",  # デフォルト（CSVから推測可能なら変更）
                "chunk_size": 300,
                "qa_per_chunk": 3,
                "type": dataset_type
            }
        
        elif self.input_file:
            # ローカルファイル用の動的設定
            from pathlib import Path
            file_basename = Path(self.input_file).stem
            lang = "ja"  # デフォルト
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
            
            config = DATASET_CONFIGS[self.dataset_name].copy()
            # a02の拡張設定をマージ
            if self.dataset_name in LOCAL_DATASET_EXTENSIONS:
                config.update(LOCAL_DATASET_EXTENSIONS[self.dataset_name])
            
            config["type"] = self.dataset_name
            return config
        
        else:
            raise ValueError("設定の読み込みに失敗しました")

    def load_data(self) -> pd.DataFrame:
        """データを読み込む"""
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

    # ================================================================
    # ✅ 新規追加: チャンクCSV読み込み機能
    # ================================================================
    
    def load_chunks_from_csv(self) -> List[Dict]:
        """
        チャンクCSVを読み込んで既存形式に変換
        
        Returns:
            チャンクのリスト（既存形式）
        
        Raises:
            FileNotFoundError: ファイルが存在しない
            ValueError: 必須カラムが不足している
        """
        logger.info("\n[2/4] チャンクCSV読み込み...")
        
        # ファイル存在確認
        chunk_path = Path(self.input_chunks)
        if not chunk_path.exists():
            raise FileNotFoundError(f"チャンクファイルが見つかりません: {self.input_chunks}")
        
        logger.info(f"  📁 ファイル: {self.input_chunks}")
        
        # CSV読み込み
        try:
            df = pd.read_csv(self.input_chunks)
        except Exception as e:
            raise ValueError(f"CSV読み込みエラー: {e}")
        
        logger.info(f"  📊 読み込み: {len(df)} チャンク")
        
        # 必須カラムのチェック
        required_cols = ['chunk_id', 'text', 'tokens', 'chunk_idx']
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            logger.error(f"  ❌ 必須カラムが不足: {missing_cols}")
            logger.error(f"  現在のカラム: {list(df.columns)}")
            raise ValueError(f"必須カラムが不足しています: {missing_cols}")
        
        # 既存形式に変換
        chunks = []
        for idx, row in df.iterrows():
            # センテンス情報の再生成（オプション）
            sentences = self._split_sentences(row['text'])
            
            chunk = {
                'id': row['chunk_id'],
                'text': row['text'],
                'tokens': int(row['tokens']),
                'chunk_idx': int(row['chunk_idx']),
                'type': row.get('type', 'llm_chunk'),
                'dataset_type': row.get('dataset_type', 'custom'),
                'sentences': sentences,
                'sentence_count': row.get('sentence_count', len(sentences)),
                'source_file': row.get('source_file', ''),
                # 追加のメタデータ
                'doc_id': row.get('doc_id', f"doc_{idx}"),
                'doc_idx': row.get('doc_idx', 0),
            }
            
            chunks.append(chunk)
        
        # 統計情報をログ出力
        total_tokens = sum(c['tokens'] for c in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        
        logger.info(f"  ✅ 変換完了:")
        logger.info(f"     - チャンク数: {len(chunks)}")
        logger.info(f"     - 総トークン数: {total_tokens:,}")
        logger.info(f"     - 平均トークン数: {avg_tokens:.1f}")
        logger.info(f"     - データセット種別: {chunks[0]['dataset_type'] if chunks else 'N/A'}")
        
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        テキストを文に分割（簡易版）
        
        Args:
            text: 分割対象テキスト
        
        Returns:
            文のリスト
        """
        import re
        
        # 句点・疑問符・感嘆符で分割
        sentences = re.findall(r'[^。．.！？!?]+[。．.！？!?]\s*', text)
        
        if not sentences:
            # 句点がない場合は全体を1文とする
            sentences = [text.strip()] if text.strip() else []
        else:
            # 最後に句点がない残りテキストを追加
            last_pos = text.rfind(sentences[-1]) + len(sentences[-1])
            if last_pos < len(text):
                remaining = text[last_pos:].strip()
                if remaining:
                    sentences.append(remaining)
        
        return [s.strip() for s in sentences if s.strip()]

    # ================================================================
    # 既存のメソッド（一部変更）
    # ================================================================

    def create_chunks(self, df: pd.DataFrame, 
                      overlap_tokens: int = 0, 
                      use_similarity: bool = False, 
                      similarity_threshold: float = 0.7,
                      max_workers: int = 8) -> List[Dict]:
        """チャンクを作成する（既存メソッド）"""
        logger.info("\n[2/4] チャンク作成...")
        dataset_type = self.config.get("type", "unknown")
        max_docs_for_chunks = None if self.input_file else self.max_docs
        
        chunks = create_document_chunks(
            df, dataset_type, max_docs_for_chunks, config=self.config,
            overlap_tokens=overlap_tokens,
            use_similarity=use_similarity,
            similarity_threshold=similarity_threshold,
            max_workers=max_workers
        )
        
        if not chunks:
            logger.error("チャンクが作成されませんでした")
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

    def evaluate_coverage(self, chunks: List[Dict], qa_pairs: List[Dict], 
                         threshold: Optional[float] = None) -> Dict:
        """カバレッジを評価する"""
        logger.info("\n[4/4] カバレージ分析...")
        dataset_type = self.config.get("type", "unknown")
        return analyze_coverage(chunks, qa_pairs, dataset_type, custom_threshold=threshold)

    def save(self, qa_pairs: List[Dict], coverage_results: Dict) -> Dict[str, str]:
        """結果を保存する"""
        from qa_generation.data_io import save_results
        
        logger.info("\n結果を保存中...")
        dataset_type = self.config.get("type", "unknown")
        return save_results(qa_pairs, coverage_results, dataset_type, self.output_dir)

    # ================================================================
    # ✅ 改修: run メソッド（チャンクCSV対応）
    # ================================================================

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
        """
        パイプライン実行のショートカット
        
        ✅ 改修ポイント:
        - input_chunks が指定されている場合は、CSVからチャンクを読み込む
        - それ以外は既存のフローと同じ
        """
        try:
            # ================================================================
            # チャンクの取得（3つのパターン）
            # ================================================================
            
            if self.input_chunks:
                # ✅ パターン1: チャンクCSVから読み込み
                logger.info("="*60)
                logger.info("モード: チャンクCSV読み込み")
                logger.info("="*60)
                chunks = self.load_chunks_from_csv()
                
            else:
                # パターン2 & 3: 既存のフロー
                logger.info("="*60)
                logger.info("モード: 通常チャンク作成")
                logger.info("="*60)
                df = self.load_data()
                chunks = self.create_chunks(
                    df, 
                    overlap_tokens=overlap_tokens, 
                    use_similarity=use_similarity, 
                    similarity_threshold=similarity_threshold,
                    max_workers=celery_workers
                )
            
            # ================================================================
            # 以降は共通処理
            # ================================================================
            
            # Q/A生成
            qa_pairs = self.generate_qa(
                chunks, use_celery, celery_workers, batch_chunks, 
                merge_chunks, min_tokens, max_tokens
            )
            
            if not qa_pairs:
                logger.warning("Q/Aペアが生成されませんでした")
            
            # カバレージ分析
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
            
            # 結果保存
            saved_files = self.save(qa_pairs, coverage_results)
            
            # 返り値
            return {
                "saved_files": saved_files,
                "qa_count": len(qa_pairs),
                "coverage_results": coverage_results,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"パイプライン実行エラー: {e}")
            raise
