#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
csv_to_chunks_text_para.py - 改修版
CSV出力機能を追加（メタデータ付き）
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import tiktoken

# 既存のインポート（元のファイルから）
from chunking.async_api_client import AsyncAPIClient
from chunking.checkpoint_manager import CheckpointManager
from chunking.models import StructuralResult, ParagraphUnit, ContinuityResult
from chunking.prompts import (
    PARAGRAPH_SEPARATION_PROMPT,
    SEMANTIC_CHUNKING_PROMPT,
    CONTINUITY_CHECK_PROMPT
)
from chunking.utils import (
    setup_logging,
    format_time,
    format_size,
    estimate_api_calls
)

logger = logging.getLogger(__name__)


# ================================================================
# ✅ 新規追加: CSV保存機能
# ================================================================

def save_chunks_as_csv(
    chunks: List[str],
    output_file: str,
    dataset_type: str = "custom",
    source_file: Optional[str] = None
) -> str:
    """
    チャンクをCSV形式で保存（メタデータ付き）
    
    Args:
        chunks: チャンクのリスト
        output_file: 出力ファイルパス
        dataset_type: データセット種別
        source_file: 元ファイル名
    
    Returns:
        保存したCSVファイルパス
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # メタデータ付きでデータ構築
    data = []
    for i, chunk_text in enumerate(chunks):
        # センテンスに分割（簡易版）
        sentences = _split_sentences_simple(chunk_text)
        
        data.append({
            'chunk_id': f"{dataset_type}_chunk_{i}",
            'text': chunk_text,
            'tokens': len(tokenizer.encode(chunk_text)),
            'chunk_idx': i,
            'dataset_type': dataset_type,
            'type': 'llm_chunk',  # LLMで作成されたチャンク
            'sentence_count': len(sentences),
            'source_file': source_file or ''
        })
    
    # DataFrameに変換してCSV保存
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"✅ CSV保存完了")
    logger.info(f"{'='*60}")
    logger.info(f"  ファイル: {output_file}")
    logger.info(f"  チャンク数: {len(df)}")
    logger.info(f"  総トークン数: {df['tokens'].sum()}")
    logger.info(f"  平均トークン数: {df['tokens'].mean():.1f}")
    logger.info(f"{'='*60}")
    
    return output_file


def save_chunks_as_text(chunks: List[str], output_file: str) -> str:
    """
    チャンクをテキスト形式で保存（既存形式・後方互換性のため維持）
    
    Args:
        chunks: チャンクのリスト
        output_file: 出力ファイルパス
    
    Returns:
        保存したファイルパス
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n---\n')
    
    logger.info(f"テキストファイル保存: {output_file} ({len(chunks)}チャンク)")
    return output_file


def _split_sentences_simple(text: str) -> List[str]:
    """
    簡易的な文分割（日本語対応）
    
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
# ✅ 改修: chunks_all_async 関数（出力形式を選択可能に）
# ================================================================

async def chunks_all_async(
    text: str,
    model: str = "gemini-2.0-flash-exp",
    max_workers: int = 8,
    block_size: int = 2000,
    checkpoint_manager: Optional[CheckpointManager] = None,
    output_file: Optional[str] = None,  # ✅ 新規追加
    dataset_type: str = "custom",       # ✅ 新規追加
    source_file: Optional[str] = None   # ✅ 新規追加
) -> List[str]:
    """
    テキストを3段階で意味的にチャンク化（非同期・並列処理）
    
    Args:
        text: 入力テキスト
        model: 使用するGeminiモデル
        max_workers: 並列ワーカー数
        block_size: バッチサイズ（文字数）
        checkpoint_manager: チェックポイント管理（オプション）
        output_file: 出力ファイルパス（拡張子で形式を判定）
        dataset_type: データセット種別
        source_file: 元ファイル名
    
    Returns:
        最終的なチャンクのリスト
    """
    import os
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEYが設定されていません")
    
    client = AsyncAPIClient(
        api_key=api_key,
        max_workers=max_workers,
        max_retries=3,
        max_output_tokens=4096
    )
    
    # チェックポイント管理
    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager()
    
    # 処理開始
    logger.info("="*60)
    logger.info("チャンク化処理開始 (3段階)")
    logger.info("="*60)
    logger.info(f"入力テキスト: {format_size(len(text))}")
    logger.info(f"モデル: {model}")
    logger.info(f"並列ワーカー数: {max_workers}")
    
    # Step 1: 階層構造化
    step1_chunks = await _step1_hierarchical_split(
        text, client, model, block_size, checkpoint_manager
    )
    
    # Step 2: 意味的分割
    step2_chunks = await _step2_semantic_chunking(
        step1_chunks, client, model, checkpoint_manager
    )
    
    # Step 3: 文脈連続性チェック
    final_chunks = await _step3_continuity_check(
        step2_chunks, client, model, checkpoint_manager
    )
    
    # ✅ 新規: 出力ファイルが指定されている場合は保存
    if output_file:
        output_path = Path(output_file)
        
        if output_path.suffix.lower() == '.csv':
            # CSV形式で保存
            save_chunks_as_csv(
                chunks=final_chunks,
                output_file=output_file,
                dataset_type=dataset_type,
                source_file=source_file
            )
        else:
            # テキスト形式で保存（既存形式）
            save_chunks_as_text(
                chunks=final_chunks,
                output_file=output_file
            )
    
    return final_chunks


# ================================================================
# 既存の内部関数（変更なし）
# ================================================================

async def _step1_hierarchical_split(
    text: str,
    client: AsyncAPIClient,
    model: str,
    block_size: int,
    checkpoint_manager: CheckpointManager
) -> List[str]:
    """Step 1: 階層構造化（段落 > 文）"""
    
    # チェックポイントチェック
    if checkpoint_manager.exists("step1"):
        logger.info("Step1: チェックポイントから再開")
        return checkpoint_manager.load("step1")
    
    logger.info("\n[Step 1/3] 階層構造化（段落 > 文）")
    
    # テキストをブロックに分割
    blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
    logger.info(f"  ブロック数: {len(blocks)}")
    
    # 並列処理
    tasks = []
    for i, block in enumerate(blocks):
        prompt = f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{block}"
        task = client.generate_content(
            model=model,
            contents=prompt,
            response_schema=StructuralResult,
            task_id=f"step1_block_{i}"
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # パラグラフを抽出
    paragraphs = []
    for result_json in results:
        if result_json:
            try:
                result = StructuralResult.model_validate_json(result_json)
                for para in result.paragraphs:
                    paragraphs.append(para.full_text)
            except Exception as e:
                logger.warning(f"パース失敗: {e}")
    
    logger.info(f"  出力: {len(paragraphs)} 段落")
    
    # チェックポイント保存
    checkpoint_manager.save("step1", paragraphs)
    
    return paragraphs


async def _step2_semantic_chunking(
    paragraphs: List[str],
    client: AsyncAPIClient,
    model: str,
    checkpoint_manager: CheckpointManager
) -> List[str]:
    """Step 2: 意味的分割"""
    
    # チェックポイントチェック
    if checkpoint_manager.exists("step2"):
        logger.info("Step2: チェックポイントから再開")
        return checkpoint_manager.load("step2")
    
    logger.info("\n[Step 2/3] 意味的分割")
    logger.info(f"  入力: {len(paragraphs)} 段落")
    
    # 並列処理
    tasks = []
    for i, para in enumerate(paragraphs):
        prompt = f"{SEMANTIC_CHUNKING_PROMPT}\n\n【入力テキスト】\n{para}"
        task = client.generate_content(
            model=model,
            contents=prompt,
            response_schema=StructuralResult,
            task_id=f"step2_para_{i}"
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # チャンクを抽出
    chunks = []
    for result_json in results:
        if result_json:
            try:
                result = StructuralResult.model_validate_json(result_json)
                for para in result.paragraphs:
                    chunks.append(para.full_text)
            except Exception as e:
                logger.warning(f"パース失敗: {e}")
    
    logger.info(f"  出力: {len(chunks)} チャンク")
    
    # チェックポイント保存
    checkpoint_manager.save("step2", chunks)
    
    return chunks


async def _step3_continuity_check(
    chunks: List[str],
    client: AsyncAPIClient,
    model: str,
    checkpoint_manager: CheckpointManager
) -> List[str]:
    """Step 3: 文脈連続性チェック"""
    
    # チェックポイントチェック
    if checkpoint_manager.exists("step3"):
        logger.info("Step3: チェックポイントから再開")
        return checkpoint_manager.load("step3")
    
    logger.info("\n[Step 3/3] 文脈連続性チェック")
    logger.info(f"  入力: {len(chunks)} チャンク")
    
    if len(chunks) <= 1:
        checkpoint_manager.save("step3", chunks)
        return chunks
    
    # 隣接チャンク間の連続性を判定
    tasks = []
    for i in range(len(chunks) - 1):
        prompt = f"{CONTINUITY_CHECK_PROMPT}\n\n【前のテキスト】\n{chunks[i]}\n\n【次のテキスト】\n{chunks[i+1]}"
        task = client.generate_content(
            model=model,
            contents=prompt,
            response_schema=ContinuityResult,
            task_id=f"step3_pair_{i}"
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # 連続性に基づいてマージ
    final_chunks = [chunks[0]]
    for i, result_json in enumerate(results):
        if result_json:
            try:
                result = ContinuityResult.model_validate_json(result_json)
                if result.is_connected:
                    # マージ
                    final_chunks[-1] += "\n\n" + chunks[i+1]
                else:
                    # 新規チャンク
                    final_chunks.append(chunks[i+1])
            except Exception as e:
                logger.warning(f"パース失敗: {e}")
                final_chunks.append(chunks[i+1])
        else:
            final_chunks.append(chunks[i+1])
    
    logger.info(f"  出力: {len(final_chunks)} チャンク（マージ後）")
    
    # チェックポイント保存
    checkpoint_manager.save("step3", final_chunks)
    
    return final_chunks


# ================================================================
# メイン関数（改修版）
# ================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="LLMベースのセマンティックチャンキング（CSV出力対応）"
    )
    parser.add_argument("-i", "--input", required=True, help="入力ファイル")
    parser.add_argument("-o", "--output", required=True, help="出力ファイル (.csv または .txt)")
    parser.add_argument("-m", "--model", default="gemini-2.0-flash-exp", help="モデル")
    parser.add_argument("-w", "--workers", type=int, default=8, help="並列ワーカー数")
    parser.add_argument("-b", "--block-size", type=int, default=2000, help="バッチサイズ")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細ログ")
    parser.add_argument("--resume", type=str, default=None, help="再開するジョブID")
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(verbose=args.verbose)
    
    # 入力ファイル読み込み
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"入力ファイルが見つかりません: {args.input}")
        return
    
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    
    logger.info(f"入力ファイル: {args.input}")
    logger.info(f"テキストサイズ: {format_size(len(text))}")
    
    # チェックポイント管理
    checkpoint_manager = CheckpointManager(job_id=args.resume) if args.resume else CheckpointManager()
    
    # データセット名を入力ファイル名から生成
    dataset_type = input_path.stem
    
    # チャンク化実行
    final_chunks = await chunks_all_async(
        text=text,
        model=args.model,
        max_workers=args.workers,
        block_size=args.block_size,
        checkpoint_manager=checkpoint_manager,
        output_file=args.output,      # ✅ 追加
        dataset_type=dataset_type,    # ✅ 追加
        source_file=input_path.name   # ✅ 追加
    )
    
    logger.info(f"\n✅ 処理完了: {len(final_chunks)} チャンク")


if __name__ == "__main__":
    asyncio.run(main())
