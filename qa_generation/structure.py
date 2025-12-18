#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/structure.py - チャンク作成・統合モジュール
"""

import logging
from typing import List, Dict, Optional
import pandas as pd
import tiktoken
from qa_generation.semantic import SemanticCoverage
from config import DATASET_CONFIGS

logger = logging.getLogger(__name__)

def create_semantic_chunks(text: str, lang: str = "ja", max_tokens: int = 200, chunk_id_prefix: str = "chunk") -> List[Dict]:
    """
    セマンティック分割によるチャンク作成（段落優先）

    helper_rag_qa.pyのSemanticCoverage.create_semantic_chunks()を使用し、
    段落境界を最優先したセマンティック分割を実行。
    文脈を保持しながら適切なサイズでチャンクを作成。

    Args:
        text: 分割対象テキスト
        lang: 言語（"ja" or "en"）※現在は自動判定
        max_tokens: チャンクの最大トークン数
        chunk_id_prefix: チャンクIDのプレフィックス

    Returns:
        チャンクのリスト
    """
    # SemanticCoverageを使用してセマンティック分割を実行
    semantic_analyzer = SemanticCoverage(embedding_model="gemini-embedding-001")

    # 段落優先のセマンティック分割を実行
    # prefer_paragraphs=True: 段落境界を最優先
    # max_tokens: チャンクの最大トークン数
    # min_tokens: 最小トークン数（小さすぎるチャンクは自動マージ）
    # verbose=False: 詳細ログを抑制
    semantic_chunks = semantic_analyzer.create_semantic_chunks(
        document=text,
        max_tokens=max_tokens,
        min_tokens=50,  # 最小トークン数
        prefer_paragraphs=True,  # 段落優先モード
        verbose=False
    )

    # SemanticCoverageの出力形式をa02の形式に変換
    chunks = []
    tokenizer = tiktoken.get_encoding("cl100k_base")

    for i, semantic_chunk in enumerate(semantic_chunks):
        chunk_text = semantic_chunk['text']
        chunk_tokens = len(tokenizer.encode(chunk_text))

        chunks.append({
            'id': f"{chunk_id_prefix}_{i}",
            'text': chunk_text,
            'tokens': chunk_tokens,
            'type': semantic_chunk.get('type', 'unknown'),  # paragraph/sentence_group/forced_split
            'sentences': semantic_chunk.get('sentences', [])
        })

    return chunks


def create_document_chunks(df: pd.DataFrame, dataset_type: str, max_docs: Optional[int] = None, config: Optional[Dict] = None) -> List[Dict]:
    """DataFrameから文書チャンクを作成（セマンティック分割）
    Args:
        df: データフレーム
        dataset_type: データセットタイプ
        max_docs: 処理する最大文書数
        config: データセット設定（指定がない場合はDATASET_CONFIGSから取得）
    Returns:
        チャンクのリスト
    """
    if config is None:
        config = DATASET_CONFIGS.get(dataset_type)
        if not config:
            raise ValueError(f"未対応のデータセット: {dataset_type}")

    text_col = config["text_column"]
    title_col = config.get("title_column")
    chunk_size = config["chunk_size"]
    lang = config["lang"]

    all_chunks = []

    # 処理する文書数を制限
    docs_to_process = df.head(max_docs) if max_docs else df

    logger.info(f"チャンク作成開始: {len(docs_to_process)}件の文書（セマンティック分割）")

    total_docs = len(docs_to_process)
    for i, (idx, row) in enumerate(docs_to_process.iterrows()):
        # 進捗ログ（10件ごと）
        if (i + 1) % 10 == 0 or (i + 1) == total_docs:
            logger.info(f"  チャンク作成進捗: {i + 1}/{total_docs} 文書完了")

        # row[text_col]はSeriesやオブジェクトの可能性があるため、明示的にstrに変換
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""

        # タイトルがある場合は含める
        if title_col and title_col in row and pd.notna(row[title_col]):
            doc_id = f"{dataset_type}_{idx}_{str(row[title_col])[:30]}"
        else:
            doc_id = f"{dataset_type}_{idx}"

        # セマンティック分割によるチャンク作成を使用
        try:
            chunk_id_prefix = f"{doc_id}_chunk"
            chunks = create_semantic_chunks(
                text=text,
                lang=lang,
                max_tokens=chunk_size,
                chunk_id_prefix=chunk_id_prefix
            )

            # 各チャンクにメタデータを追加
            for i, chunk in enumerate(chunks):
                chunk['doc_id'] = doc_id
                chunk['doc_idx'] = idx
                chunk['chunk_idx'] = i
                chunk['dataset_type'] = dataset_type
                all_chunks.append(chunk)

        except Exception as e:
            logger.warning(f"チャンク作成エラー (doc {idx}): {e}")
            continue

    logger.info(f"チャンク作成完了: {len(all_chunks)}個のチャンク（セマンティック分割）")
    return all_chunks


def merge_small_chunks(chunks: List[Dict], min_tokens: int = 150, max_tokens: int = 400) -> List[Dict]:
    """小さいチャンクを統合して適切なサイズにする
    Args:
        chunks: チャンクのリスト
        min_tokens: このトークン数未満のチャンクは統合対象
        max_tokens: 統合後の最大トークン数
    Returns:
        統合されたチャンクのリスト
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    merged_chunks = []
    current_merge = None

    for chunk in chunks:
        chunk_tokens = len(tokenizer.encode(chunk['text']))

        # 大きいチャンクはそのまま追加
        if chunk_tokens >= min_tokens:
            if current_merge:
                merged_chunks.append(current_merge)
                current_merge = None
            merged_chunks.append(chunk)
        else:
            # 小さいチャンクは統合候補
            if current_merge is None:
                current_merge = chunk.copy()
                current_merge['merged'] = True
                current_merge['original_chunks'] = [chunk['id']]
            else:
                # 統合可能かチェック
                merge_tokens = len(tokenizer.encode(current_merge['text']))
                if merge_tokens + chunk_tokens <= max_tokens:
                    # 同じ文書からのチャンクのみ統合
                    if current_merge.get('doc_id') == chunk.get('doc_id'):
                        current_merge['text'] += "\n\n" + chunk['text']
                        current_merge['original_chunks'].append(chunk['id'])
                        if 'chunk_idx' in current_merge:
                            current_merge['chunk_idx'] = f"{current_merge['chunk_idx']}-{chunk['chunk_idx']}"
                    else:
                        # 異なる文書の場合は別々に
                        merged_chunks.append(current_merge)
                        current_merge = chunk.copy()
                        current_merge['merged'] = True
                        current_merge['original_chunks'] = [chunk['id']]
                else:
                    # サイズオーバーの場合は現在の統合を追加して新規開始
                    merged_chunks.append(current_merge)
                    current_merge = chunk.copy()
                    current_merge['merged'] = True
                    current_merge['original_chunks'] = [chunk['id']]

    # 最後の統合チャンクを追加
    if current_merge:
        merged_chunks.append(current_merge)

    logger.info(f"チャンク統合: {len(chunks)}個 → {len(merged_chunks)}個 ({100*(1-len(merged_chunks)/len(chunks)):.1f}%削減)")
    return merged_chunks
