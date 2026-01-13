# csv_text_to_chunks_text_csv.py
"""
主要機能:
- chunks_all_async(): テキストからチャンクを作成（LLMベース、asyncio並列処理）
- load_text_from_csv(): CSVファイルからテキストを読み込み（✅ v1.2.0追加）
- save_chunks_as_csv(): チャンクをCSV形式で保存（✅ v1.2.0追加）
- save_chunks_as_text(): チャンクをテキスト形式で保存（✅ v1.2.0追加）

テキストまたはCSVファイルを意味的なチャンクに分割するパイプライン。非同期・並列処理により高速化。

Usage:
python -m chunking.csv_to_chunks_text_para -i ./OUTPUT/wikipedia_ja_20251130_041304.txt -o ./OUTPUT/wikipedia_ja_chunked.txt -w 10

python -m chunking.csv_text_to_chunks_text_csv -i ./OUTPUT/cc_news_5per.csv -o ./OUTPUT/cc_news_5per_chunked.csv -w 8 -b 1500 -m gemini-2.0-flash
python -m chunking.csv_text_to_chunks_text_csv -i ./OUTPUT/fineweb_edu_ja_5per.csv -o ./OUTPUT/fineweb_edu_ja_5per_chunked.csv -w 8 -b 1500 -m gemini-2.0-flash
python -m chunking.csv_text_to_chunks_text_csv -i ./OUTPUT/japanese_text_5per.csv -o ./OUTPUT/japanese_text_5per_chunked.csv -w 8 -b 1500 -m gemini-2.0-flash
python -m chunking.csv_text_to_chunks_text_csv -i ./OUTPUT/livedoor_5per.csv -o ./OUTPUT/livedoor_5per_chunked.csv -w 8 -b 1500 -m gemini-2.0-flash
python -m chunking.csv_text_to_chunks_text_csv -i ./OUTPUT/wikipedia_ja_5per.csv -o ./OUTPUT/wikipedia_ja_5per_chunked.csv -w 8 -b 1500 -m gemini-2.0-flash

"""
# csv_text_to_chunks_text_csv.py - 改行削除版
"""
CSV出力時に改行を削除してクリーンなCSVを作成
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import tiktoken
import re
from tqdm.asyncio import tqdm as async_tqdm

# 既存のインポート
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
# ✅ 新規追加: テキスト正規化関数
# ================================================================

def _normalize_whitespace(text: str) -> str:
    """
    テキストの改行・空白を正規化

    - 改行(\n)を半角スペースに置換
    - 連続する空白を1つに正規化
    - 先頭・末尾の空白を削除

    Args:
        text: 正規化対象テキスト

    Returns:
        正規化されたテキスト

    Examples:
        >>> _normalize_whitespace("行1\\n\\n行2")
        '行1 行2'
        >>> _normalize_whitespace("  複数    空白  ")
        '複数 空白'
    """
    # 改行を半角スペースに置換
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')

    # タブを半角スペースに置換
    text = text.replace('\t', ' ')

    # 連続する空白を1つに正規化
    text = re.sub(r'\s+', ' ', text)

    # 先頭・末尾の空白を削除
    text = text.strip()

    return text


# ================================================================
# CSV読み込み機能
# ================================================================

def load_text_from_csv(
        csv_path: str,
        text_column: Optional[str] = None,
        max_rows: Optional[int] = None,
        combine_rows: bool = False
) -> str:
    """CSVファイルからテキストを読み込む"""
    logger.info("=" * 60)
    logger.info("CSV読み込み処理")
    logger.info("=" * 60)

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"  📁 読み込み: {len(df)} 行")
    except Exception as e:
        logger.error(f"CSV読み込みエラー: {e}")
        raise

    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        logger.info(f"  ✂️  制限: {len(df)} 行に制限")

    if text_column:
        if text_column not in df.columns:
            raise ValueError(
                f"指定されたカラム '{text_column}' が見つかりません。\n"
                f"利用可能なカラム: {list(df.columns)}"
            )
        col = text_column
    else:
        text_candidates = [
            'text', 'Text', 'TEXT',
            'content', 'Content', 'CONTENT',
            'Combined_Text', 'combined_text',
            'body', 'Body', 'BODY',
            'document', 'Document',
            'answer', 'Answer'
        ]

        col = None
        for candidate in text_candidates:
            if candidate in df.columns:
                col = candidate
                break

        if col is None:
            col = df.columns[0]
            logger.warning(
                f"テキストカラムを自動検出できませんでした。\n"
                f"  最初のカラム '{col}' を使用します。"
            )

    logger.info(f"  📝 テキストカラム: '{col}'")

    texts = df[col].fillna('').astype(str).tolist()
    texts = [t.strip() for t in texts if t.strip()]

    logger.info(f"  ✅ 抽出: {len(texts)} 件の非空テキスト")

    if combine_rows:
        combined_text = "\n\n".join(texts)
        logger.info(f"  🔗 結合モード: 全 {len(texts)} 行を1つのテキストに結合")
    else:
        combined_text = "\n\n".join(texts)
        logger.info(f"  📄 個別モード: {len(texts)} 個のテキストを改行区切りで処理")

    logger.info(f"  📊 総サイズ: {format_size(len(combined_text))}")
    return combined_text


# ================================================================
# ✅ 改修: CSV保存機能（改行削除対応）
# ================================================================

def save_chunks_as_csv(
        chunks: List[str],
        output_file: str,
        dataset_type: str = "custom",
        source_file: Optional[str] = None,
        normalize_whitespace: bool = True  # ✅ 新規パラメータ
) -> str:
    """
    チャンクをCSV形式で保存（メタデータ付き）

    Args:
        chunks: チャンクのリスト
        output_file: 出力ファイルパス
        dataset_type: データセット種別
        source_file: 元ファイル名
        normalize_whitespace: 改行・空白を正規化するか（デフォルト: True）

    Returns:
        保存したCSVファイルパス
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")

    data = []
    for i, chunk_text in enumerate(chunks):
        # ✅ 改行・空白を正規化（CSV出力をクリーンにする）
        if normalize_whitespace:
            chunk_text_cleaned = _normalize_whitespace(chunk_text)
        else:
            chunk_text_cleaned = chunk_text

        # センテンス分割（正規化前のテキストで実施）
        sentences = _split_sentences_simple(chunk_text)

        data.append({
            'chunk_id'      : f"{dataset_type}_chunk_{i}",
            'text'          : chunk_text_cleaned,  # ✅ 正規化されたテキスト
            'tokens'        : len(tokenizer.encode(chunk_text_cleaned)),
            'chunk_idx'     : i,
            'dataset_type'  : dataset_type,
            'type'          : 'llm_chunk',
            'sentence_count': len(sentences),
            'source_file'   : source_file or ''
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8')

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ CSV保存完了")
    logger.info("=" * 60)
    logger.info(f"  ファイル: {output_file}")
    logger.info(f"  チャンク数: {len(df)}")
    logger.info(f"  総トークン数: {df['tokens'].sum()}")
    logger.info(f"  平均トークン数: {df['tokens'].mean():.1f}")
    logger.info(f"  改行正規化: {'有効' if normalize_whitespace else '無効'}")
    logger.info("=" * 60)

    return output_file


def save_chunks_as_text(chunks: List[str], output_file: str) -> str:
    """テキスト形式で保存（既存形式・後方互換性）"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n---\n')

    logger.info(f"テキストファイル保存: {output_file} ({len(chunks)}チャンク)")
    return output_file


def _split_sentences_simple(text: str) -> List[str]:
    """簡易的な文分割（日本語対応）"""
    sentences = re.findall(r'[^。．.！？!?]+[。．.！？!?]\s*', text)

    if not sentences:
        sentences = [text.strip()] if text.strip() else []
    else:
        last_pos = text.rfind(sentences[-1]) + len(sentences[-1])
        if last_pos < len(text):
            remaining = text[last_pos:].strip()
            if remaining:
                sentences.append(remaining)

    return [s.strip() for s in sentences if s.strip()]


# ================================================================
# chunks_all_async関数（変更なし - 既存コードと同じ）
# ================================================================

async def chunks_all_async(
        text: str,
        model: str = "gemini-2.0-flash-exp",
        max_workers: int = 8,
        block_size: int = 2000,
        checkpoint_manager: Optional[CheckpointManager] = None,
        output_file: Optional[str] = None,
        dataset_type: str = "custom",
        source_file: Optional[str] = None
) -> List[str]:
    """テキストを3段階で意味的にチャンク化"""
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

    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager()

    logger.info("=" * 60)
    logger.info("チャンク化処理開始 (3段階)")
    logger.info("=" * 60)
    logger.info(f"入力テキスト: {format_size(len(text))}")
    logger.info(f"モデル: {model}")
    logger.info(f"並列ワーカー数: {max_workers}")

    step1_chunks = await _step1_hierarchical_split(
        text, client, model, block_size, checkpoint_manager
    )

    step2_chunks = await _step2_semantic_chunking(
        step1_chunks, client, model, checkpoint_manager
    )

    final_chunks = await _step3_continuity_check(
        step2_chunks, client, model, checkpoint_manager
    )

    if output_file:
        output_path = Path(output_file)

        if output_path.suffix.lower() == '.csv':
            save_chunks_as_csv(
                chunks=final_chunks,
                output_file=output_file,
                dataset_type=dataset_type,
                source_file=source_file,
                normalize_whitespace=True  # ✅ 改行正規化を有効化
            )
        else:
            save_chunks_as_text(
                chunks=final_chunks,
                output_file=output_file
            )

    return final_chunks


async def _step1_hierarchical_split(
        text: str,
        client: AsyncAPIClient,
        model: str,
        block_size: int,
        checkpoint_manager: CheckpointManager
) -> List[str]:
    """Step 1: 階層構造化"""
    if checkpoint_manager.exists("step1"):
        logger.info("Step1: チェックポイントから再開")
        return checkpoint_manager.load("step1")

    logger.info("\n[Step 1/3] 階層構造化（段落 > 文）")

    blocks = [text[i:i + block_size] for i in range(0, len(text), block_size)]
    logger.info(f"  ブロック数: {len(blocks)}")

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

    # results = await asyncio.gather(*tasks)
    results = await async_tqdm.gather(
        *tasks,
        desc="Step1: 階層構造化",  # 各ステップで説明を変更
        total=len(tasks)
    )

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
    checkpoint_manager.save("step1", paragraphs)

    return paragraphs


async def _step2_semantic_chunking(
        paragraphs: List[str],
        client: AsyncAPIClient,
        model: str,
        checkpoint_manager: CheckpointManager
) -> List[str]:
    """Step 2: 意味的分割"""
    if checkpoint_manager.exists("step2"):
        logger.info("Step2: チェックポイントから再開")
        return checkpoint_manager.load("step2")

    logger.info("\n[Step 2/3] 意味的分割")
    logger.info(f"  入力: {len(paragraphs)} 段落")

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

    # results = await asyncio.gather(*tasks)
    results = await async_tqdm.gather(
        *tasks,
        desc="Step2: 意味的分割",  # 各ステップで説明を変更
        total=len(tasks)
    )

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
    checkpoint_manager.save("step2", chunks)

    return chunks


async def _step3_continuity_check(
        chunks: List[str],
        client: AsyncAPIClient,
        model: str,
        checkpoint_manager: CheckpointManager
) -> List[str]:
    """Step 3: 文脈連続性チェック"""
    if checkpoint_manager.exists("step3"):
        logger.info("Step3: チェックポイントから再開")
        return checkpoint_manager.load("step3")

    logger.info("\n[Step 3/3] 文脈連続性チェック")
    logger.info(f"  入力: {len(chunks)} チャンク")

    if len(chunks) <= 1:
        checkpoint_manager.save("step3", chunks)
        return chunks

    tasks = []
    for i in range(len(chunks) - 1):
        prompt = f"{CONTINUITY_CHECK_PROMPT}\n\n【前のテキスト】\n{chunks[i]}\n\n【次のテキスト】\n{chunks[i + 1]}"
        task = client.generate_content(
            model=model,
            contents=prompt,
            response_schema=ContinuityResult,
            task_id=f"step3_pair_{i}"
        )
        tasks.append(task)

    # results = await asyncio.gather(*tasks)
    results = await async_tqdm.gather(
        *tasks,
        desc="Step2: 連続性チェック",  # 各ステップで説明を変更
        total=len(tasks)
    )

    final_chunks = [chunks[0]]
    for i, result_json in enumerate(results):
        if result_json:
            try:
                result = ContinuityResult.model_validate_json(result_json)
                if result.is_connected:
                    final_chunks[-1] += "\n\n" + chunks[i + 1]
                else:
                    final_chunks.append(chunks[i + 1])
            except Exception as e:
                logger.warning(f"パース失敗: {e}")
                final_chunks.append(chunks[i + 1])
        else:
            final_chunks.append(chunks[i + 1])

    logger.info(f"  出力: {len(final_chunks)} チャンク（マージ後）")
    checkpoint_manager.save("step3", final_chunks)

    return final_chunks


# ================================================================
# メイン関数
# ================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="LLMベースのセマンティックチャンキング（CSV/TXT入力対応、改行正規化対応）"
    )
    parser.add_argument("-i", "--input", required=True, help="入力ファイル (.txt または .csv)")
    parser.add_argument("-o", "--output", required=True, help="出力ファイル (.csv または .txt)")
    parser.add_argument("-m", "--model", default="gemini-2.0-flash-exp", help="モデル")
    parser.add_argument("-w", "--workers", type=int, default=8, help="並列ワーカー数")
    parser.add_argument("-b", "--block-size", type=int, default=2000, help="バッチサイズ")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細ログ")
    parser.add_argument("--resume", type=str, default=None, help="再開するジョブID")
    parser.add_argument("--text-column", type=str, default=None, help="CSVのテキストカラム名")
    parser.add_argument("--max-rows", type=int, default=None, help="最大処理行数（CSV用）")
    parser.add_argument("--combine-rows", action="store_true", help="CSV全行を結合")

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"入力ファイルが見つかりません: {args.input}")
        return

    file_extension = input_path.suffix.lower()

    if file_extension == '.csv':
        text = load_text_from_csv(
            csv_path=args.input,
            text_column=args.text_column,
            max_rows=args.max_rows,
            combine_rows=args.combine_rows
        )
    else:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()

    logger.info(f"入力ファイル: {args.input}")
    logger.info(f"テキストサイズ: {format_size(len(text))}")

    checkpoint_manager = CheckpointManager(job_id=args.resume) if args.resume else CheckpointManager()
    dataset_type = input_path.stem

    final_chunks = await chunks_all_async(
        text=text,
        model=args.model,
        max_workers=args.workers,
        block_size=args.block_size,
        checkpoint_manager=checkpoint_manager,
        output_file=args.output,
        dataset_type=dataset_type,
        source_file=input_path.name
    )

    logger.info(f"\n✅ 処理完了: {len(final_chunks)} チャンク")


if __name__ == "__main__":
    asyncio.run(main())


