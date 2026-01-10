# csv_to_chunks_text_para.py
"""
（1）「チャンク」テキストチャンキング処理（並列版） - output.csv
    　csv_to_chunks_text_para.py
（2）「Q/Aペア」& Qdrant登録
    　make_qa_register_qdrant.py　
    （make_qa.py + register_csv_to_qdrant.py）

テキストまたはCSVファイルを意味的なチャンクに分割するパイプライン。
非同期・並列処理により高速化。

Usage:
    # テキストファイルの処理
    python -m chunking.csv_to_chunks_text_para -i input.txt -o output.txt

    # CSVファイルの処理（text または Combined_Text カラムを使用）
    python -m chunking.csv_to_chunks_text_para -i input.csv -o output.txt

    # 並列数の指定
    python -m chunking.csv_to_chunks_text_para -i input.txt -o output.txt -w 16

    # 中断からの再開
    python -m chunking.csv_to_chunks_text_para --resume JOB_ID -i input.txt -o output.txt

python -m chunking.csv_to_chunks_text_para -i ./OUTPUT/wikipedia_ja_20251130_041304.txt -o ./OUTPUT/wikipedia_ja_chunked.txt -w 10

python -m chunking.csv_to_chunks_text_para -i ./OUTPUT/cc_news_5per.csv -o ./OUTPUT/cc_news_5per_chunked.csv -w 10
python -m chunking.csv_to_chunks_text_para -i ./OUTPUT/fineweb_edu_ja_5per.csv -o ./OUTPUT/fineweb_edu_ja_5per_chunked.csv -w 10
python -m chunking.csv_to_chunks_text_para -i ./OUTPUT/japanese_text_5per.csv -o ./OUTPUT/japanese_text_5per_chunked.csv -w 10
python -m chunking.csv_to_chunks_text_para -i ./OUTPUT/livedoor_5per.csv -o ./OUTPUT/livedoor_5per_chunked.csv -w 10
python -m chunking.csv_to_chunks_text_para -i ./OUTPUT/wikipedia_ja_5per.csv -o ./OUTPUT/wikipedia_ja_5per_chunked.csv -w 10

"""

import argparse
import asyncio
import os
import re
import time
from datetime import datetime
from typing import List, Optional
import logging
import pandas as pd

from tqdm.asyncio import tqdm_asyncio

from .async_api_client import AsyncAPIClient
from .checkpoint_manager import CheckpointManager
from .models import StructuralResult, ContinuityResult
from .prompts import (
    PARAGRAPH_SEPARATION_PROMPT,
    SEMANTIC_CHUNKING_PROMPT,
    CONTINUITY_CHECK_PROMPT
)
from .utils import setup_logging, show_paragraphs, format_time, print_stats

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> List[str]:
    """
    言語に依存しない文分割（英語・日本語両対応）
    Args:
        text: 分割対象のテキスト
    Returns:
        文のリスト
    """
    if not text:
        return []

    # 日本語の句点: 。．！？
    # 英語の句点: .!? (後ろに空白+大文字が続く場合、または文末)
    # 省略語（Dr., Mr., U.S.等）を考慮して、空白+大文字の後続を条件に追加
    pattern = r'(?<=[。．！？])|(?<=[.!?])(?=\s+[A-Z])|(?<=[.!?])(?=\s*$)'

    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 分割できなかった場合は元のテキストを返す
    if not sentences:
        return [text]

    return sentences


class LargeTextProcessorPara:
    """
    並列処理対応の大規模テキストプロセッサ
    Features:
        - 非同期・並列API呼び出し
        - 動的並列数調整
        - チェックポイント機能
        - プログレスバー表示
    """

    def __init__(
            self,
            block_size: int = 2000,
            max_workers: int = 8,
            max_retries: int = 3,
            max_output_tokens: int = 4096,
            checkpoint_dir: str = "./checkpoints",
            resume_job_id: Optional[str] = None
    ):
        """
        Args:
            block_size: バッチサイズ（デフォルト: 2000文字）
            max_workers: 並列数（デフォルト: 8）
            max_retries: リトライ回数（デフォルト: 3）
            max_output_tokens: 出力トークン制限（デフォルト: 4096）
            checkpoint_dir: チェックポイント保存ディレクトリ
            resume_job_id: 再開するジョブID（指定時は途中から再開）
        """
        self.block_size = block_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.api_client: Optional[AsyncAPIClient] = None
        self.checkpoint = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            job_id=resume_job_id
        )

    """ 前処理 """

    def split_into_batches(self, text: str) -> List[str]:
        """
        テキストを block_size 以下に分割

        Args:
            text: 入力テキスト

        Returns:
            バッチのリスト
        """
        # 改行コードを統一
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        batches = []
        current_batch = []
        current_length = 0

        raw_lines = text.split('\n')

        for raw_line in raw_lines:
            # 行自体が block_size を超えている場合
            if len(raw_line) > self.block_size:
                if current_batch:
                    batches.append("\n".join(current_batch))
                    current_batch = []
                    current_length = 0

                # 長い行を step: block_size ずつスライス
                for i in range(0, len(raw_line), self.block_size):
                    chunk = raw_line[i: i + self.block_size]
                    batches.append(chunk)
                continue

            # 通常の積み上げ処理
            line_len = len(raw_line) + 1  # 改行分

            if current_length + line_len > self.block_size:
                if current_batch:
                    batches.append("\n".join(current_batch))
                current_batch = [raw_line]
                current_length = line_len
            else:
                current_batch.append(raw_line)
                current_length += line_len

        if current_batch:
            batches.append("\n".join(current_batch))

        return batches

    async def process(
            self,
            text: str,
            model: str = "gemini-2.0-flash",
            api_key: Optional[str] = None
    ) -> List[str]:
        """
        メイン処理（非同期版）

        Args:
            text: 処理対象テキスト
            model: Geminiモデル名
            api_key: APIキー（省略時は環境変数から取得）

        Returns:
            チャンク化されたテキストのリスト
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        self.api_client = AsyncAPIClient(
            api_key=api_key,
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            max_output_tokens=self.max_output_tokens
        )

        start_time = datetime.now()
        logger.info(f"Processing started at {start_time}")
        logger.info(f"Text length: {len(text)} chars, Block size: {self.block_size}")
        logger.info(f"Max workers: {self.max_workers}")

        # チェックポイントから再開確認
        resume_step, resume_data = self.checkpoint.get_resume_point()
        if resume_data:
            logger.info(f"Resuming from checkpoint. Next step: {resume_step}")

        # Step1: 階層分割　ここ
        if resume_step == "step1" or resume_step is None:
            batches = self.split_into_batches(text)
            logger.info(f"Total batches: {len(batches)}")

            print("\n=== Step1: Hierarchical Splitting (Parallel) ===")
            step1_results = await self._step1_parallel(batches, model)
            self.checkpoint.save("step1", step1_results, {"batch_count": len(batches)})
        else:
            step1_results = self.checkpoint.load("step1") or []
            logger.info(f"Step1 loaded from checkpoint: {len(step1_results)} paragraphs")

        # Step2: 意味分割
        if resume_step in ["step1", "step2"] or resume_step is None:
            print("\n=== Step2: Semantic Chunking (Parallel) ===")
            step2_results = await self._step2_parallel(step1_results, model)
            self.checkpoint.save("step2", step2_results)
        else:
            step2_results = self.checkpoint.load("step2") or []
            logger.info(f"Step2 loaded from checkpoint: {len(step2_results)} chunks")

        # Step3: オーバーラップ付与
        if resume_step in ["step1", "step2", "step3"] or resume_step is None:
            print("\n=== Step3: Smart Overlap (Parallel) ===")
            step3_results = await self._step3_parallel(step2_results, model)
            self.checkpoint.save("step3", step3_results)
        else:
            step3_results = self.checkpoint.load("step3") or []
            logger.info(f"Step3 loaded from checkpoint: {len(step3_results)} final chunks")

        # 統計情報
        elapsed = (datetime.now() - start_time).total_seconds()
        stats = self.api_client.get_stats()

        print(f"\n=== Processing Complete ===")
        print(f"Total time: {format_time(elapsed)}")
        print(f"Total chunks: {len(step3_results)}")
        print_stats(stats, "API Statistics")

        return step3_results

    async def _step1_parallel(self, batches: List[str], model: str) -> List[str]:
        """Step1: 各バッチを並列処理"""

        async def process_one(idx: int, batch_text: str) -> tuple[int, List[str]]:
            """1つのバッチを処理"""
            if not batch_text.strip():
                return (idx, [])

            if len(batch_text) > 10000:
                logger.warning(f"Batch {idx} too large ({len(batch_text)} chars), skipping")
                return (idx, [batch_text])

            response = await self.api_client.generate_content(
                model=model,
                contents=f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【対象テキスト】\n{batch_text}",
                response_schema=StructuralResult,
                task_id=f"step1-{idx}"
            )

            if response:
                try:
                    result = StructuralResult.model_validate_json(response)
                    paragraphs = [p.full_text.replace('\n', ' ') for p in result.paragraphs]
                    return (idx, paragraphs)
                except Exception as e:
                    logger.error(f"Step1 parse error at batch {idx}: {e}")

            # フォールバック: 元テキストを保持
            logger.warning(f"Batch {idx}: Using original text as fallback")
            return (idx, [batch_text])

        # タスク生成
        tasks = [process_one(i, batch) for i, batch in enumerate(batches)]

        # 並列実行（tqdmでプログレス表示）
        results_map = {}
        for coro in tqdm_asyncio.as_completed(tasks, desc="Step1", total=len(tasks)):
            idx, result = await coro
            results_map[idx] = result

        # 順序を維持してフラット化
        paragraphs = []
        for i in range(len(batches)):
            paragraphs.extend(results_map.get(i, []))

        logger.info(f"Step1 complete: {len(paragraphs)} paragraphs")
        return paragraphs

    async def _step2_parallel(self, paragraphs: List[str], model: str) -> List[str]:
        """Step2: 各パラグラフを並列処理"""

        async def process_one(idx: int, para_text: str) -> tuple[int, List[str]]:
            """1つのパラグラフを処理"""
            if not para_text.strip():
                return (idx, [])

            response = await self.api_client.generate_content(
                model=model,
                contents=f"{SEMANTIC_CHUNKING_PROMPT}\n\n【対象テキスト】\n{para_text}",
                response_schema=StructuralResult,
                task_id=f"step2-{idx}"
            )

            if response:
                try:
                    result = StructuralResult.model_validate_json(response)
                    chunks = []
                    for p in result.paragraphs:
                        clean_text = p.full_text.replace('\n', ' ')
                        if not clean_text.endswith('\n'):
                            clean_text += '\n'
                        chunks.append(clean_text)
                    return (idx, chunks)
                except Exception as e:
                    logger.error(f"Step2 parse error at para {idx}: {e}")

            # フォールバック: 元テキストを保持
            logger.warning(f"Paragraph {idx}: Using original text as fallback")
            fallback_text = para_text if para_text.endswith('\n') else para_text + '\n'
            return (idx, [fallback_text])

        # タスク生成
        tasks = [process_one(i, para) for i, para in enumerate(paragraphs)]

        # 並列実行
        results_map = {}
        for coro in tqdm_asyncio.as_completed(tasks, desc="Step2", total=len(tasks)):
            idx, result = await coro
            results_map[idx] = result

        # 順序を維持してフラット化
        chunks = []
        for i in range(len(paragraphs)):
            chunks.extend(results_map.get(i, []))

        logger.info(f"Step2 complete: {len(chunks)} chunks")
        return chunks

    async def _step3_parallel(self, chunks: List[str], model: str) -> List[str]:
        """Step3: 連続性判定を並列実行"""
        return await chunk_overlap_para(chunks, self.api_client, model)


async def chunk_overlap_para(
        paragraphs: List[str],
        api_client: AsyncAPIClient,
        model: str = "gemini-2.0-flash"
) -> List[str]:
    """
    並列版オーバーラップ処理

    連続性判定(check_continuity)は独立して並列実行可能。
    結果の適用は元の順序を維持する。
    Args:
        paragraphs: チャンクのリスト
        api_client: AsyncAPIClientインスタンス
        model: Geminiモデル名
    Returns:
        オーバーラップ処理後のチャンクリスト
    """
    if not paragraphs:
        return []

    if len(paragraphs) == 1:
        return paragraphs

    async def check_one(idx: int, prev_text: str, next_text: str) -> tuple[int, bool]:
        """1ペアの連続性を判定"""
        response = await api_client.generate_content(
            model=model,
            contents=(
                f"{CONTINUITY_CHECK_PROMPT}\n\n"
                f"【前のテキスト】\n{prev_text}\n\n"
                f"【次のテキスト】\n{next_text}"
            ),
            response_schema=ContinuityResult,
            task_id=f"step3-{idx}"
        )

        if response:
            try:
                result = ContinuityResult.model_validate_json(response)
                return (idx, result.is_connected)
            except Exception as e:
                logger.error(f"Step3 parse error at pair {idx}: {e}")

        # エラー時は安全側（分割）
        return (idx, False)

    # 全ペアのタスクを生成
    tasks = []
    for i in range(1, len(paragraphs)):
        prev_text = paragraphs[i - 1]
        current_text = paragraphs[i]
        tasks.append(check_one(i, prev_text, current_text))

    logger.info(f"Checking continuity for {len(tasks)} pairs...")

    # 並列実行（プログレス表示付き）
    continuity_map = {}
    for coro in tqdm_asyncio.as_completed(tasks, desc="Step3", total=len(tasks)):
        idx, is_connected = await coro
        continuity_map[idx] = is_connected

    # オーバーラップ適用（順序通りに処理）
    overlapped_result = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        current_text = paragraphs[i]
        is_connected = continuity_map.get(i, False)

        if not is_connected:
            overlapped_result.append(current_text)
            continue

        # オーバーラップ処理: 前のチャンクの最後の1文を追加
        prev_text = paragraphs[i - 1]
        sentences = split_sentences(prev_text)  # ← 修正箇所: 新しい文分割関数を使用

        overlap_part = sentences[-1] if sentences else prev_text
        combined_text = overlap_part + current_text
        overlapped_result.append(combined_text)

    logger.info(f"Step3 complete: {len(overlapped_result)} final chunks")
    return overlapped_result


# === 一気通貫処理関数 (Wrapper) ===

async def chunks_all_async(
        text: str,
        model: str = "gemini-2.0-flash",
        max_workers: int = 8,
        block_size: int = 2000
) -> List[str]:
    """
    テキスト処理パイプラインを一気通貫で実行（非同期版）

    Args:
        text: 入力テキスト
        model: Geminiモデル名
        max_workers: 並列数
        block_size: バッチサイズ

    Returns:
        チャンク化されたテキストのリスト
    """
    processor = LargeTextProcessorPara(
        block_size=block_size,
        max_workers=max_workers
    )
    return await processor.process(text, model)


def chunks_all(
        text: str,
        model: str = "gemini-2.0-flash",
        max_workers: int = 8,
        block_size: int = 2000
) -> List[str]:
    """
    テキスト処理パイプラインを一気通貫で実行（同期版ラッパー）
    Args:
        text: 入力テキスト
        model: Geminiモデル名
        max_workers: 並列数
        block_size: バッチサイズ
    Returns:
        チャンク化されたテキストのリスト
    """
    return asyncio.run(chunks_all_async(text, model, max_workers, block_size))


# === コマンドライン処理 ===

def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="テキストチャンキング処理（並列版） - CSVとテキストファイルに対応",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # テキストファイルの処理
  python -m chunking.csv_to_chunks_text_para -i input.txt -o output.txt

  # CSVファイルの処理（text または Combined_Text カラムを自動検出）
  python -m chunking.csv_to_chunks_text_para -i input.csv -o output.txt

  # 並列数を16に指定
  python -m chunking.csv_to_chunks_text_para -i input.txt -o output.txt -w 16

  # 中断したジョブを再開
  python -m chunking.csv_to_chunks_text_para -i input.txt -o output.txt --resume 20250108_143022

  # 保存済みジョブの一覧表示
  python -m chunking.csv_to_chunks_text_para --list-jobs

CSVファイルの場合:
  - 'Combined_Text' カラムがあればそれを優先的に使用
  - なければ 'text' カラムを使用
  - どちらもなければ最初のカラムを使用
        """
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        help="入力ファイルパス（.txt または .csv）"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="出力ファイルパス（省略時は標準出力）"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=8,
        help="並列処理数（デフォルト: 8）"
    )
    parser.add_argument(
        "-b", "--block-size",
        type=int,
        default=2000,
        help="バッチサイズ（デフォルト: 2000文字）"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gemini-2.0-flash",
        help="使用するGeminiモデル（デフォルト: gemini-2.0-flash）"
    )
    parser.add_argument(
        "-r", "--max-retries",
        type=int,
        default=3,
        help="リトライ回数（デフォルト: 3）"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="出力トークン制限（デフォルト: 4096）"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="チェックポイント保存ディレクトリ（デフォルト: ./checkpoints）"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="JOB_ID",
        help="指定したジョブIDから処理を再開"
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="保存済みジョブの一覧を表示"
    )
    parser.add_argument(
        "--cleanup-jobs",
        type=int,
        default=None,
        metavar="KEEP_COUNT",
        help="古いジョブを削除し、指定した数だけ保持"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="詳細ログを出力"
    )

    return parser.parse_args()


async def async_main():
    """非同期メイン関数"""
    args = parse_args()

    # ジョブ一覧表示
    if args.list_jobs:
        jobs = CheckpointManager.list_jobs(args.checkpoint_dir)
        if jobs:
            print("Saved jobs:")
            for job in jobs:
                job_id = job["job_id"]
                steps = job.get("steps", {})
                latest_step = max(steps.keys()) if steps else "none"
                print(f"  - {job_id} (latest: {latest_step})")
        else:
            print("No saved jobs found.")
        return

    # 古いジョブの削除
    if args.cleanup_jobs is not None:
        CheckpointManager.cleanup_old_jobs(args.checkpoint_dir, args.cleanup_jobs)
        print(f"Cleanup complete. Keeping {args.cleanup_jobs} most recent jobs.")
        return

    # 入力ファイルチェック
    if not args.input:
        print("Error: --input is required")
        return

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # ロギング設定
    setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("Text Chunking Processor (Parallel Version)")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output or 'stdout'}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Block size: {args.block_size}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max retries: {args.max_retries}")
    logger.info(f"Max output tokens: {args.max_output_tokens}")
    logger.info(f"Resume job: {args.resume or 'None'}")

    # ファイル読み込み（CSVまたはテキスト）
    file_extension = os.path.splitext(args.input)[1].lower()

    if file_extension == '.csv':
        # CSVファイルの場合
        logger.info("Input format: CSV")
        try:
            df = pd.read_csv(args.input, encoding='utf-8')
            logger.info(f"CSV columns: {list(df.columns)}")

            # テキストカラムを探す（優先順位: Combined_Text > text > 最初のカラム）
            if 'Combined_Text' in df.columns:
                text_column = 'Combined_Text'
            elif 'text' in df.columns:
                text_column = 'text'
            else:
                text_column = df.columns[0]
                logger.warning(f"Using first column '{text_column}' as text source")

            logger.info(f"Using column: {text_column}")

            # テキストを結合（各行を改行で区切る）
            text = '\n'.join(df[text_column].astype(str).tolist())
            logger.info(f"Loaded {len(df)} rows from CSV")

        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            print(f"Error: Failed to read CSV file: {e}")
            return
    else:
        # テキストファイルの場合
        logger.info("Input format: Text")
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read text file: {e}")
            print(f"Error: Failed to read text file: {e}")
            return

    logger.info(f"Text length: {len(text)} chars")

    # プロセッサ初期化
    processor = LargeTextProcessorPara(
        block_size=args.block_size,
        max_workers=args.workers,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        checkpoint_dir=args.checkpoint_dir,
        resume_job_id=args.resume
    )

    # 処理実行
    start_time = time.time()
    results = await processor.process(text, model=args.model)
    elapsed = time.time() - start_time

    logger.info(f"Processing completed in {format_time(elapsed)}")
    logger.info(f"Total chunks: {len(results)}")

    # 出力
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for chunk in results:
                # チャンク本体を書き込み
                f.write(chunk.rstrip('\n'))
                f.write("\n")
        logger.info(f"Results saved to: {args.output}")
    else:
        print("\n=== Results (first 10 chunks) ===")
        for i, chunk in enumerate(results[:10]):
            display = chunk.replace('\n', ' ').strip()
            if len(display) > 100:
                display = display[:100] + "..."
            print(f"[{i + 1}] {display}")

    # Job ID を表示
    print(f"\nJob ID: {processor.checkpoint.job_id}")
    print(f"Checkpoint dir: {processor.checkpoint.job_dir}")


def main():
    """エントリーポイント"""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
