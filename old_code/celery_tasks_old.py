#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
celery_tasks.py - Celeryタスク定義（スマート生成対応版）

改修内容（v2.1）:
- use_smart_generationパラメータの追加
- SmartQAGeneratorとの統合
- エラーハンドリングの強化
- リトライ機能の実装
"""

import logging
from celery import Celery
from typing import List, Dict, Optional

# Celeryアプリケーション
app = Celery('qa_generation')
app.config_from_object('celery_config')

logger = logging.getLogger(__name__)


# ================================================================
# Q/A生成タスク
# ================================================================

def submit_unified_qa_generation(
        chunks: List[Dict],
        config: Dict,
        model: str,
        provider: str = "gemini",
        use_smart_generation: bool = True  # ✨ v2.1で追加
) -> List:
    """
    チャンクのQ/A生成タスクを並列実行

    Args:
        chunks: チャンクのリスト
        config: データセット設定
        model: 使用するモデル
        provider: プロバイダー名（"gemini" or "openai"）
        use_smart_generation: スマート生成を使用するか（デフォルト: True）
            - True: SmartQAGeneratorによる動的Q/A数決定（0-5個）
            - False: 従来方式（トークン数ベース、2-8個）

    Returns:
        Celeryタスクのリスト（AsyncResultオブジェクト）

    Example:
        >>> chunks = [{'id': 'chunk_0', 'text': '...'}]
        >>> tasks = submit_unified_qa_generation(
        ...     chunks, config, "gemini-2.0-flash",
        ...     use_smart_generation=True
        ... )
    """
    logger.info(f"Celeryタスクを投入: {len(chunks)}チャンク")
    logger.info(f"生成モード: {'スマート生成' if use_smart_generation else '従来方式'}")

    tasks = []
    for chunk in chunks:
        task = generate_qa_for_chunk_task.apply_async(
            args=(chunk, config, model, provider, use_smart_generation),  # ✨ 引数追加
            queue='qa_generation'
        )
        tasks.append(task)

    logger.info(f"タスク投入完了: {len(tasks)}個")
    return tasks


@app.task(
    name='generate_qa_for_chunk',
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def generate_qa_for_chunk_task(
        self,
        chunk: Dict,
        config: Dict,
        model: str,
        provider: str,
        use_smart_generation: bool = True  # ✨ v2.1で追加
) -> List[Dict]:
    """
    単一チャンクのQ/A生成タスク

    Args:
        self: Celeryタスクインスタンス（bind=True時に自動注入）
        chunk: チャンク
            例: {'id': 'chunk_0', 'text': '...', 'tokens': 250}
        config: データセット設定
            例: {'type': 'local_file', 'qa_per_chunk': 3, ...}
        model: 使用するモデル
            例: 'gemini-2.0-flash'
        provider: プロバイダー名
            例: 'gemini'
        use_smart_generation: スマート生成を使用するか

    Returns:
        Q/Aペアのリスト
        例: [{'question': '...', 'answer': '...', 'topic': '...'}]

    Raises:
        Retry: 一時的なエラー時にリトライ

    Example:
        >>> chunk = {'id': 'chunk_0', 'text': '技術文書...'}
        >>> result = generate_qa_for_chunk_task(
        ...     chunk, config, "gemini-2.0-flash", "gemini",
        ...     use_smart_generation=True
        ... )
    """
    chunk_id = chunk.get('id', 'unknown')

    try:
        logger.info(f"タスク開始: chunk={chunk_id}, smart={use_smart_generation}")

        # 遅延インポート（Celeryワーカー起動時の依存関係エラーを回避）
        from qa_generation.generation import generate_qa_dataset

        # ✨ use_smart_generationを渡す
        qa_pairs = generate_qa_dataset(
            chunks=[chunk],
            dataset_type=config.get("type", "unknown"),
            model=model,
            config=config,
            provider=provider,
            use_smart_generation=use_smart_generation  # ✨ v2.1で追加
        )

        logger.info(f"タスク完了: chunk={chunk_id}, Q/A数={len(qa_pairs)}")
        return qa_pairs

    except ImportError as exc:
        # モジュールインポートエラー（致命的）
        logger.error(f"モジュールインポートエラー: chunk={chunk_id}, error={exc}")
        return []

    except ValueError as exc:
        # データ形式エラー（致命的）
        logger.error(f"データ形式エラー: chunk={chunk_id}, error={exc}")
        return []

    except Exception as exc:
        # 一時的なエラー（リトライ可能）
        logger.error(f"タスクエラー: chunk={chunk_id}, error={exc}")

        # リトライ
        if self.request.retries < self.max_retries:
            retry_count = self.request.retries + 1
            logger.warning(f"リトライ {retry_count}/{self.max_retries}: chunk={chunk_id}")
            raise self.retry(exc=exc, countdown=60)
        else:
            logger.error(f"最大リトライ回数超過: chunk={chunk_id}, retries={self.max_retries}")
            return []


# ================================================================
# 結果収集
# ================================================================

def collect_results(tasks: List, timeout: int = 600) -> List[Dict]:
    """
    Celeryタスクの結果を収集

    Args:
        tasks: Celeryタスクのリスト（AsyncResultオブジェクト）
        timeout: タイムアウト（秒）
            - スマート生成: 600秒推奨（チャンクあたり2回のLLM呼び出し）
            - 従来方式: 300秒で十分

    Returns:
        Q/Aペアのリスト（全タスクの結果を結合）

    Example:
        >>> tasks = submit_unified_qa_generation(chunks, ...)
        >>> qa_pairs = collect_results(tasks, timeout=600)
        >>> print(f"Total Q/A pairs: {len(qa_pairs)}")
    """
    logger.info(f"結果収集中: {len(tasks)}タスク, timeout={timeout}秒")

    all_qa_pairs = []
    success_count = 0
    failed_count = 0
    timeout_count = 0

    for i, task in enumerate(tasks, 1):
        try:
            # タスクの完了を待機
            result = task.get(timeout=timeout)

            if result:
                all_qa_pairs.extend(result)
                success_count += 1
                logger.debug(f"タスク {i}/{len(tasks)}: 成功（Q/A数={len(result)}）")
            else:
                failed_count += 1
                logger.warning(f"タスク {i}/{len(tasks)}: 結果が空")

        except TimeoutError:
            # タイムアウト
            timeout_count += 1
            failed_count += 1
            logger.error(f"タスク {i}/{len(tasks)}: タイムアウト（{timeout}秒）")
            continue

        except Exception as e:
            # その他のエラー
            failed_count += 1
            logger.error(f"タスク {i}/{len(tasks)}: エラー: {e}")
            continue

    # 結果サマリー
    logger.info(f"収集完了:")
    logger.info(f"  - 成功: {success_count}/{len(tasks)}")
    logger.info(f"  - 失敗: {failed_count}/{len(tasks)}")
    logger.info(f"  - タイムアウト: {timeout_count}/{len(tasks)}")
    logger.info(f"  - Q/A総数: {len(all_qa_pairs)}")

    if failed_count > 0:
        logger.warning(f"⚠️ {failed_count}個のタスクが失敗しました")

    return all_qa_pairs


# ================================================================
# ワーカー状態確認
# ================================================================

def check_celery_workers(min_workers: int = 1) -> bool:
    """
    Celeryワーカーの状態確認

    Args:
        min_workers: 最小ワーカー数

    Returns:
        True: 必要数のワーカーが起動している
        False: ワーカーが不足または起動していない

    Example:
        >>> if check_celery_workers(min_workers=8):
        ...     print("ワーカーが起動しています")
        ... else:
        ...     print("ワーカーを起動してください")
    """
    try:
        # アクティブなワーカーを取得
        inspect = app.control.inspect()
        stats = inspect.stats()

        if stats is None:
            logger.error("❌ Celeryワーカーが応答しません")
            logger.error("以下を確認してください:")
            logger.error("  1. Redisが起動しているか: redis-cli ping")
            logger.error("  2. Celeryワーカーが起動しているか: ./start_celery.sh status")
            return False

        worker_count = len(stats)
        logger.info(f"アクティブなワーカー: {worker_count}個")

        if worker_count < min_workers:
            logger.error(f"❌ ワーカー数が不足: {worker_count} < {min_workers}")
            logger.error(f"推奨: ./start_celery.sh start -w {min_workers}")
            return False

        # ワーカーの詳細情報
        for worker_name, worker_stats in stats.items():
            concurrency = worker_stats.get('pool', {}).get('max-concurrency', 'N/A')
            logger.info(f"  - {worker_name}: concurrency={concurrency}")

        logger.info("✅ Celeryワーカーの準備完了")
        return True

    except Exception as e:
        logger.error(f"❌ ワーカー確認エラー: {e}")
        return False


# ================================================================
# ユーティリティ
# ================================================================

def get_active_tasks() -> Dict:
    """
    アクティブなタスクの情報を取得

    Returns:
        ワーカーごとのアクティブタスク情報

    Example:
        >>> active = get_active_tasks()
        >>> for worker, tasks in active.items():
        ...     print(f"{worker}: {len(tasks)} active tasks")
    """
    try:
        inspect = app.control.inspect()
        active = inspect.active()
        return active or {}
    except Exception as e:
        logger.error(f"アクティブタスク取得エラー: {e}")
        return {}


def purge_queue(queue_name: str = 'qa_generation') -> int:
    """
    キューをクリア（未実行のタスクを削除）

    Args:
        queue_name: キュー名

    Returns:
        削除されたタスク数

    Warning:
        実行中のタスクには影響しません

    Example:
        >>> count = purge_queue('qa_generation')
        >>> print(f"Purged {count} tasks")
    """
    try:
        from celery.bin import amqp

        purged_count = app.control.purge()
        logger.warning(f"キュークリア: {purged_count}タスクを削除")
        return purged_count

    except Exception as e:
        logger.error(f"キュークリアエラー: {e}")
        return 0


# ================================================================
# デバッグ用
# ================================================================

if __name__ == "__main__":
    # テスト実行
    print("Celeryタスクモジュール")
    print(f"アプリケーション: {app.main}")
    print(f"ブローカー: {app.conf.broker_url}")

    # ワーカー確認
    if check_celery_workers():
        print("✅ ワーカーが起動しています")
    else:
        print("❌ ワーカーが起動していません")
        print("起動方法: ./start_celery.sh start -w 8")
