#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
celery_tasks.py - Celeryタスク定義（修正版 v2.6）

修正内容（v2.6）:
- ★重要★ generate_qa_dataset() の呼び出しを正しいシグネチャに修正
- 削除: provider 引数（存在しない）
- 確認済みシグネチャ:
    generate_qa_dataset(
        chunks, dataset_type, model, chunk_batch_size, merge_chunks,
        min_tokens, max_tokens, config, client, use_smart_generation
    )
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional

# ================================================================
# 重要: プロジェクトルートをsys.pathに追加
# ================================================================
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# helper/ディレクトリもsys.pathに追加
helper_path = project_root / 'helper'
if helper_path.exists() and str(helper_path) not in sys.path:
    sys.path.insert(0, str(helper_path))

from celery_config import app

logger = logging.getLogger(__name__)


# ================================================================
# Q/A生成タスク
# ================================================================

def submit_unified_qa_generation(
        chunks: List[Dict],
        config: Dict,
        model: str,
        provider: str = "gemini",  # 互換性のために残すが使用しない
        use_smart_generation: bool = True
) -> List:
    """
    チャンクのQ/A生成タスクを並列実行

    Args:
        chunks: チャンクのリスト
        config: データセット設定
        model: 使用するモデル（例: "gemini-2.0-flash"）
        provider: 互換性のために残すが使用しない
        use_smart_generation: スマート生成を使用するか（デフォルト: True）

    Returns:
        Celeryタスクのリスト（AsyncResultオブジェクト）
    """
    logger.info(f"Celeryタスクを投入: {len(chunks)}チャンク")
    logger.info(f"モデル: {model}")
    logger.info(f"生成モード: {'スマート生成' if use_smart_generation else '従来方式'}")

    task_list = []
    for chunk in chunks:
        # ★ 正しい引数のみを渡す（provider は渡さない）
        task = generate_qa_for_chunk_task.apply_async(
            args=(chunk, config, model, use_smart_generation)
        )
        task_list.append(task)

    logger.info(f"タスク投入完了: {len(task_list)}個")
    return task_list


@app.task(
    name='generate_qa_for_chunk',
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True
)
def generate_qa_for_chunk_task(
        self,
        chunk: Dict,
        config: Dict,
        model: str,
        use_smart_generation: bool = True
) -> List[Dict]:
    """
    単一チャンクのQ/A生成タスク

    Args:
        self: Celeryタスクインスタンス
        chunk: チャンク
        config: データセット設定
        model: 使用するモデル
        use_smart_generation: スマート生成を使用するか

    Returns:
        Q/Aペアのリスト
    """
    chunk_id = chunk.get('id', 'unknown')

    logger.info("=" * 60)
    logger.info(f"[ワーカー] タスク開始")
    logger.info("=" * 60)
    logger.info(f"  chunk_id: {chunk_id}")
    logger.info(f"  model: {model}")
    logger.info(f"  use_smart_generation: {use_smart_generation}")
    logger.info(f"  sys.path[0]: {sys.path[0]}")

    # qa_gen_pathを事前に定義
    qa_gen_path = project_root / 'qa_generation'

    try:
        # インポート
        logger.info(f"[ワーカー] qa_generation.generationをインポート中...")
        from qa_generation.generation import generate_qa_dataset
        logger.info(f"[ワーカー] ✅ インポート成功")

        # ★★★ 修正ポイント: 正しいシグネチャで呼び出し ★★★
        # generate_qa_dataset(chunks, dataset_type, model, ..., use_smart_generation)
        logger.info(f"[ワーカー] Q/A生成開始: chunk={chunk_id}")

        qa_pairs = generate_qa_dataset(
            chunks=[chunk],
            dataset_type=config.get("type", "unknown"),
            model=model,
            chunk_batch_size=1,  # 単一チャンク処理
            merge_chunks=False,  # マージ不要（既に1チャンク）
            min_tokens=150,
            max_tokens=400,
            config=config,
            client=None,  # 自動生成させる
            use_smart_generation=use_smart_generation
        )

        logger.info(f"[ワーカー] ✅ タスク完了: chunk={chunk_id}, Q/A数={len(qa_pairs)}")
        logger.info("=" * 60)
        return qa_pairs

    except ImportError as exc:
        logger.error("=" * 60)
        logger.error(f"[ワーカー] ❌ モジュールインポートエラー")
        logger.error("=" * 60)
        logger.error(f"  chunk_id: {chunk_id}")
        logger.error(f"  エラー: {exc}")
        logger.error(f"  sys.path: {sys.path[:3]}")
        logger.error("=" * 60)

        raise ImportError(
            f"qa_generation.generationのインポート失敗: {exc}\n"
            f"sys.path[0]={sys.path[0]}"
        )

    except ValueError as exc:
        logger.error(f"[ワーカー] ❌ データ形式エラー: chunk={chunk_id}, error={exc}")
        raise ValueError(f"データ形式エラー: {exc}")

    except Exception as exc:
        logger.error(f"[ワーカー] ❌ タスクエラー: chunk={chunk_id}")
        logger.error(f"[ワーカー] エラー詳細:", exc_info=True)

        # リトライ
        if self.request.retries < self.max_retries:
            retry_count = self.request.retries + 1
            logger.warning(f"[ワーカー] リトライ {retry_count}/{self.max_retries}: chunk={chunk_id}")
            raise self.retry(exc=exc, countdown=60)
        else:
            logger.error(f"[ワーカー] ❌ 最大リトライ回数超過: chunk={chunk_id}")
            raise RuntimeError(f"最大リトライ回数超過: {exc}")


# ================================================================
# 結果収集
# ================================================================

def collect_results(tasks: List, timeout: int = 600) -> List[Dict]:
    """
    Celeryタスクの結果を収集
    """
    logger.info(f"結果収集中: {len(tasks)}タスク, timeout={timeout}秒")

    all_qa_pairs = []
    success_count = 0
    failed_count = 0
    timeout_count = 0
    error_details = []

    for i, task in enumerate(tasks, 1):
        try:
            result = task.get(timeout=timeout)

            if result:
                all_qa_pairs.extend(result)
                success_count += 1
                logger.debug(f"タスク {i}/{len(tasks)}: ✅ 成功（Q/A数={len(result)}）")
            else:
                failed_count += 1
                logger.warning(f"タスク {i}/{len(tasks)}: ⚠️ 結果が空")

        except Exception as e:
            error_msg = str(e)

            if "timeout" in error_msg.lower():
                timeout_count += 1
                logger.error(f"タスク {i}/{len(tasks)}: ⏱️ タイムアウト")
                error_details.append(f"タスク{i}: タイムアウト")
            else:
                logger.error(f"タスク {i}/{len(tasks)}: ❌ エラー: {error_msg}")
                error_details.append(f"タスク{i}: {error_msg[:100]}")

            failed_count += 1
            continue

    # サマリー
    logger.info("=" * 60)
    logger.info("結果収集完了")
    logger.info("=" * 60)
    logger.info(f"  成功: {success_count}/{len(tasks)}")
    logger.info(f"  失敗: {failed_count}/{len(tasks)}")
    logger.info(f"  タイムアウト: {timeout_count}/{len(tasks)}")
    logger.info(f"  Q/A総数: {len(all_qa_pairs)}")

    if error_details:
        logger.error("\n⚠️ エラー詳細:")
        for detail in error_details[:5]:
            logger.error(f"  - {detail}")

    logger.info("=" * 60)

    return all_qa_pairs


# ================================================================
# ワーカー状態確認
# ================================================================

def check_celery_workers(min_workers: int = 1) -> bool:
    """Celeryワーカーの状態確認"""
    try:
        inspect = app.control.inspect()
        stats = inspect.stats()

        if stats is None:
            logger.error("❌ Celeryワーカーが応答しません")
            return False

        worker_count = len(stats)
        logger.info(f"アクティブなワーカー: {worker_count}個")

        if worker_count < min_workers:
            logger.error(f"❌ ワーカー数が不足: {worker_count} < {min_workers}")
            return False

        for worker_name, worker_stats in stats.items():
            pool_info = worker_stats.get('pool', {})
            concurrency = pool_info.get('max-concurrency', 'N/A')
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
    """アクティブなタスクの情報を取得"""
    try:
        inspect = app.control.inspect()
        active = inspect.active()
        return active or {}
    except Exception as e:
        logger.error(f"アクティブタスク取得エラー: {e}")
        return {}


def purge_queue(queue_name: str = 'celery') -> int:
    """キューをクリア"""
    try:
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
    print("=" * 60)
    print("Celeryタスクモジュール - 修正版 v2.6")
    print("=" * 60)
    print(f"プロジェクトルート: {project_root}")
    print(f"アプリケーション: {app.main}")
    print(f"ブローカー: {app.conf.broker_url}")
    print()

    # インポートテスト
    print("generate_qa_datasetのインポートテスト...")
    try:
        from qa_generation.generation import generate_qa_dataset
        import inspect as py_inspect

        print("✅ インポート成功")

        # シグネチャ確認
        sig = py_inspect.signature(generate_qa_dataset)
        print(f"シグネチャ: {sig}")
        print()
        print("パラメータ一覧:")
        for name, param in sig.parameters.items():
            default = param.default if param.default != py_inspect.Parameter.empty else "(必須)"
            print(f"  - {name}: {default}")

    except ImportError as e:
        print(f"❌ インポート失敗: {e}")
    print()

    # ワーカー確認
    print("ワーカー状態を確認中...")
    if check_celery_workers():
        print("✅ ワーカーが起動しています")
    else:
        print("❌ ワーカーが起動していません")
