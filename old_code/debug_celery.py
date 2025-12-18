#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_celery.py - Celeryタスクのデバッグツール
"""

import time
import redis
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_redis_queues():
    """Redisキューの状態を確認"""
    r = redis.Redis(host='localhost', port=6379, db=0)

    print("\n=== Redis キュー状態 ===")

    # 全てのキーをチェック
    all_keys = r.keys('*')
    print(f"Redisに存在するキー数: {len(all_keys)}")

    # Celery関連のキューをチェック
    celery_queues = [k.decode() for k in all_keys if b'celery' in k]
    print(f"Celery関連のキー: {celery_queues}")

    # デフォルトキューのチェック
    default_queue = r.llen('celery')
    print(f"デフォルトキュー（'celery'）のタスク数: {default_queue}")

    # qa_generationキューのチェック
    qa_queue = r.llen('qa_generation')
    print(f"qa_generationキューのタスク数: {qa_queue}")

    # 結果バックエンドのチェック
    result_keys = [k.decode() for k in all_keys if b'celery-task-meta' in k]
    print(f"保存されている結果数: {len(result_keys)}")

    # 最新の結果をいくつか表示
    if result_keys:
        print("\n最新の結果（最大3個）:")
        for key in result_keys[:3]:
            result = r.get(key)
            if result:
                data = json.loads(result)
                print(f"  - {key}: status={data.get('status')}")


def check_worker_queues():
    """ワーカーが監視しているキューを確認"""
    from celery_tasks import app

    print("\n=== ワーカー設定 ===")

    # インスペクト
    inspect = app.control.inspect()

    # アクティブキューを確認
    active_queues = inspect.active_queues()
    if active_queues:
        for worker, queues in active_queues.items():
            print(f"\nワーカー: {worker}")
            print("監視キュー:")
            for q in queues:
                print(f"  - {q.get('name')} (routing_key: {q.get('routing_key')})")
    else:
        print("アクティブなワーカーが見つかりません")


def send_test_task_directly():
    """直接タスクを送信してテスト"""
    from celery_tasks import generate_qa_for_chunk_async

    print("\n=== 直接タスク送信テスト ===")

    test_chunk = {
        'id': 'debug_test',
        'text': 'これはデバッグ用のテストテキストです。',
        'doc_id': 'debug',
        'dataset_type': 'test'
    }

    test_config = {
        'lang': 'ja',
        'qa_per_chunk': 1
    }

    # デフォルトキューに送信
    print("1. デフォルトキューにタスク送信...")
    task1 = generate_qa_for_chunk_async.apply_async(
        args=[test_chunk, test_config, 'gpt-5-mini']
    )
    print(f"   タスクID: {task1.id}")

    # 3秒待って状態確認
    time.sleep(3)
    print(f"   状態: {task1.state}")

    # qa_generationキューに送信（明示的指定）
    print("\n2. qa_generationキューにタスク送信...")
    task2 = generate_qa_for_chunk_async.apply_async(
        args=[test_chunk, test_config, 'gpt-5-mini'],
        queue='qa_generation'
    )
    print(f"   タスクID: {task2.id}")

    # 3秒待って状態確認
    time.sleep(3)
    print(f"   状態: {task2.state}")

    return task1, task2


def monitor_task(task_id):
    """特定のタスクIDを監視"""
    from celery_tasks import app
    from celery.result import AsyncResult

    print(f"\n=== タスク {task_id} の監視 ===")

    task = AsyncResult(task_id, app=app)

    for i in range(10):
        state = task.state
        info = task.info

        print(f"  {i}秒: 状態={state}")

        if state == 'SUCCESS':
            print(f"  成功! 結果: {info}")
            break
        elif state == 'FAILURE':
            print(f"  失敗! エラー: {info}")
            break

        time.sleep(1)


def main():
    print("=" * 60)
    print("Celeryデバッグツール")
    print("=" * 60)

    # 1. Redisキューの状態確認
    check_redis_queues()

    # 2. ワーカーの設定確認
    check_worker_queues()

    # 3. テストタスク送信
    task1, task2 = send_test_task_directly()

    # 4. タスクの監視
    print("\n=== タスク監視 ===")
    print("デフォルトキューのタスク:")
    monitor_task(task1.id)

    print("\nqa_generationキューのタスク:")
    monitor_task(task2.id)

    # 5. 最終的なキュー状態
    print("\n=== 最終キュー状態 ===")
    check_redis_queues()


if __name__ == "__main__":
    main()