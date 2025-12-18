#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stuck結果の緊急回収スクリプト
768/771で止まったタスクの結果を強制的に収集
"""

import redis
import json
import sys

def recover_all_results():
    """Redisから全てのタスク結果を回収"""

    # Redis接続
    r = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True
    )

    # 全てのCeleryタスクキーを取得
    all_keys = r.keys('celery-task-meta-*')
    print(f"発見したタスクキー数: {len(all_keys)}")

    success_count = 0
    failed_count = 0
    total_qa_pairs = 0

    for key in all_keys:
        try:
            data = r.get(key)
            if data:
                task_result = json.loads(data)
                status = task_result.get('status')

                if status == 'SUCCESS':
                    result = task_result.get('result', {})
                    if isinstance(result, dict) and result.get('success'):
                        qa_pairs = result.get('qa_pairs', [])
                        total_qa_pairs += len(qa_pairs)
                        success_count += 1
                    else:
                        failed_count += 1
                elif status == 'FAILURE':
                    failed_count += 1
        except Exception as e:
            print(f"エラー処理中: {key[:50]}... - {str(e)[:50]}")

    print(f"""
    ========================================
    Redis結果回収完了:
    - 成功タスク: {success_count}
    - 失敗タスク: {failed_count}
    - 総Q/Aペア数: {total_qa_pairs}
    - 総タスク数: {len(all_keys)}
    ========================================
    """)

    # 771タスクと比較
    expected = 771
    if success_count < expected:
        print(f"⚠️ 警告: {expected - success_count}タスクが未完了の可能性があります")
    elif success_count == expected:
        print("✅ 全771タスクが正常に完了しています")

    return success_count, failed_count, total_qa_pairs

if __name__ == "__main__":
    recover_all_results()