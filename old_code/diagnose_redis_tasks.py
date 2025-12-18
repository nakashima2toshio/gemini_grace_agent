#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Redis タスク状態 診断スクリプト
================================
Celeryタスクの状態をRedisから直接読み取り、問題を診断する
"""

import redis
import json
from collections import Counter, defaultdict
from datetime import datetime

def diagnose_redis_tasks():
    """Redisから全てのCeleryタスク状態を詳細診断"""

    # Redis接続
    r = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=True
    )

    # 全てのCeleryタスクキーを取得
    all_keys = r.keys('celery-task-meta-*')
    print(f"\n{'='*60}")
    print(f"Redis診断レポート")
    print(f"{'='*60}")
    print(f"タイムスタンプ: {datetime.now().isoformat()}")
    print(f"発見したタスクキー数: {len(all_keys)}")
    print(f"{'='*60}\n")

    # 状態別カウンター
    status_counter = Counter()
    result_type_counter = Counter()

    # 詳細分析用
    success_with_qa = 0
    success_without_qa = 0
    success_empty_result = 0
    success_null_result = 0
    success_wrong_type = 0

    total_qa_pairs = 0
    problematic_tasks = []

    for i, key in enumerate(sorted(all_keys)):
        try:
            data = r.get(key)
            if not data:
                status_counter['NO_DATA'] += 1
                problematic_tasks.append({
                    'key': key[:60],
                    'problem': 'NO_DATA',
                    'detail': 'Redis key exists but no data'
                })
                continue

            try:
                task_result = json.loads(data)
            except json.JSONDecodeError as e:
                status_counter['JSON_ERROR'] += 1
                problematic_tasks.append({
                    'key': key[:60],
                    'problem': 'JSON_ERROR',
                    'detail': str(e)[:100]
                })
                continue

            status = task_result.get('status', 'UNKNOWN')
            status_counter[status] += 1

            if status == 'SUCCESS':
                result = task_result.get('result')
                result_type = type(result).__name__
                result_type_counter[result_type] += 1

                if result is None:
                    success_null_result += 1
                    problematic_tasks.append({
                        'key': key[:60],
                        'problem': 'SUCCESS_NULL_RESULT',
                        'detail': 'status=SUCCESS but result=None'
                    })
                elif not isinstance(result, dict):
                    success_wrong_type += 1
                    problematic_tasks.append({
                        'key': key[:60],
                        'problem': 'SUCCESS_WRONG_TYPE',
                        'detail': f'result type is {result_type}, expected dict'
                    })
                elif 'success' not in result:
                    success_empty_result += 1
                    problematic_tasks.append({
                        'key': key[:60],
                        'problem': 'SUCCESS_NO_SUCCESS_KEY',
                        'detail': f'result dict has no "success" key. keys={list(result.keys())[:5]}'
                    })
                elif result.get('success'):
                    qa_pairs = result.get('qa_pairs', [])
                    if qa_pairs:
                        success_with_qa += 1
                        total_qa_pairs += len(qa_pairs)
                    else:
                        success_without_qa += 1
                        # Q/Aが0個なのは問題かもしれない
                        if i < 10 or i % 100 == 0:  # サンプルだけ記録
                            problematic_tasks.append({
                                'key': key[:60],
                                'problem': 'SUCCESS_ZERO_QA',
                                'detail': 'success=True but qa_pairs is empty'
                            })
                else:
                    # success=False
                    error = result.get('error', 'unknown')
                    if len(problematic_tasks) < 50:  # 多すぎないように制限
                        problematic_tasks.append({
                            'key': key[:60],
                            'problem': 'TASK_RETURNED_FAILURE',
                            'detail': f'success=False, error={str(error)[:80]}'
                        })

            elif status == 'FAILURE':
                traceback = task_result.get('traceback', 'no traceback')
                if len(problematic_tasks) < 50:
                    problematic_tasks.append({
                        'key': key[:60],
                        'problem': 'CELERY_FAILURE',
                        'detail': traceback[:200] if traceback else 'no traceback'
                    })

            elif status in ['PENDING', 'STARTED']:
                if len(problematic_tasks) < 50:
                    problematic_tasks.append({
                        'key': key[:60],
                        'problem': f'STATUS_{status}',
                        'detail': f'Task still in {status} state'
                    })

        except Exception as e:
            status_counter['EXCEPTION'] += 1
            problematic_tasks.append({
                'key': key[:60] if key else 'unknown',
                'problem': 'EXCEPTION',
                'detail': str(e)[:100]
            })

    # レポート出力
    print("【1. ステータス別サマリー】")
    print("-" * 40)
    for status, count in sorted(status_counter.items(), key=lambda x: -x[1]):
        pct = (count / len(all_keys)) * 100 if all_keys else 0
        print(f"  {status:15s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n【2. SUCCESS タスクの内訳】")
    print("-" * 40)
    print(f"  Q/Aペア生成成功    : {success_with_qa:5d}")
    print(f"  Q/Aペア0個(成功)   : {success_without_qa:5d}")
    print(f"  result=None        : {success_null_result:5d}")
    print(f"  result型エラー     : {success_wrong_type:5d}")
    print(f"  successキーなし    : {success_empty_result:5d}")

    print(f"\n【3. 結果型別サマリー】(SUCCESSのみ)")
    print("-" * 40)
    for result_type, count in sorted(result_type_counter.items(), key=lambda x: -x[1]):
        print(f"  {result_type:15s}: {count:5d}")

    print(f"\n【4. 総Q/Aペア数】")
    print("-" * 40)
    print(f"  合計: {total_qa_pairs:d}個")

    # 問題のあるタスク詳細
    if problematic_tasks:
        print(f"\n【5. 問題のあるタスク（最初の20件）】")
        print("-" * 40)
        problem_counter = Counter(t['problem'] for t in problematic_tasks)
        print(f"問題タイプ別:")
        for prob, count in sorted(problem_counter.items(), key=lambda x: -x[1]):
            print(f"  {prob}: {count}")

        print(f"\n詳細（最初の20件）:")
        for i, task in enumerate(problematic_tasks[:20]):
            print(f"\n  [{i+1}] {task['problem']}")
            print(f"      キー: {task['key']}")
            print(f"      詳細: {task['detail'][:100]}")

    print(f"\n{'='*60}")
    print(f"診断完了")
    print(f"{'='*60}")

    # 期待値との比較
    expected = 771
    actual_success = success_with_qa + success_without_qa
    if actual_success < expected:
        print(f"\n⚠️ 警告: 期待={expected}, 実際のSUCCESS(Q/A取得可能)={actual_success}")
        print(f"   差分: {expected - actual_success}タスク")

        # 取得できなかった原因の推測
        print(f"\n【取得失敗の原因推測】")
        if success_null_result > 0:
            print(f"  - result=Nullのタスク: {success_null_result}個")
            print(f"    → Celeryワーカーがresultを正しく返していない可能性")
        if success_wrong_type > 0:
            print(f"  - 結果型不正のタスク: {success_wrong_type}個")
            print(f"    → タスク関数が辞書以外を返している可能性")
        if success_empty_result > 0:
            print(f"  - successキーなしのタスク: {success_empty_result}個")
            print(f"    → タスク関数の戻り値形式が不正")
    else:
        print(f"\n✅ 全{expected}タスクが正常に完了")

    return status_counter, problematic_tasks

if __name__ == "__main__":
    diagnose_redis_tasks()
