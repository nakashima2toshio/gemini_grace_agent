#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gemini APIレート制限対策パッチ
================================
Q/A生成タスクのレート制限問題を修正
"""

import time
from functools import wraps
import threading
from collections import deque
from datetime import datetime, timedelta

class RateLimiter:
    """Gemini APIのレート制限管理"""

    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.request_times = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """必要に応じて待機"""
        with self.lock:
            now = datetime.now()
            # 1分以上前のリクエストを削除
            cutoff = now - timedelta(minutes=1)
            while self.request_times and self.request_times[0] < cutoff:
                self.request_times.popleft()

            # レート制限に達している場合は待機
            if len(self.request_times) >= self.max_requests:
                oldest = self.request_times[0]
                wait_time = (oldest + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    print(f"レート制限のため {wait_time:.1f}秒待機...")
                    time.sleep(wait_time)
                    # 再帰的に再チェック
                    self.wait_if_needed()
            else:
                # リクエスト時刻を記録
                self.request_times.append(now)

# グローバルレート制限インスタンス
gemini_rate_limiter = RateLimiter(max_requests_per_minute=50)  # 安全マージンを持たせる

def with_rate_limit(func):
    """レート制限デコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        gemini_rate_limiter.wait_if_needed()
        return func(*args, **kwargs)
    return wrapper


# celery_tasks.pyの generate_qa_unified_async に適用する修正例：
"""
# celery_tasks.py の修正箇所

# ファイル冒頭に追加
from celery_rate_limit_fix import with_rate_limit

# generate_qa_unified_async関数の修正（296行目あたり）
@app.task(bind=True, max_retries=3)
@with_rate_limit  # ← この行を追加
def generate_qa_unified_async(self, chunk_data: Dict, ...):
    # 既存のコード

"""

# 代替案：より簡単な修正方法
def add_simple_delay():
    """
    シンプルな遅延追加
    celery_tasks.pyの generate_qa_unified_async 内の
    Gemini API呼び出し前に追加
    """
    import random
    # ランダムな遅延を追加（0.5〜2秒）
    delay = random.uniform(0.5, 2.0)
    time.sleep(delay)


# 使用例とトラブルシューティング
if __name__ == "__main__":
    print("=== Gemini API レート制限対策 ===")
    print("\n修正方法:")
    print("1. このファイルを celery_rate_limit_fix.py として保存")
    print("2. celery_tasks.py の以下を修正:")
    print("   - ファイル冒頭: from celery_rate_limit_fix import with_rate_limit")
    print("   - generate_qa_unified_async関数に @with_rate_limit デコレータを追加")
    print("\n3. または、簡易的な修正として:")
    print("   - generate_qa_unified_async内のAPI呼び出し前に time.sleep(1) を追加")
    print("\n4. UIでワーカー数を 4〜8 に減らす")
    print("\n推奨設定:")
    print("- Celeryワーカー数: 4〜8")
    print("- バッチチャンク数: 1〜2")
    print("- 最大ドキュメント数: 5〜10（テスト時）")