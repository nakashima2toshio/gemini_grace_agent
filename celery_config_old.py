#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
celery_config.py - Celery設定（スマート生成対応版）

改修内容（v2.1）:
- スマート生成に対応したタイムアウト設定
- メモリリーク対策の強化
- エラーハンドリングの改善
"""

from kombu import Queue
import os

# ================================================================
# ブローカー設定
# ================================================================

# Redisブローカー（環境変数で上書き可能）
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
broker_url = REDIS_URL
result_backend = REDIS_URL

# 接続プール設定
broker_pool_limit = 10  # 接続プールサイズ
broker_connection_timeout = 30  # 接続タイムアウト（秒）
broker_connection_retry = True  # 接続リトライ
broker_connection_max_retries = 5  # 最大リトライ回数

# シリアライザー設定
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']

# タイムゾーン設定
timezone = 'Asia/Tokyo'
enable_utc = True

# ================================================================
# タスク設定（スマート生成対応）
# ================================================================

# タイムアウト設定
# ✨ スマート生成は2回のLLM呼び出しがあるため、従来の2倍に設定
task_soft_time_limit = 300  # 5分（soft limit）
task_time_limit = 600  # 10分（hard limit）

# 従来方式の場合:
# task_soft_time_limit = 120  # 2分
# task_time_limit = 300  # 5分

# リトライ設定
task_acks_late = True  # タスク完了後にACK（失敗時に再投入）
task_reject_on_worker_lost = True  # ワーカー消失時に再投入
task_acks_on_failure_or_timeout = False  # 失敗/タイムアウト時はNACK

# 結果の設定
result_expires = 3600  # 結果の有効期限: 1時間
result_persistent = False  # 結果を永続化しない（メモリ節約）

# タスク圧縮
task_compression = 'gzip'  # タスクデータの圧縮
result_compression = 'gzip'  # 結果データの圧縮

# ================================================================
# キュー設定
# ================================================================

# キュー定義
task_queues = (
    Queue(
        'qa_generation',
        routing_key='qa.#',
        queue_arguments={
            'x-max-priority': 10,  # 優先度キュー（0-10）
        }
    ),
)

# ルーティング設定
task_routes = {
    'generate_qa_for_chunk': {
        'queue'      : 'qa_generation',
        'routing_key': 'qa.generate',
    },
}

# デフォルトキュー
task_default_queue = 'qa_generation'
task_default_exchange = 'celery'
task_default_routing_key = 'celery'

# ================================================================
# ワーカー設定
# ================================================================

# プリフェッチ設定
# ✨ スマート生成は処理時間が長いため、1タスクずつ取得
worker_prefetch_multiplier = 1  # 一度に1タスクのみ取得

# 従来方式の場合:
# worker_prefetch_multiplier = 4  # 4タスクまとめて取得

# ワーカーのメモリリーク対策
# ✨ スマート生成はメモリ使用量が多いため、早めに再起動
worker_max_tasks_per_child = 50  # 50タスクごとにワーカー再起動

# 従来方式の場合:
# worker_max_tasks_per_child = 100

# ワーカーのメモリ制限
worker_max_memory_per_child = 500000  # 500MB（キロバイト単位）

# ワーカーのタイムアウト
worker_disable_rate_limits = True  # レート制限を無効化

# ================================================================
# ログ設定
# ================================================================

# ログレベル
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = (
    '[%(asctime)s: %(levelname)s/%(processName)s]'
    '[%(task_name)s(%(task_id)s)] %(message)s'
)

# ログファイル
worker_hijack_root_logger = False  # ルートロガーを乗っ取らない
worker_redirect_stdouts = True  # stdout/stderrをログにリダイレクト
worker_redirect_stdouts_level = 'INFO'

# ================================================================
# モニタリング設定
# ================================================================

# イベント設定
worker_send_task_events = True  # タスクイベントを送信
task_send_sent_event = True  # タスク送信イベントを送信

# ================================================================
# パフォーマンスチューニング
# ================================================================

# 並列処理設定
worker_concurrency = None  # コマンドライン引数で指定（デフォルト: CPUコア数）

# プール設定
worker_pool = 'prefork'  # prefork プール（multiprocessing）
# 代替: 'threads', 'solo', 'gevent', 'eventlet'

# ソケットタイムアウト
broker_transport_options = {
    'visibility_timeout': 3600,  # タスクの可視性タイムアウト: 1時間
    'socket_keepalive'  : True,  # ソケットキープアライブ
    'socket_timeout'    : 300,  # ソケットタイムアウト: 5分
}

# ================================================================
# セキュリティ設定
# ================================================================

# 結果バックエンドのセキュリティ
result_backend_transport_options = {
    'master_name': 'mymaster',  # Redis Sentinelの場合
}

# タスクの署名検証（オプション）
# task_serializer = 'auth'
# task_always_eager = False

# ================================================================
# デバッグ設定
# ================================================================

# 開発モード設定（本番では False に）
CELERY_DEBUG = os.getenv('CELERY_DEBUG', 'False').lower() == 'true'

if CELERY_DEBUG:
    # デバッグモードでは即座に実行
    task_always_eager = True
    task_eager_propagates = True
    print("⚠️ CELERYデバッグモード: タスクは同期的に実行されます")

# ================================================================
# スマート生成用のカスタム設定
# ================================================================

# カスタム設定（アプリケーション側で参照可能）
SMART_GENERATION_DEFAULTS = {
    'timeout_multiplier': 2.0,  # スマート生成のタイムアウト倍率
    'max_retries'       : 3,  # 最大リトライ回数
    'retry_delay'       : 60,  # リトライ遅延（秒）
    'batch_size'        : 1,  # バッチサイズ（スマート生成では1推奨）
}

# ================================================================
# 環境別設定
# ================================================================

# 環境判定
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

if ENVIRONMENT == 'production':
    # 本番環境設定
    worker_max_tasks_per_child = 100  # 本番では長めに
    task_soft_time_limit = 600  # 10分
    task_time_limit = 1200  # 20分
    print("🚀 本番環境設定を適用")

elif ENVIRONMENT == 'staging':
    # ステージング環境設定
    worker_max_tasks_per_child = 75
    task_soft_time_limit = 450  # 7.5分
    task_time_limit = 900  # 15分
    print("🧪 ステージング環境設定を適用")

else:
    # 開発環境設定（デフォルト）
    print("💻 開発環境設定を適用")


# ================================================================
# 設定の検証
# ================================================================

def validate_config():
    """設定の妥当性を検証"""
    errors = []

    # タイムアウト設定の検証
    if task_time_limit <= task_soft_time_limit:
        errors.append("task_time_limit must be greater than task_soft_time_limit")

    # ワーカー設定の検証
    if worker_max_tasks_per_child < 10:
        errors.append("worker_max_tasks_per_child should be at least 10")

    # プリフェッチ設定の検証
    if worker_prefetch_multiplier > 4:
        errors.append("worker_prefetch_multiplier > 4 may cause memory issues")

    if errors:
        print("⚠️ 設定エラー:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("✅ 設定検証OK")
    return True


# 起動時に設定を検証
if __name__ != '__main__':
    validate_config()

# ================================================================
# 使用例・コメント
# ================================================================

"""
使用例:

1. Celeryワーカーの起動:
   ```bash
   # デフォルト設定で起動
   celery -A celery_config worker --loglevel=info

   # ワーカー数を指定
   celery -A celery_config worker --loglevel=info --concurrency=8

   # スクリプトを使用
   ./start_celery.sh start -w 8
   ```

2. 設定の確認:
   ```python
   from celery_config import *
   print(f"Broker: {broker_url}")
   print(f"Timeout: {task_time_limit}秒")
   ```

3. 環境変数での上書き:
   ```bash
   export REDIS_URL="redis://my-redis:6379/0"
   export ENVIRONMENT="production"
   export CELERY_DEBUG="false"
   ```

4. モニタリング:
   ```bash
   # Flowerの起動（推奨）
   celery -A celery_config flower
   # ブラウザで http://localhost:5555 にアクセス
   ```

パフォーマンスチューニング:

- スマート生成（デフォルト）:
  - worker_prefetch_multiplier = 1
  - worker_max_tasks_per_child = 50
  - task_time_limit = 600秒

- 従来方式（高速処理）:
  - worker_prefetch_multiplier = 4
  - worker_max_tasks_per_child = 100
  - task_time_limit = 300秒

トラブルシューティング:

1. タスクがタイムアウトする:
   - task_time_limitを増やす（現在: 600秒）
   - ワーカー数を増やす

2. メモリ使用量が多い:
   - worker_max_tasks_per_childを減らす（現在: 50）
   - worker_max_memory_per_childを設定

3. タスクが実行されない:
   - Redisが起動しているか確認: redis-cli ping
   - ワーカーが起動しているか確認: celery -A celery_config inspect active

4. 結果が取得できない:
   - result_expiresを確認（現在: 3600秒）
   - result_backendが正しく設定されているか確認
"""
