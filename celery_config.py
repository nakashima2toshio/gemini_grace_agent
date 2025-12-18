#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
celery_config.py - Celery設定ファイル
Q/A生成タスクの並列処理用設定
"""

import os
from kombu import Exchange, Queue
from celery import Celery
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

# Celeryアプリケーション初期化
app = Celery('qa_generation')

# Redis設定
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Celery設定クラス
class CeleryConfig:
    # ブローカー設定（Redis）
    broker_url = REDIS_URL
    result_backend = REDIS_URL

    # タスク設定
    task_serializer = 'json'
    accept_content = ['json']
    result_serializer = 'json'
    timezone = 'Asia/Tokyo'
    enable_utc = True

    # ワーカー設定
    worker_prefetch_multiplier = 1  # 各ワーカーが一度に取得するタスク数
    worker_max_tasks_per_child = 50  # メモリリーク対策
    worker_disable_rate_limits = False

    # タスク実行設定
    task_acks_late = True  # タスク完了後にACK
    task_reject_on_worker_lost = True
    task_time_limit = 300  # タスクタイムアウト（5分）
    task_soft_time_limit = 270  # ソフトタイムアウト（4.5分）

    # レート制限（OpenAI API制限対応）
    task_annotations = {
        'tasks.process_chunk_task': {
            'rate_limit': '50/m',  # 分あたり50リクエスト
        },
        'tasks.process_batch_task': {
            'rate_limit': '10/m',  # 分あたり10バッチ
        }
    }

    # キュー設定
    task_routes = {
        'tasks.process_chunk_task': 'high_priority',
        'tasks.process_batch_task': 'normal_priority',
        'tasks.aggregate_results_task': 'low_priority',
    }

    # キュー定義
    task_queues = (
        Queue('high_priority', Exchange('high_priority'), routing_key='high'),
        Queue('normal_priority', Exchange('normal_priority'), routing_key='normal'),
        Queue('low_priority', Exchange('low_priority'), routing_key='low'),
    )

    # リトライ設定
    task_autoretry_for = (Exception,)
    task_retry_kwargs = {
        'max_retries': 3,
        'countdown': 60,  # 60秒後にリトライ
        'retry_jitter': True,  # ジッター追加
    }

    # 結果の有効期限
    result_expires = 3600  # 1時間

    # Celery Beat設定（定期タスク用）
    beat_schedule = {
        'cleanup-old-results': {
            'task': 'tasks.cleanup_old_results',
            'schedule': 3600.0,  # 1時間ごと
        },
    }

# 設定を適用
app.config_from_object(CeleryConfig())

# OpenAI API設定
OPENAI_CONFIG = {
    'api_key': os.getenv('OPENAI_API_KEY'),
    'models': {
        'gpt-5-mini': {
            'rpm_limit': 3500,  # Requests Per Minute
            'tpm_limit': 200000,  # Tokens Per Minute
            'max_retries': 3,
            'retry_delay': 60
        },
        'gpt-4': {
            'rpm_limit': 500,
            'tpm_limit': 40000,
            'max_retries': 3,
            'retry_delay': 120
        },
        'gpt-4o': {
            'rpm_limit': 500,
            'tpm_limit': 30000,
            'max_retries': 3,
            'retry_delay': 120
        },
        'gpt-4o-mini': {
            'rpm_limit': 500,
            'tpm_limit': 200000,
            'max_retries': 3,
            'retry_delay': 60
        }
    }
}

# ログ設定
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logs/celery_qa_generation.log',
            'formatter': 'default',
        },
    },
    'loggers': {
        'celery': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
        'tasks': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
    },
}

# エクスポート
__all__ = ['app', 'CeleryConfig', 'OPENAI_CONFIG', 'LOGGING_CONFIG']

if __name__ == '__main__':
    # 設定確認用
    print("Celery Configuration:")
    print(f"Broker URL: {CeleryConfig.broker_url}")
    print(f"Result Backend: {CeleryConfig.result_backend}")
    print(f"Task Time Limit: {CeleryConfig.task_time_limit}s")
    print("\nOpenAI Configuration:")
    for model, config in OPENAI_CONFIG['models'].items():
        print(f"  {model}: RPM={config['rpm_limit']}, TPM={config['tpm_limit']}")