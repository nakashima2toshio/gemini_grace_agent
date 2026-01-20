#!/bin/bash
# start_celery_fixed.sh - Celeryワーカー起動スクリプト（修正版）
#
# 問題点:
# 元のスクリプトで -Q オプションが指定されていなかったため、
# ワーカーがデフォルトキュー 'celery' を監視していなかった
#
# 使用方法:
#   ./start_celery_fixed.sh start -w 8
#   ./start_celery_fixed.sh stop
#   ./start_celery_fixed.sh status
#   ./start_celery_fixed.sh restart -w 8

set -e

# プロジェクトルート
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

# ログディレクトリ
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 環境変数
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/helper"

# ★★★ 重要: すべてのキューを指定 ★★★
QUEUES="celery,high_priority,normal_priority,low_priority"

# デフォルト設定
WORKERS=8
LOGLEVEL="INFO"

# ヘルプ表示
show_help() {
    echo "使用方法: $0 {start|stop|restart|status} [-w workers]"
    echo ""
    echo "コマンド:"
    echo "  start   - ワーカーを起動"
    echo "  stop    - ワーカーを停止"
    echo "  restart - ワーカーを再起動"
    echo "  status  - ワーカーの状態を表示"
    echo ""
    echo "オプション:"
    echo "  -w, --workers  ワーカー数 (デフォルト: 8)"
    echo ""
    echo "例:"
    echo "  $0 start -w 4"
    echo "  $0 stop"
}

# ワーカー停止
stop_workers() {
    echo "Celeryワーカーを停止中..."
    pkill -9 -f "celery.*worker" 2>/dev/null || true
    pkill -9 -f "celery_config" 2>/dev/null || true
    sleep 2

    # 確認
    remaining=$(ps aux | grep -E 'celery.*worker' | grep -v grep | wc -l)
    if [ "$remaining" -eq 0 ]; then
        echo "✅ ワーカーを停止しました"
    else
        echo "⚠️ まだプロセスが残っています"
        ps aux | grep -E 'celery.*worker' | grep -v grep
    fi
}

# ワーカー起動
start_workers() {
    echo "Celeryワーカーを起動中..."
    echo "プロジェクトルート: $PROJECT_ROOT"
    echo "PYTHONPATH: $PYTHONPATH"
    echo "ワーカー数: $WORKERS"
    echo "監視キュー: $QUEUES"  # ★ 重要

    # helper/helper_llm.py の存在確認
    if [ -f "$PROJECT_ROOT/helper/helper_llm.py" ]; then
        echo "✅ helper/helper_llm.py が見つかりました"
    else
        echo "⚠️ helper/helper_llm.py が見つかりません"
    fi

    # ★★★ 修正ポイント: -Q オプションで全キューを指定 ★★★
    nohup celery -A celery_config worker \
        --loglevel=$LOGLEVEL \
        --concurrency=$WORKERS \
        -Q $QUEUES \
        -n qa_worker@%h \
        > "$LOG_DIR/celery_qa_worker.log" 2>&1 &

    sleep 3

    # 起動確認
    if pgrep -f "celery.*worker" > /dev/null; then
        echo "✅ Celeryワーカーを起動しました（$WORKERS ワーカー）"
        echo "ログファイル: $LOG_DIR/celery_qa_worker.log"

        # 監視キューの確認
        echo ""
        echo "📋 監視キューの確認中..."
        sleep 2
        python3 -c "
from celery_config import app
inspect = app.control.inspect()
queues = inspect.active_queues()
if queues:
    for worker, q_list in queues.items():
        print(f'ワーカー: {worker}')
        for q in q_list:
            print(f'  ✅ キュー: {q[\"name\"]}')
else:
    print('⚠️ キュー情報を取得できません')
"
    else
        echo "❌ ワーカーの起動に失敗しました"
        echo "ログを確認: tail -50 $LOG_DIR/celery_qa_worker.log"
        exit 1
    fi
}

# ステータス確認
show_status() {
    echo "Celeryワーカーの状態:"

    # プロセス確認
    if pgrep -f "celery.*worker" > /dev/null; then
        echo "✅ ワーカーが起動しています"
        ps aux | grep -E 'celery.*worker' | grep -v grep
    else
        echo "❌ ワーカーが起動していません"
        return
    fi

    # 詳細情報
    python3 -c "
from celery_config import app
inspect = app.control.inspect()

print()
print('--- ワーカー統計 ---')
stats = inspect.stats()
if stats:
    for worker, info in stats.items():
        pool = info.get('pool', {})
        print(f'{worker}: concurrency={pool.get(\"max-concurrency\", \"N/A\")}')
else:
    print('⚠️ 統計情報を取得できません')

print()
print('--- 監視キュー ---')
queues = inspect.active_queues()
if queues:
    for worker, q_list in queues.items():
        for q in q_list:
            print(f'  ✅ {q[\"name\"]}')
else:
    print('⚠️ キュー情報を取得できません')
"
}

# Redis確認
check_redis() {
    echo "Redisサーバーを確認中..."
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redisサーバーが起動しています"
        return 0
    else
        echo "❌ Redisサーバーが起動していません"
        echo "起動方法: brew services start redis (macOS)"
        return 1
    fi
}

# メイン処理
COMMAND=${1:-help}
shift || true

# オプション解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

case $COMMAND in
    start)
        check_redis
        start_workers
        ;;
    stop)
        stop_workers
        ;;
    restart)
        stop_workers
        redis-cli FLUSHALL > /dev/null
        echo "✅ Redisをクリアしました"
        check_redis
        start_workers
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "不明なコマンド: $COMMAND"
        show_help
        exit 1
        ;;
esac
