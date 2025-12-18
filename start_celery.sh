#!/bin/bash
# start_celery.sh - Celeryワーカー起動スクリプト
# =============================================
# Q/A生成のCeleryワーカーを起動・管理

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# デフォルト設定
WORKERS=24
LOG_LEVEL="info"
QUEUE_NAME="qa_generation"

# ヘルプメッセージ
usage() {
    echo "Usage: $0 [start|stop|status|restart] [options]"
    echo ""
    echo "Commands:"
    echo "  start    Celeryワーカーを起動"
    echo "  stop     Celeryワーカーを停止"
    echo "  status   ワーカーのステータスを確認"
    echo "  restart  ワーカーを再起動"
    echo ""
    echo "Options:"
    echo "  -w, --workers NUM    ワーカー数（デフォルト: 4）"
    echo "  -l, --loglevel LEVEL ログレベル（debug|info|warning|error）"
    echo ""
    echo "Example:"
    echo "  $0 start -w 8        # 8ワーカーで起動"
    echo "  $0 status            # ステータス確認"
    exit 1
}

# Redisサーバーの確認
check_redis() {
    echo -e "${YELLOW}Redisサーバーを確認中...${NC}"
    if redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Redisサーバーが起動しています${NC}"
        return 0
    else
        echo -e "${RED}✗ Redisサーバーが起動していません${NC}"
        echo "以下のコマンドでRedisを起動してください："
        echo "  brew services start redis  # macOS"
        echo "  sudo systemctl start redis  # Linux"
        return 1
    fi
}

# Celeryワーカーの完全クリーンアップ
cleanup_workers() {
    # 古いPIDファイルを削除
    if [ -f /tmp/celery_qa.pid ]; then
        rm -f /tmp/celery_qa.pid
    fi

    # 全てのCeleryワーカープロセスを強制終了
    pkill -9 -f "celery.*worker" 2>/dev/null
    sleep 1
}

# Celeryワーカーの起動
start_workers() {
    echo -e "${YELLOW}Celeryワーカーを起動中...${NC}"

    # 既存のワーカーをチェック
    if pgrep -f "celery.*worker.*qa_generation" > /dev/null; then
        echo -e "${YELLOW}既にワーカーが起動しています${NC}"
        echo -e "${YELLOW}クリーンアップして再起動します...${NC}"
        cleanup_workers
    fi

    # Celeryワーカーを起動
    celery -A celery_tasks worker \
        --loglevel=${LOG_LEVEL} \
        --concurrency=${WORKERS} \
        --pool=prefork \
        --queues=${QUEUE_NAME} \
        --hostname=qa_worker@%h \
        --pidfile=/tmp/celery_qa.pid \
        --logfile=logs/celery_qa_%n.log \
        --detach

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Celeryワーカーを起動しました（${WORKERS}ワーカー）${NC}"
        echo "ログファイル: logs/celery_qa_*.log"
        return 0
    else
        echo -e "${RED}✗ ワーカーの起動に失敗しました${NC}"
        return 1
    fi
}

# Celeryワーカーの停止
stop_workers() {
    echo -e "${YELLOW}Celeryワーカーを停止中...${NC}"

    # PIDファイルから停止
    if [ -f /tmp/celery_qa.pid ]; then
        PID=$(cat /tmp/celery_qa.pid)
        kill -TERM $PID 2>/dev/null
        sleep 2

        # 完全に停止したか確認
        if ! kill -0 $PID 2>/dev/null; then
            rm -f /tmp/celery_qa.pid
            echo -e "${GREEN}✓ ワーカーを停止しました${NC}"
        else
            echo -e "${YELLOW}強制終了を試みます...${NC}"
            kill -9 $PID 2>/dev/null
            rm -f /tmp/celery_qa.pid
            echo -e "${GREEN}✓ ワーカーを強制停止しました${NC}"
        fi
    else
        # プロセス名で検索して停止
        pkill -f "celery.*worker.*qa_generation"
        echo -e "${GREEN}✓ ワーカーを停止しました${NC}"
    fi
}

# ステータス確認
check_status() {
    echo -e "${YELLOW}=== Celeryワーカー ステータス ===${NC}"

    # Redisの状態
    if redis-cli ping > /dev/null 2>&1; then
        echo -e "Redis: ${GREEN}✓ 起動中${NC}"

        # キューの状態を確認
        QUEUE_LENGTH=$(redis-cli llen celery 2>/dev/null || echo "0")
        echo -e "キュー長: ${QUEUE_LENGTH}"
    else
        echo -e "Redis: ${RED}✗ 停止中${NC}"
    fi

    # ワーカーの状態
    if pgrep -f "celery.*worker.*qa_generation" > /dev/null; then
        echo -e "ワーカー: ${GREEN}✓ 起動中${NC}"

        # 詳細情報
        celery -A celery_tasks inspect active --timeout=2 2>/dev/null || true
        celery -A celery_tasks inspect stats --timeout=2 2>/dev/null | head -20 || true
    else
        echo -e "ワーカー: ${RED}✗ 停止中${NC}"
    fi

    # ログファイルの最新行
    if [ -f logs/celery_qa_*.log ]; then
        echo -e "\n${YELLOW}=== 最新のログ ===${NC}"
        tail -5 logs/celery_qa_*.log 2>/dev/null || true
    fi
}

# logsディレクトリの作成
mkdir -p logs

# コマンドライン引数の解析
COMMAND=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -l|--loglevel)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "不明なオプション: $1"
            usage
            ;;
    esac
done

# メイン処理
case $COMMAND in
    start)
        check_redis && start_workers
        ;;
    stop)
        stop_workers
        ;;
    status)
        check_status
        ;;
    restart)
        stop_workers
        sleep 2
        check_redis && start_workers
        ;;
    *)
        usage
        ;;
esac