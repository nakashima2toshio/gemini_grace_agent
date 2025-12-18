#!/bin/bash

# register_qa_to_qdrant.sh
# a20_output_qa_csv.pyで生成されたCSVファイルをQdrantに登録するスクリプト

echo "=================================================="
echo "Q&A データのQdrant登録スクリプト"
echo "=================================================="

# 1. CSVファイルの生成確認と実行
echo ""
echo "[1/3] CSVファイルの準備..."
echo "--------------------------------------------------"

if [ ! -f "qa_output/a02_qa_pairs_cc_news.csv" ] || [ ! -f "qa_output/a03_qa_pairs_cc_news.csv" ] || [ ! -f "qa_output/a10_qa_pairs_cc_news.csv" ]; then
    echo "CSVファイルが見つかりません。a20_output_qa_csv.pyを実行します..."
    python a20_output_qa_csv.py
    if [ $? -ne 0 ]; then
        echo "エラー: CSVファイルの生成に失敗しました"
        exit 1
    fi
else
    echo "CSVファイルが存在します:"
    ls -lh qa_output/a02_qa_pairs_cc_news.csv
    ls -lh qa_output/a03_qa_pairs_cc_news.csv
    ls -lh qa_output/a10_qa_pairs_cc_news.csv
fi

# 2. Qdrantサーバーの確認
echo ""
echo "[2/3] Qdrantサーバーの確認..."
echo "--------------------------------------------------"

# Qdrantのヘルスチェック
curl -s http://localhost:6333/health > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "警告: Qdrantサーバーが起動していません"
    echo "以下のコマンドでQdrantを起動してください:"
    echo "  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant"
    echo ""
    read -p "Qdrantが起動していることを確認したら、Enterキーを押してください..."
fi

# 3. Qdrantへのデータ登録
echo ""
echo "[3/3] Qdrantへのデータ登録..."
echo "--------------------------------------------------"

# OpenAI APIキーの確認
if [ -z "$OPENAI_API_KEY" ]; then
    echo "エラー: OPENAI_API_KEY環境変数が設定されていません"
    echo "以下のコマンドで設定してください:"
    echo "  export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

echo "Qdrantにデータを登録します..."
echo "オプション:"
echo "  --recreate: コレクションを再作成"
echo "  --include-answer: 埋め込みにanswerも含める"
echo ""

# デフォルトオプションで実行（必要に応じて変更）
python a40_qdrant_registration.py --recreate --include-answer

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ データ登録が完了しました"
    echo "=================================================="
    echo ""
    echo "検索テストを実行するには:"
    echo "  python a30_qdrant_registration.py --search '気候変動' --method a02_make_qa"
    echo ""
else
    echo ""
    echo "❌ データ登録中にエラーが発生しました"
    exit 1
fi