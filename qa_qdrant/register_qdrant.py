#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
register_qdrant.py - CSVデータをQdrantに登録するCLIツール
=======================================================
make_qa.py で生成されたQ/Aペア、または前処理済みのテキストデータを
Embedding（ベクトル化）してQdrantデータベースに登録します。

推奨される使用法（高精度RAG向け）:
make_qa.py で生成した "qa_pairs_*.csv" を入力とし、
--collection には "qa_{データセット名}" のような名前を指定してください。

# 生成されたファイルを確認
ls -t qa_output/pipeline/qa_pairs_fineweb_edu_ja_*.csv | head -n 1

# 登録実行 (例: ファイル名が qa_pairs_fineweb_edu_ja_20251230_123456.csv の場合)：
python register_qdrant.py \
--input-file qa_output/pipeline/qa_pairs_fineweb_edu_ja_20251230_123456.csv \
--collection qa_fineweb_edu_ja \
--recreate \
--batch-size 100

"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
register_qdrant.py - CSVデータをQdrantに登録するCLIツール（修正版）
=======================================================
🔧 qa_qdrant/ ディレクトリ配下に移動後の修正版

使用方法:
cd /path/to/project_root
python qa_qdrant/register_qdrant.py \
--input-file qa_output/pipeline/qa_pairs_fineweb_edu_ja_20251230_123456.csv \
--collection qa_fineweb_edu_ja \
--recreate \
--batch-size 100
"""

import sys
import os

# 🔧 プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import re
import pandas as pd
from typing import List, Dict

# プロジェクト内のモジュールをインポート
from services.qdrant_service import (
    create_or_recreate_collection_for_qdrant,
    embed_texts_for_qdrant,
    upsert_points_to_qdrant,
    build_points_for_qdrant
)
from qdrant_client_wrapper import create_qdrant_client
from config import QdrantConfig

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 🔧 プロジェクトルートの絶対パスを取得
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalize_source_filename(filename: str) -> str:
    """
    ファイル名から日時サフィックス（例: _20251230_232641）を除去して正規化する。
    UI(agent_rag.py)での参照を安定させるための処理。
    """
    normalized = re.sub(r'_\d{8}_\d{6}', '', filename)
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="CSVデータをQdrantに登録・インデックス化するツール"
    )

    # 必須引数
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="登録するCSVファイルのパス (例: qa_output/pipeline/qa_pairs_....csv)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="登録先のQdrantコレクション名 (例: qa_fineweb_edu_ja)"
    )

    # オプション引数
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="既存の同名コレクションがある場合、削除して作り直す"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="1回のEmbedding API呼び出し/登録処理で扱う件数 (デフォルト: 50)"
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="ベクトル化対象のカラム名。指定がない場合は 'question' + 'answer' を結合して使用"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="ペイロードの 'domain' フィールドに設定する値 (デフォルト: コレクション名)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="登録する最大ドキュメント数 (テスト用)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "openai"],
        help="Embeddingに使用するプロバイダー (デフォルト: gemini)"
    )

    args = parser.parse_args()

    # APIキー確認 (Geminiの場合)
    if args.provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY環境変数が設定されていません。")
        sys.exit(1)

    # 1. データ読み込み
    if not os.path.exists(args.input_file):
        logger.error(f"入力ファイルが見つかりません: {args.input_file}")
        sys.exit(1)

    logger.info(f"📁 ファイル読み込み中: {args.input_file}")
    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"   -> 読み込み完了: {len(df)} 行")
    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {e}")
        sys.exit(1)

    # 件数制限
    if args.max_docs and len(df) > args.max_docs:
        df = df.head(args.max_docs)
        logger.info(f"   -> {args.max_docs} 件に制限しました")

    # 2. ベクトル化対象テキストの準備
    texts: List[str] = []

    if args.text_col:
        if args.text_col not in df.columns:
            logger.error(f"指定されたカラム '{args.text_col}' がCSVに含まれていません。")
            logger.info(f"存在するカラム: {list(df.columns)}")
            sys.exit(1)
        texts = df[args.text_col].astype(str).tolist()
        logger.info(f"📝 ベクトル化対象: カラム '{args.text_col}'")

    elif 'question' in df.columns and 'answer' in df.columns:
        texts = (df['question'].astype(str) + "\n" + df['answer'].astype(str)).tolist()
        logger.info("📝 ベクトル化対象: 'question' と 'answer' を結合")

    elif 'Combined_Text' in df.columns:
        texts = df['Combined_Text'].astype(str).tolist()
        logger.info("📝 ベクトル化対象: 'Combined_Text' (自動検出)")

    else:
        logger.error("ベクトル化対象のカラムを特定できませんでした。--text-col で指定してください。")
        sys.exit(1)

    # 3. Qdrantクライアント初期化 & コレクション準備
    try:
        client = create_qdrant_client()
        collections = client.get_collections()

        if args.recreate:
            logger.info(f"🗑️ コレクション '{args.collection}' を再作成します...")
            create_or_recreate_collection_for_qdrant(client, args.collection, recreate=True)
        else:
            exists = False
            for c in collections.collections:
                if c.name == args.collection:
                    exists = True
                    break

            if not exists:
                logger.info(f"🆕 コレクション '{args.collection}' を新規作成します...")
                create_or_recreate_collection_for_qdrant(client, args.collection, recreate=False)
            else:
                logger.info(f"ℹ️ 既存のコレクション '{args.collection}' に追記します")

    except Exception as e:
        logger.error(f"Qdrant接続エラー: {e}")
        logger.error("Dockerコンテナが起動しているか確認してください (docker-compose up -d)")
        sys.exit(1)

    # 4. バッチ処理によるEmbedding生成と登録
    total_processed = 0
    domain_val = args.domain if args.domain else args.collection

    logger.info(f"🚀 登録処理開始 (全 {len(df)} 件, バッチサイズ: {args.batch_size})")

    try:
        for i in range(0, len(df), args.batch_size):
            end_idx = min(i + args.batch_size, len(df))
            batch_df = df.iloc[i: end_idx]
            batch_texts = texts[i: end_idx]

            # A. ベクトル化
            vectors = embed_texts_for_qdrant(batch_texts)

            if not vectors:
                logger.warning(f"   Batch {i}-{end_idx}: ベクトル生成に失敗しました（スキップ）")
                continue

            # B. ポイント構築
            source_filename = os.path.basename(args.input_file)
            normalized_filename = normalize_source_filename(source_filename)

            points = build_points_for_qdrant(
                batch_df,
                vectors,
                domain=domain_val,
                source_file=normalized_filename
            )

            # Embeddingメタデータの追加
            for point in points:
                if "source" not in point.payload:
                    point.payload["source"] = normalized_filename

                point.payload["embedding_provider"] = args.provider
                if args.provider == "gemini":
                    point.payload["embedding_model"] = "gemini-embedding-001"
                elif args.provider == "openai":
                    point.payload["embedding_model"] = "text-embedding-3-small"

            # C. Qdrantへアップサート
            upsert_points_to_qdrant(client, args.collection, points)

            total_processed += len(points)
            logger.info(f"   ✅ 進捗: {total_processed} / {len(df)} 件完了")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ 処理が中断されました。")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info(f"\n🎉 完了！ 合計 {total_processed} 件のデータをコレクション '{args.collection}' に登録しました。")
    logger.info(f"   使用プロバイダー: {args.provider}")

    # =================================================================
    # 🔧 UI用正規化CSVの作成（絶対パスを使用）
    # =================================================================
    try:
        source_filename = os.path.basename(args.input_file)
        normalized_filename = normalize_source_filename(source_filename)

        # 🔧 プロジェクトルートからの絶対パス
        output_dir = os.path.join(PROJECT_ROOT, "qa_output")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, normalized_filename)

        logger.info(f"📋 UI用ファイル作成: {output_path}")

        columns_to_keep = ['question', 'answer']
        available_columns = [col for col in columns_to_keep if col in df.columns]

        if available_columns:
            df_ui = df[available_columns].copy()
            df_ui.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"   -> 作成完了 ({len(df_ui)}行)")
        else:
            logger.warning(f"   -> 必要なカラム({columns_to_keep})が見つからないためスキップ")

    except Exception as e:
        logger.error(f"UI用ファイル作成エラー: {e}")


if __name__ == "__main__":
    main()
