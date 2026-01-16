#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
コレクション名調査スクリプト
実際に存在するQdrantコレクションを確認
"""

from qdrant_client import QdrantClient


def check_collections():
    """Qdrantに存在するコレクションを確認"""
    try:
        client = QdrantClient(url="http://localhost:6333")
        collections_response = client.get_collections()

        print("=" * 60)
        print("📊 Qdrantに存在するコレクション一覧")
        print("=" * 60)

        if collections_response.collections:
            for i, col in enumerate(collections_response.collections, 1):
                print(f"{i}. {col.name}")

                # コレクション詳細を取得
                try:
                    col_info = client.get_collection(col.name)
                    print(f"   - ポイント数: {col_info.points_count}")
                    print(f"   - ステータス: {col_info.status}")
                except Exception as e:
                    print(f"   - エラー: {e}")
                print()
        else:
            print("⚠️ コレクションが存在しません")

        print("=" * 60)

    except Exception as e:
        print(f"❌ エラー: {e}")
        print("Qdrantサーバーが起動していることを確認してください")


if __name__ == "__main__":
    check_collections()
