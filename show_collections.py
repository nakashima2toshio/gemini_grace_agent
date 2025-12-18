#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
登録されたQdrantコレクションの一覧を表示
"""
from qdrant_client import QdrantClient

def main():
    # Qdrantに接続
    client = QdrantClient(url="http://localhost:6333")

    # コレクション一覧を取得
    collections = client.get_collections()

    print("登録されたコレクション一覧:")
    print("-" * 80)
    for col in collections.collections:
        try:
            info = client.get_collection(col.name)
            print(f"  • {col.name} ({info.points_count:,}件)")
        except Exception as e:
            print(f"  • {col.name} (エラー: {e})")
    print("-" * 80)
    print(f"合計: {len(collections.collections)}個のコレクション")

if __name__ == "__main__":
    main()
