#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a41_qdrant_truncate.py — Qdrantコレクションのデータ削除ツール
--------------------------------------------------------------------------------
Qdrantに登録されたRAGデータを安全に削除するためのユーティリティ。
コレクション全体の削除、特定ドメインのみの削除、統計情報の表示などをサポート。

使い方：
# 実行コマンド

  # 全コレクションを削除（危険！） ---------------------------------
  python a41_qdrant_truncate.py --all-collections --force
  # -----------------------------------------------------------

  python a41_qdrant_truncate.py --collection product_embeddings --drop-collection --force

  # 統計情報を表示（削除なし）
  python a41_qdrant_truncate.py --stats
  
  # ドライラン（削除対象を表示するが実行しない）
  python a41_qdrant_truncate.py --dry-run
  
  # 特定ドメインのデータを削除
  python a41_qdrant_truncate.py --domain medical --force
  
  # 全データを削除（コレクションは残す）
  python a41_qdrant_truncate.py --all --force
  
  # コレクション自体を削除
  python a41_qdrant_truncate.py --drop-collection --force


主要引数：
  --collection         : コレクション名（既定: config.yml または 'qa_corpus'）
  --qdrant-url        : Qdrant URL（既定: http://localhost:6333）
  --domain            : 削除対象ドメイン（customer/medical/legal/sciq/trivia/unified/cc_news_llm/cc_news_coverage/cc_news_hybrid）
  --all               : 全データを削除（コレクションは保持）
  --all-collections   : 全コレクションを削除（危険！）
  --drop-collection   : コレクション自体を削除
  --stats             : 統計情報のみ表示
  --dry-run           : 削除対象を表示するが実行しない
  --force             : 確認プロンプトをスキップ
  --exclude           : 削除から除外するコレクション（--all-collections使用時）
  --batch-size        : 削除バッチサイズ（既定: 100）
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Any

try:
    import yaml
except ImportError:
    yaml = None

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

# カラー出力用のANSIコード
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text: str, color: str = Colors.ENDC):
    """カラー付きでテキストを出力"""
    print(f"{color}{text}{Colors.ENDC}")

def print_header(text: str):
    """ヘッダーを装飾付きで出力"""
    print()
    print_colored("=" * 80, Colors.HEADER)
    print_colored(f" {text}", Colors.HEADER + Colors.BOLD)
    print_colored("=" * 80, Colors.HEADER)
    print()

# デフォルト設定（config.ymlがない場合のフォールバック）
DEFAULTS = {
    "rag": {
        "collection": "qa_corpus",
    },
    "qdrant": {
        "url": "http://localhost:6333"
    }
}

# サポートされるドメインリスト
SUPPORTED_DOMAINS = [
    "customer",
    "medical",
    "legal",
    "sciq",
    "trivia",
    "unified",
    "cc_news_llm",      # CC News LLM生成方式
    "cc_news_coverage",  # CC Newsカバレッジ改良方式
    "cc_news_hybrid"     # CC Newsハイブリッド生成方式
]

def load_config(path: str = "config.yml") -> Dict[str, Any]:
    """設定ファイルを読み込み"""
    cfg = {}
    if yaml and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print_colored(f"⚠️  config.yml読み込みエラー: {e}", Colors.WARNING)
    
    # デフォルト値とマージ
    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst.setdefault(k, v)
    
    full = {}
    merge(full, DEFAULTS)
    merge(full, cfg)
    return full

def get_collection_stats(client: QdrantClient, collection_name: str) -> Optional[Dict[str, Any]]:
    """コレクションの統計情報を取得"""
    try:
        # コレクション情報を取得
        collection_info = client.get_collection(collection_name)
        
        # 全ポイント数を取得
        total_points = collection_info.points_count
        
        # ドメイン別の統計を取得
        domain_stats = {}
        for domain in SUPPORTED_DOMAINS:
            try:
                # ドメインフィルタを使用してカウント
                result = client.count(
                    collection_name=collection_name,
                    count_filter=models.Filter(
                        must=[models.FieldCondition(
                            key="domain",
                            match=models.MatchValue(value=domain)
                        )]
                    )
                )
                if result.count > 0:
                    domain_stats[domain] = result.count
            except Exception:
                # ドメインが存在しない場合は無視
                pass
        
        # ベクトル設定情報を取得
        vectors_config = collection_info.config.params.vectors
        vector_info = {}
        
        if isinstance(vectors_config, dict):
            # Named Vectors
            for name, config in vectors_config.items():
                vector_info[name] = {
                    "size": config.size,
                    "distance": str(config.distance)
                }
        elif hasattr(vectors_config, 'size'):
            # Single Vector
            vector_info["default"] = {
                "size": vectors_config.size,
                "distance": str(vectors_config.distance)
            }
        
        return {
            "total_points": total_points,
            "domain_stats": domain_stats,
            "vector_config": vector_info,
            "status": collection_info.status
        }
    
    except UnexpectedResponse as e:
        if "doesn't exist" in str(e) or "not found" in str(e).lower():
            return None
        raise
    except Exception as e:
        print_colored(f"❌ 統計情報取得エラー: {e}", Colors.FAIL)
        return None

def display_stats(stats: Dict[str, Any], collection_name: str):
    """統計情報を表示"""
    print_header(f"📊 コレクション '{collection_name}' の統計情報")
    
    print(f"{'総ポイント数:':<20} {Colors.BOLD}{stats['total_points']:,}{Colors.ENDC}")
    print(f"{'ステータス:':<20} {stats['status']}")
    print()
    
    if stats['domain_stats']:
        print_colored("ドメイン別データ数:", Colors.OKBLUE)
        print("-" * 40)
        for domain, count in sorted(stats['domain_stats'].items()):
            bar_length = int(count / max(stats['domain_stats'].values()) * 30)
            bar = "█" * bar_length
            print(f"  {domain:<15} {count:>7,} {Colors.OKCYAN}{bar}{Colors.ENDC}")
        print("-" * 40)
    
    if stats['vector_config']:
        print()
        print_colored("ベクトル設定:", Colors.OKBLUE)
        for name, config in stats['vector_config'].items():
            print(f"  {name}: size={config['size']}, distance={config['distance']}")

def confirm_action(message: str) -> bool:
    """ユーザーに確認を求める"""
    print()
    print_colored(f"⚠️  {message}", Colors.WARNING + Colors.BOLD)
    print_colored("この操作は取り消せません！", Colors.WARNING)
    print()
    
    while True:
        response = input(f"{Colors.BOLD}実行しますか？ (yes/no): {Colors.ENDC}").lower().strip()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("'yes' または 'no' を入力してください。")

def delete_by_domain(client: QdrantClient, collection_name: str, domain: str, 
                    batch_size: int = 100, dry_run: bool = False) -> int:
    """特定ドメインのデータを削除"""
    # まず対象データをカウント
    count_result = client.count(
        collection_name=collection_name,
        count_filter=models.Filter(
            must=[models.FieldCondition(
                key="domain",
                match=models.MatchValue(value=domain)
            )]
        )
    )
    
    total_count = count_result.count
    
    if total_count == 0:
        print_colored(f"ドメイン '{domain}' のデータは存在しません。", Colors.WARNING)
        return 0
    
    print(f"削除対象: ドメイン '{domain}' のデータ {total_count:,} 件")
    
    if dry_run:
        print_colored("[DRY RUN] 実際の削除は実行されません。", Colors.OKCYAN)
        return total_count
    
    # バッチで削除
    deleted = 0
    while True:
        # 対象ポイントを検索
        search_result = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="domain",
                    match=models.MatchValue(value=domain)
                )]
            ),
            limit=batch_size,
            with_payload=False,
            with_vectors=False
        )
        
        if not search_result[0]:
            break
        
        # ポイントIDを取得
        point_ids = [point.id for point in search_result[0]]
        
        # 削除実行
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=point_ids
            )
        )
        
        deleted += len(point_ids)
        print(f"  削除進捗: {deleted:,} / {total_count:,} ({deleted*100/total_count:.1f}%)")
        
        if deleted >= total_count:
            break
    
    return deleted

def delete_all_data(client: QdrantClient, collection_name: str, 
                   batch_size: int = 100, dry_run: bool = False) -> int:
    """全データを削除（コレクションは保持）"""
    stats = get_collection_stats(client, collection_name)
    if not stats:
        print_colored(f"コレクション '{collection_name}' が存在しません。", Colors.WARNING)
        return 0
    
    total_count = stats['total_points']
    
    if total_count == 0:
        print_colored("削除するデータがありません。", Colors.WARNING)
        return 0
    
    print(f"削除対象: 全データ {total_count:,} 件")
    
    if dry_run:
        print_colored("[DRY RUN] 実際の削除は実行されません。", Colors.OKCYAN)
        return total_count
    
    # バッチで削除
    deleted = 0
    while True:
        # 全ポイントを検索
        search_result = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            with_payload=False,
            with_vectors=False
        )
        
        if not search_result[0]:
            break
        
        # ポイントIDを取得
        point_ids = [point.id for point in search_result[0]]
        
        # 削除実行
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=point_ids
            )
        )
        
        deleted += len(point_ids)
        print(f"  削除進捗: {deleted:,} / {total_count:,} ({deleted*100/total_count:.1f}%)")
        
        if deleted >= total_count:
            break
    
    return deleted

def drop_collection(client: QdrantClient, collection_name: str, dry_run: bool = False) -> bool:
    """コレクション自体を削除"""
    stats = get_collection_stats(client, collection_name)
    if not stats:
        print_colored(f"コレクション '{collection_name}' が存在しません。", Colors.WARNING)
        return False
    
    print(f"削除対象: コレクション '{collection_name}' （{stats['total_points']:,} ポイント）")
    
    if dry_run:
        print_colored("[DRY RUN] 実際の削除は実行されません。", Colors.OKCYAN)
        return True
    
    # コレクション削除
    result = client.delete_collection(collection_name=collection_name)
    return result

def get_all_collections(client: QdrantClient) -> List[Dict[str, Any]]:
    """全コレクションの情報を取得"""
    collections = client.get_collections()
    collection_list = []
    
    for collection in collections.collections:
        # 各コレクションの詳細情報を取得
        try:
            info = client.get_collection(collection.name)
            collection_list.append({
                "name": collection.name,
                "points_count": info.points_count,
                "status": info.status
            })
        except Exception:
            collection_list.append({
                "name": collection.name,
                "points_count": 0,
                "status": "unknown"
            })
    
    return collection_list

def display_all_collections_stats(collections: List[Dict[str, Any]]):
    """全コレクションの統計情報を表示"""
    print_header("📊 全コレクションの統計情報")
    
    total_collections = len(collections)
    total_points = sum(c["points_count"] for c in collections)
    
    print(f"{'総コレクション数:':<20} {Colors.BOLD}{total_collections}{Colors.ENDC}")
    print(f"{'総ポイント数:':<20} {Colors.BOLD}{total_points:,}{Colors.ENDC}")
    print()
    
    if collections:
        print_colored("コレクション一覧:", Colors.OKBLUE)
        print("-" * 60)
        print(f"  {'名前':<30} {'ポイント数':>10} {'ステータス':>10}")
        print("-" * 60)
        for col in sorted(collections, key=lambda x: x["points_count"], reverse=True):
            status_color = Colors.OKGREEN if col["status"] == "green" else Colors.WARNING
            print(f"  {col['name']:<30} {col['points_count']:>10,} {status_color}{col['status']:>10}{Colors.ENDC}")
        print("-" * 60)

def confirm_all_collections_deletion(collections: List[Dict[str, Any]], excluded: List[str]) -> bool:
    """全コレクション削除の強い確認"""
    # 削除対象を計算
    to_delete = [c for c in collections if c["name"] not in excluded]
    
    if not to_delete:
        print_colored("削除対象のコレクションがありません。", Colors.WARNING)
        return False
    
    print()
    print_colored("━" * 80, Colors.FAIL)
    print_colored("⚠️  警告: 全コレクション削除", Colors.FAIL + Colors.BOLD)
    print_colored("━" * 80, Colors.FAIL)
    print()
    print_colored(f"削除予定: {len(to_delete)} コレクション", Colors.WARNING)
    
    for col in to_delete:
        print(f"  - {col['name']} ({col['points_count']:,} ポイント)")
    
    if excluded:
        print()
        print_colored(f"除外: {len(excluded)} コレクション", Colors.OKGREEN)
        for name in excluded:
            print(f"  - {name}")
    
    print()
    print_colored("この操作により、上記の全てのコレクションが完全に削除されます。", Colors.FAIL + Colors.BOLD)
    print_colored("この操作は取り消せません！！！", Colors.FAIL + Colors.BOLD)
    print()
    
    # 第一確認
    while True:
        response = input(f"{Colors.BOLD}本当に削除しますか？ (yes/no): {Colors.ENDC}").lower().strip()
        if response in ['yes', 'y']:
            break
        elif response in ['no', 'n']:
            return False
        else:
            print("'yes' または 'no' を入力してください。")
    
    # 第二確認（カウントダウン付き）
    print()
    print_colored("最終確認：3秒後に削除を開始します。中止するにはCtrl+Cを押してください。", Colors.WARNING + Colors.BOLD)
    
    for i in range(3, 0, -1):
        print(f"{Colors.BOLD}{i}...{Colors.ENDC}", end='', flush=True)
        time.sleep(1)
    print()
    
    return True

def delete_all_collections(client: QdrantClient, excluded: List[str] = None, dry_run: bool = False) -> int:
    """全コレクションを削除"""
    excluded = excluded or []
    
    # 全コレクション情報を取得
    collections = get_all_collections(client)
    
    if not collections:
        print_colored("削除するコレクションがありません。", Colors.WARNING)
        return 0
    
    # 削除対象を計算
    to_delete = [c for c in collections if c["name"] not in excluded]
    
    if not to_delete:
        print_colored("削除対象のコレクションがありません（全て除外されています）。", Colors.WARNING)
        return 0
    
    # 統計情報を表示
    display_all_collections_stats(collections)
    
    if dry_run:
        print()
        print_colored("[DRY RUN] 削除対象:", Colors.OKCYAN)
        for col in to_delete:
            print(f"  - {col['name']} ({col['points_count']:,} ポイント)")
        print_colored(f"合計 {len(to_delete)} コレクションが削除されます。", Colors.OKCYAN)
        print_colored("[DRY RUN] 実際の削除は実行されません。", Colors.OKCYAN)
        return len(to_delete)
    
    # 実際の削除
    deleted_count = 0
    failed_count = 0
    
    print()
    print_colored("削除を開始します...", Colors.WARNING)
    
    for col in to_delete:
        try:
            print(f"  削除中: {col['name']}... ", end='', flush=True)
            client.delete_collection(collection_name=col['name'])
            print_colored("✓", Colors.OKGREEN)
            deleted_count += 1
        except Exception as e:
            print_colored(f"✗ エラー: {e}", Colors.FAIL)
            failed_count += 1
    
    print()
    if deleted_count > 0:
        print_colored(f"✅ {deleted_count} コレクションを削除しました。", Colors.OKGREEN)
    if failed_count > 0:
        print_colored(f"❌ {failed_count} コレクションの削除に失敗しました。", Colors.FAIL)
    
    return deleted_count

def main():
    """メインエントリポイント"""
    # 設定読み込み
    cfg = load_config("config.yml")
    rag_cfg = cfg.get("rag", {})
    qdrant_cfg = cfg.get("qdrant", {})
    
    # 引数パーサー
    parser = argparse.ArgumentParser(
        description="Qdrantコレクションのデータを安全に削除",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 統計情報を表示
  python a41_qdrant_truncate.py --stats
  
  # 特定ドメインを削除（確認あり）
  python a41_qdrant_truncate.py --domain medical
  
  # 全データを削除（強制実行）
  python a41_qdrant_truncate.py --all --force
  
  # ドライラン（削除せずに確認）
  python a41_qdrant_truncate.py --all --dry-run
        """
    )
    
    parser.add_argument("--collection", 
                       default=rag_cfg.get("collection", "qa_corpus"),
                       help="対象コレクション名")
    parser.add_argument("--qdrant-url", 
                       default=qdrant_cfg.get("url", "http://localhost:6333"),
                       help="Qdrant URL")
    parser.add_argument("--domain",
                       choices=SUPPORTED_DOMAINS,
                       help="削除対象のドメイン")
    parser.add_argument("--all",
                       action="store_true",
                       help="全データを削除（コレクションは保持）")
    parser.add_argument("--all-collections",
                       action="store_true",
                       help="全コレクションを削除（危険！）")
    parser.add_argument("--drop-collection",
                       action="store_true",
                       help="コレクション自体を削除")
    parser.add_argument("--exclude",
                       action="append",
                       help="削除から除外するコレクション（--all-collections使用時、複数指定可）")
    parser.add_argument("--stats",
                       action="store_true",
                       help="統計情報のみ表示（削除なし）")
    parser.add_argument("--dry-run",
                       action="store_true",
                       help="削除対象を表示するが実行しない")
    parser.add_argument("--force",
                       action="store_true",
                       help="確認プロンプトをスキップ")
    parser.add_argument("--batch-size",
                       type=int,
                       default=100,
                       help="削除バッチサイズ")
    
    args = parser.parse_args()
    
    # 排他的オプションのチェック
    action_count = sum([
        bool(args.domain),
        args.all,
        args.all_collections,
        args.drop_collection,
        args.stats
    ])
    
    if action_count == 0:
        print_colored("❌ アクションを指定してください（--stats, --domain, --all, --all-collections, --drop-collection）", Colors.FAIL)
        parser.print_help()
        sys.exit(1)
    
    if action_count > 1:
        print_colored("❌ 複数のアクションを同時に指定することはできません", Colors.FAIL)
        sys.exit(1)
    
    # --exclude は --all-collections でのみ有効
    if args.exclude and not args.all_collections:
        print_colored("❌ --exclude は --all-collections と併用してください", Colors.FAIL)
        sys.exit(1)
    
    # Qdrantクライアント初期化
    try:
        client = QdrantClient(url=args.qdrant_url, timeout=30)
        # 接続テスト
        client.get_collections()
    except Exception as e:
        print_colored(f"❌ Qdrant接続エラー: {e}", Colors.FAIL)
        print_colored(f"URL: {args.qdrant_url}", Colors.FAIL)
        print_colored("Qdrantが起動していることを確認してください。", Colors.WARNING)
        sys.exit(1)
    
    # 統計情報表示
    if args.stats:
        if args.collection != rag_cfg.get("collection", "qa_corpus"):
            # 特定コレクションの統計
            stats = get_collection_stats(client, args.collection)
            if stats:
                display_stats(stats, args.collection)
            else:
                print_colored(f"❌ コレクション '{args.collection}' が存在しません。", Colors.FAIL)
        else:
            # 全コレクションの統計
            collections = get_all_collections(client)
            if collections:
                display_all_collections_stats(collections)
            else:
                print_colored("❌ コレクションが存在しません。", Colors.FAIL)
        return
    
    # アクション実行前の確認
    print_header("🗑️  Qdrantデータ削除ツール")
    
    # 現在の統計情報を表示
    stats = get_collection_stats(client, args.collection)
    if stats:
        display_stats(stats, args.collection)
        print()
    
    # 削除アクションの実行
    try:
        if args.domain:
            # ドメイン別削除
            if not args.force and not args.dry_run:
                if not confirm_action(f"ドメイン '{args.domain}' のデータを削除します"):
                    print_colored("削除をキャンセルしました。", Colors.OKGREEN)
                    return

            deleted = delete_by_domain(client, args.collection, args.domain,
                                      args.batch_size, args.dry_run)
            if not args.dry_run and deleted > 0:
                print_colored(f"✅ {deleted:,} 件のデータを削除しました。", Colors.OKGREEN)

        elif args.all:
            # 全データ削除
            if not args.force and not args.dry_run:
                if not confirm_action("全データを削除します"):
                    print_colored("削除をキャンセルしました。", Colors.OKGREEN)
                    return

            deleted = delete_all_data(client, args.collection,
                                    args.batch_size, args.dry_run)
            if not args.dry_run and deleted > 0:
                print_colored(f"✅ {deleted:,} 件のデータを削除しました。", Colors.OKGREEN)

        elif args.drop_collection:
            # 全コレクション一覧を表示
            all_collections = get_all_collections(client)
            if all_collections:
                display_all_collections_stats(all_collections)
                print()

            # 削除対象コレクションの確認
            if args.collection not in [c["name"] for c in all_collections]:
                print_colored(f"❌ コレクション '{args.collection}' が存在しません。", Colors.FAIL)
                return
            # コレクション削除
            if not args.force and not args.dry_run:
                if not confirm_action(f"コレクション '{args.collection}' を完全に削除します"):
                    print_colored("削除をキャンセルしました。", Colors.OKGREEN)
                    return
            
            success = drop_collection(client, args.collection, args.dry_run)
            if not args.dry_run and success:
                print_colored(f"✅ コレクション '{args.collection}' を削除しました。", Colors.OKGREEN)
        
        elif args.all_collections:
            # 全コレクション削除
            collections = get_all_collections(client)
            excluded = args.exclude or []
            
            if not args.dry_run and not args.force:
                if not confirm_all_collections_deletion(collections, excluded):
                    print_colored("削除をキャンセルしました。", Colors.OKGREEN)
                    return
            
            deleted = delete_all_collections(client, excluded, args.dry_run)
            if not args.dry_run and deleted > 0:
                print_colored(f"✅ {deleted} コレクションを削除しました。", Colors.OKGREEN)
        
        # 削除後の統計情報を表示（dry-runでない場合）
        if not args.dry_run:
            if args.all_collections:
                # 全コレクション削除後の確認
                print()
                print_colored("削除後の状態:", Colors.HEADER)
                collections = get_all_collections(client)
                if collections:
                    display_all_collections_stats(collections)
                else:
                    print_colored("✅ 全てのコレクションが削除されました。", Colors.OKGREEN)
            elif not args.drop_collection:
                # 通常の削除後確認
                print()
                print_colored("削除後の状態:", Colors.HEADER)
                stats = get_collection_stats(client, args.collection)
                if stats:
                    display_stats(stats, args.collection)
    
    except Exception as e:
        print_colored(f"❌ エラーが発生しました: {e}", Colors.FAIL)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()