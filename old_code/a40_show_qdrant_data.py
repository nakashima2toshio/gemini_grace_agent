#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a40_show_qdrant_data.py - Qdrantデータ表示ツール
=============================================================
起動: streamlit run a40_show_qdrant_data.py --server.port=8502

【主要機能】
✅ Qdrantサーバーの接続状態チェック
✅ コレクション一覧の表示
✅ コレクション詳細情報の表示
✅ データソース情報の表示（qa_output/ディレクトリーとの対応）
✅ ポイントデータの表示とエクスポート（CSV, JSON）

【データソース機能】
各コレクションがqa_output/ディレクトリーのどのCSVファイルから
構成されているかを自動的に分析して表示します：
- ソースファイル名
- 推定データ件数と割合
- 生成方法（a02_make_qa, a03_coverage, a10_hybridなど）
- ドメイン情報
"""

import streamlit as st
import pandas as pd
import time
import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import socket

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrantクライアントのインポート
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant client not available. Install with: pip install qdrant-client")

# ===================================================================
# サーバー設定
# ===================================================================
QDRANT_CONFIG = {
    "name": "Qdrant",
    "host": "localhost",
    "port": 6333,
    "icon": "🎯",
    "url": "http://localhost:6333",
    "health_check_endpoint": "/collections",
    "docker_image": "qdrant/qdrant"
}

# ===================================================================
# Qdrant接続チェッククラス
# ===================================================================
class QdrantHealthChecker:
    """Qdrantサーバーの接続状態をチェック"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.client = None
        
    def check_port(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """ポートが開いているかチェック"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            if self.debug_mode:
                logger.error(f"Port check failed for {host}:{port}: {e}")
            return False
    
    def check_qdrant(self) -> Tuple[bool, str, Optional[Dict]]:
        """Qdrant接続チェック"""
        start_time = time.time()
        
        # まずポートチェック
        if not self.check_port(QDRANT_CONFIG["host"], QDRANT_CONFIG["port"]):
            return False, "Connection refused (port closed)", None
        
        if not QDRANT_AVAILABLE:
            return False, "Qdrant client not installed", None
        
        try:
            self.client = QdrantClient(url=QDRANT_CONFIG["url"], timeout=5)
            
            # コレクション取得
            collections = self.client.get_collections()
            
            metrics = {
                "collection_count": len(collections.collections),
                "collections": [c.name for c in collections.collections],
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            return True, "Connected", metrics
            
        except Exception as e:
            error_msg = str(e)
            if self.debug_mode:
                error_msg = f"{error_msg}\n{traceback.format_exc()}"
            return False, error_msg, None

# ===================================================================
# データ取得クラス
# ===================================================================
class QdrantDataFetcher:
    """Qdrantからデータを取得"""
    
    def __init__(self, client: QdrantClient):
        self.client = client
    
    def fetch_collections(self) -> pd.DataFrame:
        """コレクション一覧を取得"""
        try:
            collections = self.client.get_collections()
            
            data = []
            for collection in collections.collections:
                try:
                    info = self.client.get_collection(collection.name)
                    data.append({
                        "Collection": collection.name,
                        "Vectors Count": info.vectors_count,
                        "Points Count": info.points_count,
                        "Indexed Vectors": info.indexed_vectors_count,
                        "Status": info.status
                    })
                except Exception:
                    data.append({
                        "Collection": collection.name,
                        "Vectors Count": "N/A",
                        "Points Count": "N/A",
                        "Indexed Vectors": "N/A",
                        "Status": "Error"
                    })
            
            return pd.DataFrame(data) if data else pd.DataFrame({"Info": ["No collections found"]})
            
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})
    
    def fetch_collection_points(self, collection_name: str, limit: int = 50) -> pd.DataFrame:
        """コレクションの詳細データを取得"""
        try:
            # スクロールを使ってポイントを取得
            points_result = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            points = points_result[0]  # scrollは (points, next_offset) のタプルを返す
            
            if not points:
                return pd.DataFrame({"Info": ["No points found in collection"]})
            
            # ポイントをDataFrameに変換
            data = []
            for point in points:
                row = {"ID": point.id}
                
                # payloadの各フィールドを列として追加
                if point.payload:
                    for key, value in point.payload.items():
                        # 長すぎる文字列は切り詰め
                        if isinstance(value, str) and len(value) > 200:
                            row[key] = value[:200] + '...'
                        elif isinstance(value, (list, dict)):
                            row[key] = str(value)[:200] + '...' if len(str(value)) > 200 else str(value)
                        else:
                            row[key] = value
                
                data.append(row)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})
    
    def fetch_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """コレクションの詳細情報を取得"""
        try:
            collection_info = self.client.get_collection(collection_name)

            # configの構造を安全にアクセス
            vector_config = collection_info.config.params.vectors

            # vector_configの型を判定して適切に処理
            if hasattr(vector_config, 'size'):
                # 単一ベクトル設定
                vector_size = vector_config.size
                distance = vector_config.distance
            elif hasattr(vector_config, '__iter__'):
                # Named vectors設定の場合
                vector_sizes = {}
                distances = {}
                for name, config in vector_config.items() if isinstance(vector_config, dict) else []:
                    vector_sizes[name] = config.size if hasattr(config, 'size') else 'N/A'
                    distances[name] = config.distance if hasattr(config, 'distance') else 'N/A'
                vector_size = vector_sizes if vector_sizes else 'N/A'
                distance = distances if distances else 'N/A'
            else:
                vector_size = 'N/A'
                distance = 'N/A'

            return {
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "indexed_vectors": collection_info.indexed_vectors_count,
                "status": collection_info.status,
                "config": {
                    "vector_size": vector_size,
                    "distance": distance,
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def fetch_collection_source_info(self, collection_name: str, sample_size: int = 200) -> Dict[str, Any]:
        """コレクションのデータソース情報を取得"""
        try:
            collection_info = self.client.get_collection(collection_name)
            total_points = collection_info.points_count

            # サンプルポイントを取得
            points_result = self.client.scroll(
                collection_name=collection_name,
                limit=min(sample_size, total_points),
                with_payload=True,
                with_vectors=False
            )

            points = points_result[0]

            if not points:
                return {
                    "total_points": total_points,
                    "sources": {},
                    "sample_size": 0
                }

            # sourceとgeneration_methodを集計
            source_stats = {}
            for point in points:
                if point.payload:
                    source = point.payload.get('source', 'unknown')
                    method = point.payload.get('generation_method', 'unknown')
                    domain = point.payload.get('domain', 'unknown')

                    if source not in source_stats:
                        source_stats[source] = {
                            'sample_count': 0,
                            'method': method,
                            'domain': domain
                        }
                    source_stats[source]['sample_count'] += 1

            # 全体のデータ数を推定
            sample_total = len(points)
            for source, stats in source_stats.items():
                ratio = stats['sample_count'] / sample_total
                stats['estimated_total'] = int(total_points * ratio)
                stats['percentage'] = ratio * 100

            return {
                "total_points": total_points,
                "sources": source_stats,
                "sample_size": sample_total
            }

        except Exception as e:
            return {"error": str(e)}

# ===================================================================
# ヘルパー関数
# ===================================================================
def display_source_info(source_info: Dict[str, Any]) -> None:
    """データソース情報を表示"""
    if "error" in source_info:
        st.error(f"ソース情報取得エラー: {source_info['error']}")
        return

    total_points = source_info.get("total_points", 0)
    sources = source_info.get("sources", {})
    sample_size = source_info.get("sample_size", 0)

    if not sources:
        st.info("📂 データソース情報が見つかりません")
        return

    # ソース情報を表示
    st.markdown("### 📂 データソース構成 (qa_output/ディレクトリー)")
    st.caption(f"サンプル{sample_size}件から推定 | 総ポイント数: {total_points:,}件")

    # テーブル形式で表示
    source_data = []
    for source, stats in sorted(sources.items()):
        source_data.append({
            "ソースファイル": source,
            "推定件数": f"{stats['estimated_total']:,}件",
            "割合": f"{stats['percentage']:.1f}%",
            "生成方法": stats['method'],
            "ドメイン": stats['domain']
        })

    df_sources = pd.DataFrame(source_data)
    st.dataframe(df_sources, width='stretch', hide_index=True)

    # 詳細情報（折りたたみ可能）
    with st.expander("📊 詳細情報", expanded=False):
        for source, stats in sorted(sources.items()):
            st.markdown(f"**{source}**")
            st.markdown(f"- パス: `qa_output/{source}`")
            st.markdown(f"- 推定データ数: {stats['estimated_total']:,}件 ({stats['percentage']:.1f}%)")
            st.markdown(f"- 生成方法: `{stats['method']}`")
            st.markdown(f"- ドメイン: `{stats['domain']}`")
            st.markdown(f"- サンプル内カウント: {stats['sample_count']}件")
            st.divider()

# ===================================================================
# Streamlit UI
# ===================================================================
def main():
    st.set_page_config(
        page_title="Qdrant Monitor",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # セッションステート初期化
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 30
    
    # タイトル
    st.title("🎯 Qdrant データ表示ツール")
    st.markdown("Qdrant Vector Database の状態監視とデータ表示")
    
    # サイドバー（左ペイン）
    with st.sidebar:
        st.header("⚙️ Qdrant接続状態")
        
        # デバッグモード切り替え
        debug_mode = st.checkbox("🐛 デバッグモード", value=st.session_state.debug_mode)
        st.session_state.debug_mode = debug_mode
        
        # 自動リフレッシュ設定
        col1, col2 = st.columns(2)
        with col1:
            auto_refresh = st.checkbox("🔄 自動更新", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh
        with col2:
            if auto_refresh:
                refresh_interval = st.number_input("間隔(秒)", min_value=5, max_value=300, value=30)
                st.session_state.refresh_interval = refresh_interval
        
        # 接続チェック実行ボタン
        check_button = st.button("🔍 接続チェック実行", type="primary", width='stretch')
        
        # HealthCheckerインスタンス
        checker = QdrantHealthChecker(debug_mode=debug_mode)
        
        # 接続状態表示エリア
        status_container = st.container()
        
        # 自動リフレッシュまたはボタン押下時に実行
        if check_button or (auto_refresh and time.time() % refresh_interval < 1):
            with status_container:
                with st.spinner("チェック中..."):
                    is_connected, message, metrics = checker.check_qdrant()
                
                # Qdrantの状態表示
                if is_connected:
                    st.success(f"{QDRANT_CONFIG['icon']} **{QDRANT_CONFIG['name']}**")
                    st.caption(f"✅ {message}")
                    
                    # メトリクス表示
                    if metrics and debug_mode:
                        with st.expander("詳細情報", expanded=False):
                            for key, value in metrics.items():
                                st.text(f"{key}: {value}")
                else:
                    st.error(f"{QDRANT_CONFIG['icon']} **{QDRANT_CONFIG['name']}**")
                    st.caption(f"❌ {message}")
                    
                    # エラー詳細（デバッグモード）
                    if debug_mode:
                        with st.expander("エラー詳細", expanded=False):
                            st.code(message)
                            st.caption(f"Host: {QDRANT_CONFIG.get('host')}:{QDRANT_CONFIG.get('port')}")
                            
                            # Docker起動コマンド表示
                            st.info("Docker起動コマンド:")
                            cmd = f"docker run -d -p {QDRANT_CONFIG['port']}:{QDRANT_CONFIG['port']} {QDRANT_CONFIG['docker_image']}"
                            st.code(cmd, language="bash")
    
    # メインエリア（右ペイン）
    st.header("📊 Qdrant データ表示")
    
    # Qdrantが利用可能かチェック
    if not QDRANT_AVAILABLE:
        st.warning("Qdrant clientがインストールされていません。")
        st.code("pip install qdrant-client", language="bash")
        return
    
    try:
        # Qdrantクライアントを作成
        client = QdrantClient(url=QDRANT_CONFIG["url"], timeout=5)
        data_fetcher = QdrantDataFetcher(client)
        
        # コレクション概要表示
        st.subheader("📚 コレクション一覧")
        
        # コレクション一覧を取得
        df_collections = data_fetcher.fetch_collections()
        
        if not df_collections.empty and "Collection" in df_collections.columns:
            st.dataframe(df_collections, width='stretch')

            # コレクション名のリストを作成
            collection_names = df_collections["Collection"].tolist()

            # ===== データソース情報の表示（メインエリア先頭） =====
            st.divider()
            st.subheader("📂 コレクションのデータソース情報")
            st.caption("各コレクションがqa_output/ディレクトリーのどのファイルから構成されているかを表示します")

            # 各コレクションのソース情報を表示
            for collection_name in collection_names:
                with st.expander(f"📦 {collection_name}", expanded=(collection_name == "qa_corpus")):
                    with st.spinner(f"{collection_name} のソース情報を取得中..."):
                        source_info = data_fetcher.fetch_collection_source_info(collection_name)
                        display_source_info(source_info)
            
            # エクスポート機能
            col1, col2 = st.columns(2)
            with col1:
                csv = df_collections.to_csv(index=False)
                st.download_button(
                    label="📥 CSVダウンロード",
                    data=csv,
                    file_name=f"qdrant_collections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            with col2:
                json_str = df_collections.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 JSONダウンロード",
                    data=json_str,
                    file_name=f"qdrant_collections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # コレクション詳細表示
            st.divider()
            st.subheader("🔍 コレクション詳細データ")
            
            if collection_names:
                selected_collection = st.selectbox(
                    "詳細を表示するコレクションを選択",
                    options=collection_names,
                    key="selected_collection"
                )
                
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    limit = st.number_input("表示件数", min_value=1, max_value=500, value=50, key="qdrant_limit")
                with col2:
                    show_details = st.button("📊 詳細情報を表示", key="show_collection_details")
                with col3:
                    fetch_points = st.button("🔍 ポイントデータを取得", key="fetch_collection_points")
                
                # コレクション詳細情報の表示
                if show_details:
                    with st.spinner(f"{selected_collection} の詳細情報を取得中..."):
                        info = data_fetcher.fetch_collection_info(selected_collection)
                        
                        if "error" not in info:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("ベクトル数", info["vectors_count"])
                            with col2:
                                st.metric("ポイント数", info["points_count"])
                            with col3:
                                st.metric("インデックス済み", info["indexed_vectors"])
                            with col4:
                                st.metric("ステータス", info["status"])
                            
                            # 設定情報
                            st.write("**ベクトル設定:**")
                            st.write(f"  • ベクトル次元: {info['config']['vector_size']}")
                            st.write(f"  • 距離計算: {info['config']['distance']}")
                        else:
                            st.error(f"エラー: {info['error']}")
                
                # ポイントデータの表示
                if fetch_points:
                    with st.spinner(f"{selected_collection} のポイントデータを取得中..."):
                        df_points = data_fetcher.fetch_collection_points(selected_collection, limit)
                        
                        if not df_points.empty and "ID" in df_points.columns:
                            st.write(f"**{selected_collection} のデータサンプル ({len(df_points)} 件):**")
                            st.dataframe(df_points, width='stretch')
                            
                            # エクスポート機能
                            col1, col2 = st.columns(2)
                            with col1:
                                csv = df_points.to_csv(index=False)
                                st.download_button(
                                    label="📥 ポイントデータ CSVダウンロード",
                                    data=csv,
                                    file_name=f"{selected_collection}_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            with col2:
                                json_str = df_points.to_json(orient="records", indent=2)
                                st.download_button(
                                    label="📥 ポイントデータ JSONダウンロード",
                                    data=json_str,
                                    file_name=f"{selected_collection}_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                        elif "Info" in df_points.columns:
                            st.info(df_points.iloc[0]["Info"])
                        elif "Error" in df_points.columns:
                            st.error(f"エラー: {df_points.iloc[0]['Error']}")
                        else:
                            st.info("ポイントデータが見つかりません")
        elif "Info" in df_collections.columns:
            st.info(df_collections.iloc[0]["Info"])
        elif "Error" in df_collections.columns:
            error_msg = df_collections.iloc[0]['Error']
            
            # より詳細なエラーメッセージを表示
            if "Connection refused" in error_msg or "[Errno 61]" in error_msg:
                st.error("❌ Qdrantサーバーに接続できません")
                st.warning(
                    "**原因:** Qdrantサーバーが起動していない可能性があります。\n\n"
                    "**解決方法:**\n\n"
                    "### 🚀 方法1: 自動セットアップ（推奨）\n"
                    "```bash\n"
                    "# セットアップスクリプトでQdrantを起動\n"
                    "python setup.py\n"
                    "# または\n"
                    "python server.py\n"
                    "```\n\n"
                    "### 🐳 方法2: 手動でDocker起動\n"
                    "**ステップ1: Docker Desktopを起動**\n"
                    "- macOS: アプリケーションフォルダからDocker Desktopを起動\n"
                    "- 確認: `docker version`\n\n"
                    "**ステップ2: Qdrantを起動**\n"
                    "```bash\n"
                    "cd docker-compose\n"
                    "docker-compose up -d qdrant\n"
                    "```\n\n"
                    "**ステップ3: 動作確認**\n"
                    "```bash\n"
                    "docker-compose ps\n"
                    "# Qdrantが'Up'状態であることを確認\n"
                    "```\n\n"
                    "### 🔧 トラブルシューティング:\n"
                    "- ポート使用中: `lsof -i :6333`\n"
                    "- ログ確認: `docker-compose logs qdrant`\n"
                    "- 再起動: `docker-compose restart qdrant`"
                )
                if st.session_state.debug_mode:
                    with st.expander("🔍 詳細エラー情報", expanded=False):
                        st.error(f"詳細エラー: {error_msg}")
                        st.caption(f"接続先: {QDRANT_CONFIG['host']}:{QDRANT_CONFIG['port']}")
                        st.info("docker-compose.ymlの場所: `docker-compose/docker-compose.yml`")
            elif "timeout" in error_msg.lower():
                st.error("⏱️ Qdrantサーバーへの接続がタイムアウトしました")
                st.warning(
                    "**原因:** サーバーが応答していないか、ネットワークの問題があります。\n\n"
                    "**解決方法:**\n"
                    "• Qdrantサーバーのログを確認してください\n"
                    "• ファイアウォール設定を確認してください\n"
                    "• ポート6333が使用可能か確認してください"
                )
            else:
                st.error(f"エラー: {error_msg}")
                st.info("Qdrantサーバーが正しく起動していることを確認してください")
        else:
            st.info("コレクションが見つかりません")
            
    except Exception as e:
        error_msg = str(e)
        
        # より詳細なエラーメッセージを表示
        if "Connection refused" in error_msg or "[Errno 61]" in error_msg:
            st.error("❌ Qdrantサーバーに接続できません")
            st.warning(
                "**原因:** Qdrantサーバーが起動していない可能性があります。\n\n"
                "**解決方法:**\n\n"
                "### 🚀 方法1: 自動セットアップ（推奨）\n"
                "```bash\n"
                "# セットアップスクリプトでQdrantを起動\n"
                "python setup.py\n"
                "# または\n"
                "python server.py\n"
                "```\n\n"
                "### 🐳 方法2: 手動でDocker起動\n"
                "**ステップ1: Docker Desktopを起動**\n"
                "- macOS: アプリケーションフォルダからDocker Desktopを起動\n"
                "- 確認: `docker version`\n\n"
                "**ステップ2: Qdrantを起動**\n"
                "```bash\n"
                "cd docker-compose\n"
                "docker-compose up -d qdrant\n"
                "```\n\n"
                "**ステップ3: 動作確認**\n"
                "```bash\n"
                "docker-compose ps\n"
                "# Qdrantが'Up'状態であることを確認\n"
                "```\n\n"
                "### 🔧 トラブルシューティング:\n"
                "- ポート使用中: `lsof -i :6333`\n"
                "- ログ確認: `docker-compose logs qdrant`\n"
                "- 再起動: `docker-compose restart qdrant`"
            )
            if debug_mode:
                with st.expander("🔍 詳細エラー情報", expanded=False):
                    st.error(f"詳細エラー: {error_msg}")
                    st.info("docker-compose.ymlの場所: `docker-compose/docker-compose.yml`")
        elif "timeout" in error_msg.lower():
            st.error("⏱️ Qdrantサーバーへの接続がタイムアウトしました")
            st.warning(
                "**原因:** サーバーが応答していないか、ネットワークの問題があります。\n\n"
                "**解決方法:**\n"
                "• Qdrantサーバーのログを確認してください\n"
                "• ファイアウォール設定を確認してください\n"
                "• ポート6333が使用可能か確認してください"
            )
        else:
            st.error(f"Qdrant接続エラー: {error_msg}")
            st.info("Qdrantサーバーが正しく起動していることを確認してください")
    
    # フッター
    st.divider()
    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # デバッグ情報表示
    if debug_mode:
        with st.expander("🐛 デバッグ情報", expanded=False):
            st.subheader("インストール済みクライアント")
            st.write({
                "Qdrant": "✅" if QDRANT_AVAILABLE else "❌"
            })
            
            st.subheader("サーバー設定")
            st.json(QDRANT_CONFIG)

if __name__ == "__main__":
    main()