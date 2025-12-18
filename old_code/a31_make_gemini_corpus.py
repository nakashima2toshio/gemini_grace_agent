# gemini_corpus.py
# Gemini File API (Context Caching / RAG) 管理・検索アプリ
# streamlit run a31_make_gemini_corpus.py --server.port=8502

import streamlit as st
import pandas as pd
import os
import re
import time
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

# Google Generative AI SDK のインポート
try:
    import google.generativeai as genai
    from google.api_core import exceptions

    GEMINI_AVAILABLE = True
except ImportError as e:
    import sys
    GEMINI_AVAILABLE = False
    st.error(f"Google Generative AI SDK が見つかりません: {e}")
    st.error(f"現在のPython実行パス: {sys.executable}")
    st.info(f"検索パス: {sys.path}")
    st.error("この環境にパッケージがインストールされていない可能性があります。以下のコマンドを実行してください：")
    st.code(f"{sys.executable} -m pip install google-generativeai")
    st.stop()

# 共通機能のインポート
try:
    from helper_rag import (
        RAGConfig, TokenManager, safe_execute,
        select_model, show_model_info,
        setup_page_config, setup_page_header, setup_sidebar_header,
        create_output_directory
    )

    HELPER_AVAILABLE = True
except ImportError as e:
    HELPER_AVAILABLE = False
    logging.warning(f"ヘルパーモジュールのインポートに失敗: {e}")

# ===================================================================
# ログ設定
# ===================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================================================================
# 設定クラス
# ===================================================================
@dataclass
class FileConfig:
    """File設定データクラス"""
    dataset_type: str
    filename: str
    display_name: str
    description: str
    chunk_size: int = 1000
    overlap: int = 100
    max_file_size_mb: int = 400
    max_chunks_per_file: int = 40000
    csv_text_column: str = "Combined_Text"

    @classmethod
    def get_all_configs(cls) -> Dict[str, 'FileConfig']:
        """全データセット設定を取得"""
        return {
            "a02_wikipedia": cls(
                dataset_type="a02_wikipedia",
                filename="a02_qa_pairs_wikipedia_ja.csv",
                display_name="Wikipedia JA Q&A (a02)",
                description="Wikipedia日本語版 Q&A（a02生成）",
                csv_text_column="question" # question列とanswer列を結合して使う想定
            ),
            "a02_cc_news": cls(
                dataset_type="a02_cc_news",
                filename="a02_qa_pairs_cc_news.csv",
                display_name="CC News Q&A (a02)",
                description="CC NewsデータセットQ&A",
                csv_text_column="question"
            ),
            # 必要に応じて追加
        }


# ===================================================================
# ファイル処理クラス
# ===================================================================
class FileProcessor:
    """ファイル処理クラス"""

    def __init__(self):
        self.configs = FileConfig.get_all_configs()

    def load_csv_and_convert_to_txt(self, filepath: Path, text_columns: List[str] = ["question", "answer"]) -> str:
        """CSVを読み込み、テキスト形式（Markdown風）に変換して返す"""
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            
            text_content = ""
            for idx, row in df.iterrows():
                text_content += f"## Entry {idx}\n"
                for col in text_columns:
                    if col in df.columns:
                        val = str(row[col]).strip()
                        if val:
                            text_content += f"**{col}**: {val}\n"
                text_content += "\n---\n\n"
            
            return text_content

        except Exception as e:
            logger.error(f"CSV読み込みエラー: {filepath} - {e}")
            return ""


# ===================================================================
# Gemini File 管理クラス
# ===================================================================
class GeminiFileManager:
    """Gemini File API 管理クラス"""

    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("Google APIキーが設定されていません。環境変数 GOOGLE_API_KEY を確認してください。")

        genai.configure(api_key=api_key)
        self.processor = FileProcessor()
        self.configs = FileConfig.get_all_configs()

    def upload_file(self, content: str, display_name: str) -> Optional[Any]:
        """テキストコンテンツをGeminiにアップロード"""
        temp_file_path = None
        try:
            logger.info(f"ファイルアップロード開始: {display_name}")
            
            # 一時ファイル作成
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # アップロード
            gemini_file = genai.upload_file(
                path=temp_file_path,
                display_name=display_name,
                mime_type="text/plain"
            )
            
            logger.info(f"アップロード完了: {gemini_file.name}")
            
            # 処理完了待機
            while gemini_file.state.name == "PROCESSING":
                time.sleep(2)
                gemini_file = genai.get_file(gemini_file.name)
            
            if gemini_file.state.name == "FAILED":
                raise ValueError(f"ファイル処理失敗: {gemini_file.state.name}")
                
            return gemini_file

        except Exception as e:
            logger.error(f"アップロードエラー: {e}")
            return None
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def process_single_dataset(self, dataset_type: str, output_dir: Path = None) -> Dict[str, Any]:
        """単一データセットの処理"""
        if output_dir is None:
            output_dir = Path("qa_output")

        config = self.configs.get(dataset_type)
        if not config:
            return {"success": False, "error": f"未知のデータセットタイプ: {dataset_type}"}

        filepath = output_dir / config.filename
        
        # 簡易的なファイル検索（プレフィックス一致など）
        if not filepath.exists():
            # a02_qa_pairs_wikipedia_ja.csv のようなファイルを探す
            candidates = list(output_dir.glob(f"*{config.filename}*"))
            if candidates:
                filepath = candidates[0]
                logger.info(f"代替ファイルを使用: {filepath}")
            else:
                return {"success": False, "error": f"ファイルが見つかりません: {config.filename}"}

        try:
            # CSV読み込み & テキスト変換
            text_content = self.processor.load_csv_and_convert_to_txt(filepath)
            if not text_content:
                return {"success": False, "error": "有効なテキストコンテンツが生成できませんでした"}

            # アップロード
            gemini_file = self.upload_file(text_content, config.display_name)

            if gemini_file:
                return {
                    "success": True,
                    "file_name": gemini_file.name,
                    "display_name": gemini_file.display_name,
                    "uri": gemini_file.uri,
                    "size_bytes": gemini_file.size_bytes
                }
            else:
                return {"success": False, "error": "ファイルアップロードに失敗しました"}

        except Exception as e:
            logger.error(f"処理エラー: {e}")
            return {"success": False, "error": str(e)}

    def list_files(self) -> List[Dict]:
        """既存のファイル一覧を取得"""
        try:
            files = []
            for f in genai.list_files():
                files.append({
                    "name": f.name,
                    "display_name": f.display_name,
                    "created_time": f.create_time,
                    "update_time": f.update_time,
                    "size_bytes": f.size_bytes,
                    "state": f.state.name
                })
            return files
        except Exception as e:
            logger.error(f"ファイル一覧取得エラー: {e}")
            return []

    def delete_file(self, file_name: str) -> bool:
        """ファイルを削除"""
        try:
            genai.delete_file(name=file_name)
            logger.info(f"ファイル削除成功: {file_name}")
            return True
        except Exception as e:
            logger.error(f"ファイル削除エラー: {e}")
            return False

    def query_file(self, file_name: str, query: str) -> Dict[str, Any]:
        """ファイルに対して質問を実行（Gemini 2.0 Flash使用）"""
        try:
            logger.info(f"Querying file: {file_name} with query: {query}")
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # ファイルを取得してコンテンツに追加
            # 注意: genai.get_file(name) で取得したオブジェクトを渡す必要はない場合もあるが
            # generate_content の contents には URI または File API の name を指定できる
            
            # プロンプト構築
            prompt = [
                "以下のドキュメントに基づいて、質問に回答してください。ドキュメントにない情報は「情報がありません」と答えてください。",
                {"file_data": {"mime_type": "text/plain", "file_uri": genai.get_file(file_name).uri}},
                f"質問: {query}"
            ]
            
            response = model.generate_content(prompt)
            
            return {
                "answer": response.text,
                "source_file": file_name
            }

        except Exception as e:
            logger.error(f"Q&A実行エラー: {e}")
            return {"error": str(e)}


# ===================================================================
# Streamlit UI管理クラス
# ===================================================================
class FileManagerUI:
    """Gemini File Manager UI"""

    def __init__(self):
        self.configs = FileConfig.get_all_configs()
        self.manager = None

    def setup_page(self):
        st.set_page_config(
            page_title="Gemini File Manager",
            page_icon="📁",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def setup_header(self):
        st.title("📁 Gemini File API Manager")
        st.caption("Google Gemini API (File API) を使用したドキュメント管理と検索")
        st.markdown("---")

    def setup_sidebar(self) -> Tuple[str, bool, str]:
        st.sidebar.title("メニュー")
        mode = st.sidebar.radio("機能選択", ["ファイル管理", "検索・Q&A"], index=0)
        st.sidebar.markdown("---")
        
        # APIキー確認
        with st.sidebar.expander("🔑 API設定確認", expanded=False):
            api_key_status = "✅ 設定済み" if os.getenv("GOOGLE_API_KEY") else "❌ 未設定"
            st.write(f"**Google APIキー**: {api_key_status}")

        return "gemini-1.5-flash", False, mode

    def display_dataset_selection(self) -> List[str]:
        st.subheader("📋 データセット選択 (アップロード)")
        col1, col2 = st.columns(2)
        selected_datasets = []

        output_dir = Path("qa_output")
        for idx, (dataset_type, config) in enumerate(self.configs.items()):
            col = col1 if idx % 2 == 0 else col2
            with col:
                # ファイル存在確認（簡易）
                candidates = list(output_dir.glob(f"*{config.filename}*"))
                exists = len(candidates) > 0
                
                label = f"{config.display_name}"
                if exists:
                    label += " (✅)"
                else:
                    label += " (❌)"

                selected = st.checkbox(
                    label,
                    key=f"dataset_{dataset_type}",
                    disabled=not exists,
                    help=f"想定ファイル: {config.filename}"
                )
                if selected:
                    selected_datasets.append(dataset_type)
        
        return selected_datasets

    def display_results(self, results: Dict[str, Dict]):
        st.subheader("📊 処理結果")
        successful = {k: v for k, v in results.items() if v.get("success")}
        failed = {k: v for k, v in results.items() if not v.get("success")}

        if successful:
            st.success(f"成功: {len(successful)}件")
            for dtype, res in successful.items():
                st.write(f"- **{res.get('display_name')}**: {res.get('file_name')} ({res.get('size_bytes', 0)/1024:.1f} KB)")

        if failed:
            st.error(f"失敗: {len(failed)}件")
            for dtype, res in failed.items():
                st.write(f"- **{dtype}**: {res['error']}")

    def display_existing_files(self, manager: GeminiFileManager):
        st.subheader("📚 アップロード済みファイル一覧")
        files = manager.list_files()
        
        if files:
            for f in files:
                with st.expander(f"📄 {f['display_name']} ({f['name']})"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**作成日時**: {f['created_time']}")
                        st.write(f"**サイズ**: {f['size_bytes'] / 1024:.1f} KB")
                        st.write(f"**状態**: {f['state']}")
                    with col2:
                        if st.button("🗑️ 削除", key=f"del_{f['name']}"):
                            if manager.delete_file(f['name']):
                                st.success("削除しました")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("削除失敗")
        else:
            st.info("ファイルはありません")

    def display_search_interface(self, manager: GeminiFileManager):
        st.subheader("🔎 Semantic Search & QA")
        
        files = manager.list_files()
        active_files = [f for f in files if f['state'] == 'ACTIVE']
        
        if not active_files:
            st.warning("検索可能なファイルがありません。「ファイル管理」タブでアップロードしてください。")
            return

        file_options = {f['display_name']: f['name'] for f in active_files}
        selected_file_name = st.selectbox(
            "検索対象ファイル",
            options=list(file_options.keys()),
            format_func=lambda x: f"{x} ({file_options[x]})"
        )
        target_file_name = file_options[selected_file_name]

        query = st.text_area("質問を入力してください", height=100)
        
        if st.button("🔍 検索・回答生成", type="primary"):
            if not query:
                st.warning("質問を入力してください")
                return
            
            with st.spinner("Gemini 1.5 Flash で回答生成中..."):
                result = manager.query_file(target_file_name, query)
                
                if "error" in result:
                    st.error(f"エラーが発生しました: {result['error']}")
                else:
                    st.success("回答生成完了")
                    st.markdown("### 🤖 AI回答")
                    st.write(result["answer"])


# ===================================================================
# メイン関数
# ===================================================================
def main():
    ui = FileManagerUI()
    ui.setup_page()
    ui.setup_header()

    if not GEMINI_AVAILABLE:
        st.error("Google Generative AI SDKが必要です。`pip install google-generativeai`")
        return

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEYが設定されていません")
        return

    selected_model, process_all, mode = ui.setup_sidebar()

    try:
        manager = GeminiFileManager()
        ui.manager = manager
    except Exception as e:
        st.error(f"Manager初期化失敗: {e}")
        return

    if mode == "ファイル管理":
        tab1, tab2 = st.tabs(["🔗 アップロード", "📚 ファイル一覧"])

        with tab1:
            selected_datasets = ui.display_dataset_selection()
            if selected_datasets and st.button("🚀 アップロード開始", type="primary"):
                results = {}
                progress = st.progress(0)
                for i, dtype in enumerate(selected_datasets):
                    with st.spinner(f"処理中: {dtype}..."):
                        results[dtype] = manager.process_single_dataset(dtype)
                    progress.progress((i + 1) / len(selected_datasets))
                ui.display_results(results)

        with tab2:
            ui.display_existing_files(manager)
            if st.button("🔄 一覧更新"):
                st.rerun()
    
    elif mode == "検索・Q&A":
        ui.display_search_interface(manager)

if __name__ == "__main__":
    main()
