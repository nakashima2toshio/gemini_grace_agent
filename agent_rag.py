#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent RAG Q&A生成・Qdrant管理 Streamlit アプリケーション
sudo systemctl restart streamlit-app

実行コマンド：
streamlit run agent_rag.py --server.port 8501

詳細な仕様、実行方法、アーキテクチャについては、プロジェクトルートの `README.md` を参照してください。

[リモートサーバー管理 (GCP)]:
ssh -i ~/.ssh/gcp_key_v2 nakashima@34.84.198.115

# 設定ファイルの変更を反映
sudo systemctl daemon-reload

# サーバー起動時に自動で立ち上がるように設定
sudo systemctl enable streamlit-app

# 今すぐ起動する
sudo systemctl start streamlit-app

# 停止する
sudo systemctl stop streamlit-app

# 再起動する
sudo systemctl restart streamlit-app

# 状態確認
sudo systemctl status streamlit-app

# ログ確認
journalctl -u streamlit-app -f
"""

import streamlit as st

# UIページをインポート
from ui.pages import (
    show_system_explanation_page,
    show_rag_download_page,
    show_qa_generation_page,
    show_qdrant_registration_page,
    show_qdrant_page,
    show_qdrant_search_page,
    show_grace_chat_page,
)
from ui.pages.agent_chat_page import show_agent_chat_page
from ui.pages.log_viewer_page import show_log_viewer_page


def main():
    """メインアプリケーション - 画面選択"""

    # ページ設定
    st.set_page_config(page_title="Agent RAG(Gemini)", page_icon="🤖", layout="wide")

    # サイドバー：画面選択
    with st.sidebar:
        st.title("Agent RAG (Gemini3)")
        st.divider()

        # メニュー見出し
        st.markdown("**メニュー**")

        # 画面選択
        page = st.radio(
            "機能選択",
            options=[
                "explanation", # <-- Moved to top
                "agent_chat",
                "grace_chat",
                "log_viewer",
                "rag_download",
                "qa_generation",
                "qdrant_registration",
                "show_qdrant",
                "qdrant_search",
            ],
            format_func=lambda x: {
                "explanation": "📖 説明", # <-- Label for explanation
                "agent_chat": "🤖 エージェント対話 (ReAct+Reflection)",
                "grace_chat": "🧠 GRACE エージェント (Agent)",
                "log_viewer": "📊 未回答ログ",
                "rag_download": "📥 RAGデータダウンロード",
                "qa_generation": "🤖 Q/A生成",
                "qdrant_registration": "📥 CSVデータ登録",
                "show_qdrant": "🗄️ Qdrantデータ管理",
                "qdrant_search": "🔎 Qdrant検索",
            }[x],
            label_visibility="collapsed",
        )
        st.markdown("全ソースは： [GitHub: nakashima2toshio/gemini_agent_rag](https://github.com/nakashima2toshio/gemini_agent_rag)")
        st.divider() # Removed one of the two consecutive dividers


    # 選択された画面を表示
    page_mapping = {
        "agent_chat": show_agent_chat_page,
        "grace_chat": show_grace_chat_page,
        "log_viewer": show_log_viewer_page,
        "explanation": show_system_explanation_page,
        "rag_download": show_rag_download_page,
        "qa_generation": show_qa_generation_page,
        "qdrant_registration": show_qdrant_registration_page,
        "show_qdrant": show_qdrant_page,
        "qdrant_search": show_qdrant_search_page,
    }
    page_mapping[page]()


if __name__ == "__main__":
    main()