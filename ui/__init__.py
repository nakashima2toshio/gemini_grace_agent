#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ui - Streamlit UIモジュール
===========================
RAGツールのStreamlit UIコンポーネント

ページ一覧:
- explanation_page: システム説明
- download_page: RAGデータダウンロード
- qa_generation_page: Q/A生成
- qdrant_registration_page: Qdrant登録
- qdrant_show_page: Qdrantデータ表示
- qdrant_search_page: Qdrant検索
"""

# ページ関数のインポート（遅延インポートを推奨）
# 直接インポートする場合は以下を使用:
# from ui.pages.explanation_page import show_system_explanation_page
# from ui.pages.download_page import show_rag_download_page
# from ui.pages.qa_generation_page import show_qa_generation_page
# from ui.pages.qdrant_registration_page import show_qdrant_registration_page
# from ui.pages.qdrant_show_page import show_qdrant_page
# from ui.pages.qdrant_search_page import show_qdrant_search_page

__version__ = "1.0.0"

__all__ = [
    "show_system_explanation_page",
    "show_rag_download_page",
    "show_qa_generation_page",
    "show_qdrant_registration_page",
    "show_qdrant_page",
    "show_qdrant_search_page",
]


def get_page_mapping():
    """ページマッピングを取得（遅延インポート）"""
    from ui.pages.explanation_page import show_system_explanation_page
    from ui.pages.download_page import show_rag_download_page
    from ui.pages.qa_generation_page import show_qa_generation_page
    from ui.pages.qdrant_registration_page import show_qdrant_registration_page
    from ui.pages.qdrant_show_page import show_qdrant_page
    from ui.pages.qdrant_search_page import show_qdrant_search_page

    return {
        "explanation": show_system_explanation_page,
        "rag_download": show_rag_download_page,
        "qa_generation": show_qa_generation_page,
        "qdrant_registration": show_qdrant_registration_page,
        "show_qdrant": show_qdrant_page,
        "qdrant_search": show_qdrant_search_page,
    }


def get_page_labels():
    """ページラベルを取得"""
    return {
        "explanation": "📖 説明",
        "rag_download": "📥 RAGデータダウンロード",
        "qa_generation": "🤖 Q/A生成",
        "qdrant_registration": "🗄️ Qdrant登録",
        "show_qdrant": "🔍 Show-Qdrant",
        "qdrant_search": "🔎 Qdrant検索",
    }