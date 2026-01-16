# grace_chat_page.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
grace_chat_page.py - GRACE エージェント チャット画面
===================================================
GRACEアーキテクチャを使用したエージェントとの対話インターフェース。
"""

import os
import logging
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def show_grace_chat_page():
    st.title("🧠 GRACE エージェント (New)")
    st.caption("Goal-Reasoning-Action-Critique-Execute Architecture")

    # -------------------------------------------------------------------------
    # 元ドキュメント表示エリア (Modified to support OUTPUT/*.csv)
    # -------------------------------------------------------------------------
    with st.expander("📄 元ドキュメントの表示", expanded=False):
        st.markdown("OUTPUTディレクトリのCSVファイルを選択：")

        output_dir = "OUTPUT"

        file_options = {}
        if os.path.exists(output_dir):
            import glob

            # OUTPUT/*.csv を全て取得
            csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
            if csv_files:
                # 更新日時順にソート（最新順）
                csv_files.sort(key=os.path.getctime, reverse=True)

                for file_path in csv_files:
                    # ファイル名（拡張子なし）をラベルとして使用
                    file_name = os.path.basename(file_path)
                    label = os.path.splitext(file_name)[0]
                    file_options[label] = file_path

        if file_options:
            selected_doc_label = st.selectbox(
                "CSVファイルを選択:",
                options=list(file_options.keys()),
                key="grace_original_doc_selector"
            )

            if selected_doc_label:
                file_path = file_options[selected_doc_label]
                st.caption(f"📁 参照ファイル: `{file_path}`")

                try:
                    # CSVファイルをDataFrameとして読み込み（先頭100行）
                    df = pd.read_csv(file_path, nrows=100)

                    # ファイル全体の行数を取得（効率的な方法）
                    total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1  # ヘッダー除く

                    st.caption(f"📊 表示: 先頭100行 / 全{total_rows:,}行 | カラム数: {len(df.columns)}")

                    st.dataframe(
                        df,
                        width='stretch',
                        hide_index=False,
                        height=400
                    )

                except Exception as e:
                    st.error(f"❌ 読み込みエラー: {e}")
        else:
            st.info("📂 OUTPUTディレクトリにCSVファイルが見つかりません。")

    # -------------------------------------------------------------------------
    # GRACEエージェントのメイン機能
    # -------------------------------------------------------------------------
    st.divider()
    st.markdown("### 💬 チャット")

    # サイドバー設定
    with st.sidebar:
        st.header("⚙️ GRACE エージェント設定")

        # 設定項目のプレースホルダー
        st.info("GRACE エージェントの設定はここに表示されます。")

        if st.button("🗑️ 会話履歴をクリア"):
            if "grace_chat_history" in st.session_state:
                st.session_state.grace_chat_history = []
            st.rerun()

    # セッション状態の初期化
    if "grace_chat_history" not in st.session_state:
        st.session_state.grace_chat_history = []

    # チャット履歴の表示
    for message in st.session_state.grace_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザー入力処理
    if prompt := st.chat_input("質問を入力してください..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.grace_chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # ここにGRACEエージェントのロジックを実装
            response = f"GRACEエージェント: 「{prompt}」に対する回答を処理中..."
            st.markdown(response)
            st.session_state.grace_chat_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    show_grace_chat_page()
