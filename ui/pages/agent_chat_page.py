# agent_chat_page.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
agent_chat_page.py - ハイブリッド・ナレッジ・エージェント チャット画面
================================================================
Gemini 2.0 Flash を使用した ReAct 型エージェントとの対話インターフェース。
Qdrant 上のナレッジベース（コレクション）を動的に選択し、RAG 検索を行いながら回答します。
"""

import os
import logging
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from qdrant_client import QdrantClient  # Added QdrantClient import

# Configuration and Tools
from config import AgentConfig, GeminiConfig
from services.agent_service import ReActAgent, get_available_collections_from_qdrant_helper

logger = logging.getLogger(__name__)


def show_agent_chat_page():
    st.title("🤖 エージェント対話 (Agent Chat)")
    st.caption("Gemini 2.0 Flash + ReAct + Qdrant Hybrid RAG (Dense + Sparse)")

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
                key="original_doc_selector"
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
    # 入力クエリの参考用 Q&A表示エリア (Added)
    # -------------------------------------------------------------------------
    with st.expander(
            "📚 登録済みQ&Aの参照 (生成AI：Geminiが元ドキュメントの意味を解析しドキュメント内の重要箇所に基づいて「質問」と「回答」のペアを自動抽出しRAGシステムで利用可能なCSV形式のナレッジデータとして生成）入力クエリのヒント",
            expanded=False):
        st.markdown("登録されているコレクションから、質問と回答のサンプルを100件表示します。質問の参考にしてください。")

        # プレビュー用のコレクション取得
        preview_collections = get_available_collections_from_qdrant_helper()

        if preview_collections:
            col1, col2 = st.columns([1, 3])
            with col1:
                target_collection = st.selectbox(
                    "コレクションを選択:",
                    preview_collections,
                    index=0,
                    key="preview_collection_selector"
                )

            if target_collection:
                try:
                    # Qdrantクライアント接続
                    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

                    # 上位100件を取得
                    points, _ = client.scroll(
                        collection_name=target_collection,
                        limit=100,
                        with_payload=True,
                        with_vectors=False
                    )

                    if points:
                        data_list = []
                        for point in points:
                            payload = point.payload or {}
                            data_list.append({
                                "Question": payload.get("question", "N/A"),
                                "Answer"  : payload.get("answer", "N/A")
                            })

                        df_preview = pd.DataFrame(data_list)
                        st.dataframe(
                            df_preview,
                            width='stretch',  # use_container_width=True から変更（2025-12-31以降非推奨）
                            hide_index=True,
                            column_config={
                                "Question": st.column_config.TextColumn("質問 (Question)", width="medium"),
                                "Answer"  : st.column_config.TextColumn("回答 (Answer)", width="large"),
                            }
                        )
                    else:
                        st.info(f"コレクション '{target_collection}' にデータが見つかりませんでした。")

                except Exception as e:
                    st.error(f"データ取得エラー: {e}")
        else:
            st.warning("表示可能なコレクションがありません。Qdrantの状態を確認してください。")

    # 1. サイドバー設定
    with st.sidebar:
        st.header("⚙️ エージェント設定")

        # モデル選択の追加
        selected_model = st.selectbox(
            "使用モデル (Model)",
            options=GeminiConfig.AVAILABLE_MODELS,
            index=GeminiConfig.AVAILABLE_MODELS.index(AgentConfig.MODEL_NAME)
            if AgentConfig.MODEL_NAME in GeminiConfig.AVAILABLE_MODELS else 0
        )

        # コレクション一覧の取得
        all_collections = get_available_collections_from_qdrant_helper()

        if not all_collections:
            st.warning("利用可能なコレクションが見つかりません。Qdrantサーバーを確認してください。")
            all_collections = ["(None)"]

        # 検索対象コレクションの選択（マルチセレクトに変更）
        selected_collections = st.multiselect(
            "検索対象コレクション (Target Collections)",
            options=all_collections,
            default=all_collections if all_collections != ["(None)"] else [],  # デフォルトは全て選択
            help="エージェントが検索ツールを使用する際に、候補として提示されるコレクションです。"
        )

        if st.button("🗑️ 会話履歴をクリア"):
            st.session_state.chat_history = []
            st.session_state.chat_session = None
            # current_collections もクリアして再初期化を強制
            if "current_collections" in st.session_state:
                del st.session_state["current_collections"]
            # current_model もクリア
            if "current_model" in st.session_state:
                del st.session_state["current_model"]
            st.rerun()

        # キャッシュリセットボタン
        if st.button("🔄 キャッシュをリセット"):
            from agent_cache import collection_cache
            if "agent_session_id" in st.session_state:
                collection_cache.clear(st.session_state.agent_session_id)
                st.toast("✅ キャッシュをクリアしました")

        # キャッシュ統計表示
        with st.expander("📊 キャッシュ統計", expanded=False):
            from agent_cache import collection_cache
            if "agent_session_id" in st.session_state:
                stats = collection_cache.get_stats(st.session_state.agent_session_id)
                if stats.get("cached"):
                    st.metric("キャッシュ状態", "🟢 ヒット")
                    st.metric("コレクション", stats.get("collection", "N/A"))
                    st.metric("前回スコア", f"{stats.get('last_score', 0):.3f}")
                    st.metric("ヒット回数", stats.get("hit_count", 0))
                    st.metric("経過時間", f"{stats.get('age_seconds', 0):.1f}秒")
                else:
                    st.metric("キャッシュ状態", "⚪ なし")
            else:
                st.info("セッションIDが見つかりません")

    # 2. セッション状態の初期化と更新チェック
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # エージェント用のセッションIDを初期化
    if "agent_session_id" not in st.session_state:
        import uuid
        st.session_state.agent_session_id = str(uuid.uuid4())
        logger.info(f"New agent session ID created: {st.session_state.agent_session_id}")

    # 前回のコレクション選択状態・モデルと比較
    current_collections_key = "current_collections"
    current_model_key = "current_model"
    should_reinitialize = False

    # selected_collections はリストなのでソートして比較
    if current_collections_key not in st.session_state:
        should_reinitialize = True
    elif sorted(st.session_state[current_collections_key]) != sorted(selected_collections):
        should_reinitialize = True
        # 設定が変わったので履歴クリアするか確認（今回はしないが、メッセージ出すなどあり）
        st.toast("検索対象コレクションが変更されたため、エージェントを再設定します。")

    # モデルの変更チェック
    if current_model_key not in st.session_state:
        should_reinitialize = True
    elif st.session_state[current_model_key] != selected_model:
        should_reinitialize = True
        st.toast(f"モデルが変更されました: {selected_model}")

    if should_reinitialize or "agent" not in st.session_state or st.session_state.agent is None:
        try:
            st.session_state.agent = ReActAgent(
                selected_collections,
                selected_model,
                session_id=st.session_state.agent_session_id  # セッションIDを渡す
            )
            st.session_state[current_collections_key] = selected_collections
            st.session_state[current_model_key] = selected_model
            st.toast("エージェントの準備が完了しました（キャッシュ+並列検索）。")
        except Exception as e:
            st.error(f"エージェントの初期化に失敗しました: {e}")
            return

    # 3. チャット履歴の表示
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. ユーザー入力処理
    if prompt := st.chat_input("質問を入力してください..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            st_expander_placeholder = st.empty()  # Placeholder for the expander

            # Use a list to accumulate thought log for the expander
            current_thought_log_content: List[str] = []

            response_text_placeholder = st.empty()  # Placeholder for the final response

            final_response_content = ""

            try:
                # Iterate through events yielded by the agent
                for event in st.session_state.agent.execute_turn(prompt):
                    if event["type"] == "log":
                        current_thought_log_content.append(event["content"])
                        with st_expander_placeholder.expander("🤔 エージェントの思考プロセス", expanded=True):
                            for log_entry in current_thought_log_content:
                                st.markdown(log_entry)
                                st.divider()
                    elif event["type"] == "tool_call":
                        current_thought_log_content.append(
                            f"🛠️ **Tool Call:** `{event['name']}`\nArgs: `{event['args']}`")
                        with st_expander_placeholder.expander("🤔 エージェントの思考プロセス", expanded=True):
                            with st.spinner(f"ツールを実行中: {event['name']}..."):
                                for log_entry in current_thought_log_content:
                                    st.markdown(log_entry)
                                    st.divider()
                    elif event["type"] == "tool_result":
                        current_thought_log_content.append(f"📝 **Tool Result:**\n{event['content']}")
                        with st_expander_placeholder.expander("🤔 エージェントの思考プロセス", expanded=True):
                            for log_entry in current_thought_log_content:
                                st.markdown(log_entry)
                                st.divider()
                    elif event["type"] == "final_answer":
                        final_response_content = event["content"]
                        response_text_placeholder.markdown(final_response_content)  # Display final answer

                if final_response_content:
                    st.session_state.chat_history.append({"role": "assistant", "content": final_response_content})
                else:
                    st.warning("エージェントからの応答がありませんでした。")

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                logger.error(f"Chat Error: {e}", exc_info=True)
