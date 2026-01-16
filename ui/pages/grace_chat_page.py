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
from qdrant_client import QdrantClient

# Configuration and Services
from config import AgentConfig, GeminiConfig
from services.agent_service import ReActAgent, get_available_collections_from_qdrant_helper

logger = logging.getLogger(__name__)


def show_grace_chat_page():
    st.title("🧠 GRACE エージェント (New)")
    st.caption("Goal-Reasoning-Action-Critique-Execute Architecture")

    # -------------------------------------------------------------------------
    # コレクションデータの表示エリア (Modified)
    # -------------------------------------------------------------------------
    with st.expander("📊 コレクションデータの表示", expanded=False):
        st.markdown("登録されているコレクションから、質問と回答のデータを100件表示します。")

        # プレビュー用のコレクション取得
        preview_collections = get_available_collections_from_qdrant_helper()

        if preview_collections:
            col1, col2 = st.columns([1, 3])
            with col1:
                target_collection = st.selectbox(
                    "コレクションを選択:",
                    preview_collections,
                    index=0,
                    key="grace_collection_selector"
                )

            if target_collection:
                try:
                    # Qdrantクライアント接続
                    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

                    st.caption(f"📊 コレクション: **{target_collection}** から100件を表示")

                    # Qdrantから直接データを取得（scrollを使用）
                    points, next_page_offset = client.scroll(
                        collection_name=target_collection,
                        limit=100,
                        with_payload=True,
                        with_vectors=False
                    )

                    if points:
                        # DataFrameに変換
                        data_list = []
                        for point in points:
                            payload = point.payload or {}
                            data_list.append({
                                "ID"      : point.id,
                                "Question": payload.get("question", "N/A"),
                                "Answer"  : payload.get("answer", "N/A")
                            })

                        df_preview = pd.DataFrame(data_list)

                        # コレクションの総ポイント数を取得
                        try:
                            col_info = client.get_collection(target_collection)
                            total_points = col_info.points_count if hasattr(col_info, 'points_count') else "N/A"
                        except:
                            total_points = "N/A"

                        st.caption(f"📈 表示: {len(data_list)} 件 / 総ポイント数: {total_points}")

                        # データフレーム表示（スクロール可能）
                        st.dataframe(
                            df_preview,
                            width='stretch',
                            hide_index=True,
                            height=600,  # スクロール可能な高さ
                            column_config={
                                "ID"      : st.column_config.NumberColumn("ID", width="small"),
                                "Question": st.column_config.TextColumn("質問 (Question)", width="medium"),
                                "Answer"  : st.column_config.TextColumn("回答 (Answer)", width="large")
                            }
                        )
                    else:
                        st.info(f"コレクション '{target_collection}' にデータが見つかりませんでした。")

                except Exception as e:
                    st.error(f"データ取得エラー: {e}")
        else:
            st.warning("表示可能なコレクションがありません。Qdrantの状態を確認してください。")

    # -------------------------------------------------------------------------
    # GRACEエージェントのメイン機能
    # -------------------------------------------------------------------------
    st.divider()
    st.markdown("### 💬 チャット")

    # サイドバー設定
    with st.sidebar:
        st.header("⚙️ GRACE エージェント設定")

        # モデル選択
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

        # 検索対象コレクションの選択（マルチセレクト）
        selected_collections = st.multiselect(
            "検索対象コレクション (Target Collections)",
            options=all_collections,
            default=all_collections if all_collections != ["(None)"] else [],
            help="GRACEエージェントが検索ツールを使用する際に、候補として提示されるコレクションです。"
        )

        if st.button("🗑️ 会話履歴をクリア"):
            st.session_state.grace_chat_history = []
            # current_collections もクリアして再初期化を強制
            if "grace_current_collections" in st.session_state:
                del st.session_state["grace_current_collections"]
            # current_model もクリア
            if "grace_current_model" in st.session_state:
                del st.session_state["grace_current_model"]
            st.rerun()

        # キャッシュリセットボタン
        if st.button("🔄 キャッシュをリセット"):
            from agent_cache import collection_cache
            if "grace_session_id" in st.session_state:
                collection_cache.clear(st.session_state.grace_session_id)
                st.toast("✅ キャッシュをクリアしました")

        # キャッシュ統計表示
        with st.expander("📊 キャッシュ統計", expanded=False):
            from agent_cache import collection_cache
            if "grace_session_id" in st.session_state:
                stats = collection_cache.get_stats(st.session_state.grace_session_id)
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

    # セッション状態の初期化
    if "grace_chat_history" not in st.session_state:
        st.session_state.grace_chat_history = []

    # エージェント用のセッションIDを初期化
    if "grace_session_id" not in st.session_state:
        import uuid
        st.session_state.grace_session_id = str(uuid.uuid4())
        logger.info(f"New GRACE session ID created: {st.session_state.grace_session_id}")

    # 前回のコレクション選択状態・モデルと比較
    current_collections_key = "grace_current_collections"
    current_model_key = "grace_current_model"
    should_reinitialize = False

    # selected_collections はリストなのでソートして比較
    if current_collections_key not in st.session_state:
        should_reinitialize = True
    elif sorted(st.session_state[current_collections_key]) != sorted(selected_collections):
        should_reinitialize = True
        st.toast("検索対象コレクションが変更されたため、エージェントを再設定します。")

    # モデルの変更チェック
    if current_model_key not in st.session_state:
        should_reinitialize = True
    elif st.session_state[current_model_key] != selected_model:
        should_reinitialize = True
        st.toast(f"モデルが変更されました: {selected_model}")

    if should_reinitialize or "grace_agent" not in st.session_state or st.session_state.grace_agent is None:
        try:
            st.session_state.grace_agent = ReActAgent(
                selected_collections,
                selected_model,
                session_id=st.session_state.grace_session_id
            )
            st.session_state[current_collections_key] = selected_collections
            st.session_state[current_model_key] = selected_model
            st.toast("GRACEエージェントの準備が完了しました（キャッシュ+並列検索）。")
        except Exception as e:
            st.error(f"エージェントの初期化に失敗しました: {e}")
            return

    # チャット履歴の表示
    for message in st.session_state.grace_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザー入力処理
    if prompt := st.chat_input("質問を入力してください..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.grace_chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            st_expander_placeholder = st.empty()

            # 思考ログを蓄積するリスト
            current_thought_log_content: List[str] = []

            response_text_placeholder = st.empty()

            final_response_content = ""

            try:
                # エージェントからイベントを取得
                for event in st.session_state.grace_agent.execute_turn(prompt):
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
                        response_text_placeholder.markdown(final_response_content)

                if final_response_content:
                    st.session_state.grace_chat_history.append({"role": "assistant", "content": final_response_content})
                else:
                    st.warning("エージェントからの応答がありませんでした。")

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                logger.error(f"GRACE Chat Error: {e}", exc_info=True)


if __name__ == "__main__":
    show_grace_chat_page()

