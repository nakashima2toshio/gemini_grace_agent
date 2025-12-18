#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ui/components/rag_components.py - RAGデータ前処理用UIコンポーネント
==================================================================
Streamlit依存のUI関数群。helper_rag_ui.pyから移動。

元ファイル: helper_rag_ui.py
"""

import streamlit as st
import pandas as pd
import logging
from typing import List, Dict, Any, Optional

from helper_llm import (
    DEFAULT_LLM_PROVIDER,
    get_available_llm_models,
    get_llm_model_limits,
    get_llm_model_pricing,
    get_available_embedding_models,
    get_embedding_model_pricing,
)
from helper_rag import RAGConfig, TokenManager

logger = logging.getLogger(__name__)

# ==================================================
# UI関数群
# ==================================================
def select_model(key: str = "model_selection") -> str:
    """モデル選択UI"""
    models = get_available_llm_models()
    default_model = DEFAULT_LLM_PROVIDER

    try:
        default_index = models.index(default_model)
    except ValueError:
        default_index = 0

    selected = st.sidebar.selectbox(
        "🤖 モデルを選択",
        models,
        index=default_index,
        key=key,
        help="利用するLLMモデルを選択してください"
    )

    return selected


def show_model_info(selected_model: str) -> None:
    """選択されたモデルの情報を表示"""
    try:
        limits = get_llm_model_limits(selected_model)
        pricing = get_llm_model_pricing(selected_model)

        with st.sidebar.expander("📊 選択モデル情報", expanded=False):
            # 基本情報
            col1, col2 = st.columns(2)
            with col1:
                st.write("**最大入力**")
                st.write(f"{limits.get('max_tokens', 0):,}")
            with col2:
                st.write("**最大出力**")
                st.write(f"{limits.get('max_output', 0):,}")

            # 料金情報
            st.write("**料金（1000トークン）**")
            st.write(f"- 入力: ${pricing.get('input', 0.0):.5f}")
            st.write(f"- 出力: ${pricing.get('output', 0.0):.5f}")

            # モデル特性（Geminiに特化）
            if "gemini-2.0" in selected_model:
                st.info("✨ Gemini 2.0 シリーズ")
                st.caption("高速・高性能な次世代モデル")
            elif "gemini-1.5" in selected_model:
                st.info("💡 Gemini 1.5 シリーズ")
                st.caption("長文コンテキスト・マルチモーダル対応")
            elif "gpt" in selected_model:
                st.info("⚙️ OpenAI互換モデル")
                st.caption("OpenAI APIを介して利用可能")
            else:
                st.info("💬 その他のLLMモデル")

            # RAG用途での推奨度
            st.write("**RAG用途推奨度**")
            if "flash" in selected_model:
                st.success("✅ 最適（高速・コスト効率良好）")
            elif "pro" in selected_model:
                st.info("💡 高品質（詳細な推論に最適）")
            elif "gpt" in selected_model:
                st.info("💬 OpenAI互換（用途に応じて選択）")
            else:
                st.info("💬 標準的な性能")

    except Exception as e:
        logger.error(f"モデル情報表示エラー: {e}")
        st.sidebar.error("モデル情報の取得に失敗しました")


def estimate_token_usage(df_processed: pd.DataFrame, selected_model: str) -> None:
    """処理済みデータのトークン使用量推定"""
    try:
        if 'Combined_Text' in df_processed.columns:
            # サンプルテキストでトークン数を推定
            sample_size = min(10, len(df_processed))
            sample_texts = df_processed['Combined_Text'].head(sample_size).tolist()
            total_chars = df_processed['Combined_Text'].str.len().sum()

            if sample_texts:
                sample_text = " ".join(sample_texts)
                # TokenManagerのcount_tokensを使用
                sample_tokens = TokenManager.count_tokens(sample_text, selected_model)
                sample_chars = len(sample_text)

                if sample_chars > 0:
                    # 全体のトークン数を推定
                    estimated_total_tokens = int((total_chars / sample_chars) * sample_tokens)

                    # Embeddingモデルの料金を取得 (Gemini Embeddingを想定)
                    embedding_model_name = get_available_embedding_models()[0] # デフォルトのGemini Embeddingモデルを取得
                    embedding_pricing_per_1k_tokens = get_embedding_model_pricing(embedding_model_name)

                    with st.expander("🔢 トークン使用量推定", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("推定総トークン数", f"{estimated_total_tokens:,}")
                        with col2:
                            avg_tokens_per_record = estimated_total_tokens / len(df_processed)
                            st.metric("平均トークン/レコード", f"{avg_tokens_per_record:.0f}")
                        with col3:
                            # embedding用のコスト推定
                            embedding_cost = (estimated_total_tokens / 1000) * embedding_pricing_per_1k_tokens
                            st.metric("推定embedding費用", f"${embedding_cost:.4f}")

                        st.info(f"💡 選択LLMモデル「{selected_model}」およびEmbeddingモデル「{embedding_model_name}」での推定値")
                        st.caption("※ 実際のトークン数とは異なる場合があります")

    except Exception as e:
        logger.error(f"トークン使用量推定エラー: {e}")
        st.error("トークン使用量の推定に失敗しました")


def display_statistics(df_original: pd.DataFrame, df_processed: pd.DataFrame, dataset_type: str = None) -> None:
    """処理前後の統計情報を表示"""
    st.subheader("📊 統計情報")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("元の行数", f"{len(df_original):,}")
    with col2:
        st.metric("処理後の行数", f"{len(df_processed):,}")
    with col3:
        removed_rows = len(df_original) - len(df_processed)
        st.metric("除去された行数", f"{removed_rows:,}")

    # 結合テキストの分析
    if 'Combined_Text' in df_processed.columns:
        st.subheader("📝 結合後テキスト分析")
        text_lengths = df_processed['Combined_Text'].str.len()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均文字数", f"{text_lengths.mean():.0f}")
        with col2:
            st.metric("最大文字数", f"{text_lengths.max():,}")
        with col3:
            st.metric("最小文字数", f"{text_lengths.min():,}")

        # パーセンタイル表示
        percentiles = text_lengths.quantile([0.25, 0.5, 0.75])
        st.write("**文字数分布:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"25%点: {percentiles[0.25]:.0f}文字")
        with col2:
            st.write(f"中央値: {percentiles[0.5]:.0f}文字")
        with col3:
            st.write(f"75%点: {percentiles[0.75]:.0f}文字")


# ==================================================
# 使用方法説明関数（データセット対応）
# ==================================================
def show_usage_instructions(dataset_type: str) -> None:
    """使用方法の説明を表示（データセット別対応）"""
    config_data = RAGConfig.get_config(dataset_type)
    required_columns_str = ", ".join(config_data["required_columns"])

    st.markdown("---")
    st.subheader("📖 使用方法")

    # 基本的な使用方法
    basic_usage = f"""
    ### 📋 前処理手順
    1. **モデル選択**: サイドバーでRAG用途に適したモデルを選択
    2. **CSVファイルアップロード**: {required_columns_str} 列を含むCSVファイルを選択
    3. **前処理実行**: 以下の処理が自動で実行されます：
       - 改行・空白の正規化
       - 重複行の除去
       - 空行の除去
       - 引用符の正規化
    4. **列結合**: Vector Store/RAG用に最適化された自然な文章として結合
    5. **トークン使用量確認**: 選択モデルでのトークン数とコストを推定
    6. **ダウンロード**: 前処理済みデータを各種形式でダウンロード

    ### 🎯 RAG最適化の特徴
    - **自然な文章結合**: ラベルなしで読みやすい文章として結合
    - **Gemini Embedding対応**: `gemini-embedding-001`等に最適化
    - **検索性能向上**: 意味的検索の精度向上

    ### 💡 推奨モデル
    - **コスト重視**: `gemini-2.0-flash`
    - **品質重視**: `gemini-2.0-pro`
    - **OpenAI互換**: `gpt-4o-mini`, `gpt-4o` （OpenAI APIキーが必要）
    """

    # データセット特有の説明
    dataset_specific = ""
    if dataset_type == "customer_support_faq":
        dataset_specific = """
    ### 💬 カスタマーサポートFAQの特徴
    - **FAQ形式**: 質問と回答のペアによる構造
    - **実用的な内容**: 実際の顧客からの質問に基づく
    - **簡潔な回答**: 分かりやすく実用的な回答
        """
    elif dataset_type == "medical_qa":
        dataset_specific = """
    ### 🏥 医療QAデータの特徴
    - **複雑な推論**: Complex_CoT列による段階的推論過程
    - **専門用語**: 医療専門用語の適切な処理
    - **詳細な回答**: 医療情報に特化した包括的な回答
        """
    elif dataset_type == "sciq_qa":
        dataset_specific = """
    ### 🔬 SciQデータの特徴
    - **科学・技術問題**: 化学、物理、生物学、数学などの分野をカバー
    - **多肢選択形式**: distractor列による選択肢問題
    - **補足説明**: support列による詳細な背景情報
    - **幅広い難易度**: 基礎から応用まで様々なレベルの科学問題
        """
    elif dataset_type == "legal_qa":
        dataset_specific = """
    ### ⚖️ 法律・判例QAデータの特徴
    - **法律専門用語**: 条文、判例、法的概念の適切な処理
    - **詳細な回答**: 法的根拠を含む包括的な説明
    - **正確性重視**: 法的情報の正確性を保持した前処理
    - **引用・参照**: 条文番号や判例番号などの法的根拠の保護
        """

    st.markdown(basic_usage + dataset_specific)


# ==================================================
# ページ設定関数（共通）
# ==================================================
def setup_page_config(dataset_type: str) -> None:
    """ページ設定の初期化"""
    config_data = RAGConfig.get_config(dataset_type)

    try:
        st.set_page_config(
            page_title=f"{config_data['name']}前処理（完全独立版）",
            page_icon=config_data['icon'],
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except st.errors.StreamlitAPIException:
        pass


def setup_page_header(dataset_type: str) -> None:
    """ページヘッダーの設定"""
    config_data = RAGConfig.get_config(dataset_type)

    st.title(f"{config_data['icon']} {config_data['name']}前処理アプリ")
    st.caption("RAG（Retrieval-Augmented Generation）用データ前処理 - 完全独立版")
    st.markdown("---")


def setup_sidebar_header(dataset_type: str) -> None:
    """サイドバーヘッダーの設定"""
    config_data = RAGConfig.get_config(dataset_type)

    st.sidebar.title(f"{config_data['icon']} {config_data['name']}")
    st.sidebar.markdown("---")


# ==================================================
# エクスポート
# ==================================================
__all__ = [
    "select_model",
    "show_model_info",
    "estimate_token_usage",
    "display_statistics",
    "show_usage_instructions",
    "setup_page_config",
    "setup_page_header",
    "setup_sidebar_header",
]