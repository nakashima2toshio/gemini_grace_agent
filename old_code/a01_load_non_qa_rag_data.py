# a01_load_non_qa_rag_data.py
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a01_load_non_qa_rag_data.py - 非Q&A型RAGデータ処理ツール
===============================================
起動: streamlit run a01_load_non_qa_rag_data.py --server.port=8502

【主要機能】
✅ 日本語・英語データセットの処理
   - Wikipedia日本語版（動作確認済み）
   - CC100日本語（動作確認済み）
   - CC-News英語ニュース（動作確認済み、7,376件）
   - Livedoorニュースコーパス（動作確認済み、7,376件）
✅ データ検証・品質チェック
✅ RAG用テキスト抽出・前処理
✅ トークン使用量推定
✅ CSV/TXT/JSONフォーマット出力

【対応データセット】
1. wikipedia_ja: Wikipedia日本語版（百科事典的知識）
2. japanese_text: CC100日本語（Webテキストコーパス）
3. cc_news: CC-News（英語ニュース記事、7,376件）
4. livedoor: Livedoorニュースコーパス（日本語ニュース9カテゴリ、7,376件）
"""

import streamlit as st
import pandas as pd
import json
import io
import urllib.request
import tarfile
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any

# ローカルモジュール
from helper_rag import (
    select_model,
    show_model_info,
    validate_data,
    estimate_token_usage,
    save_files_to_output,
    clean_text,
    safe_execute
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================================================================
# 非Q&A型データセット設定
# ===================================================================

class NonQARAGConfig:
    """非Q&A型RAGデータセットの設定"""

    DATASET_CONFIGS = {
        # Wikipedia日本語版 - 動作確認済み
        "wikipedia_ja" : {
            "name"            : "Wikipedia日本語版",
            "icon"            : "📚",
            "required_columns": ["title", "text"],
            "description"     : "Wikipedia日本語版の記事データ",
            "hf_dataset"      : "wikimedia/wikipedia",
            "hf_config"       : "20231101.ja",
            "split"           : "train",
            "streaming"       : True,
            "text_field"      : "text",
            "title_field"     : "title",
            "sample_size"     : 1000
        },

        # CC100日本語 - 動作確認済み
        "japanese_text": {
            "name"            : "日本語Webテキスト（CC100）",
            "icon"            : "📰",
            "required_columns": ["text"],
            "description"     : "日本語Webテキストコーパス",
            "hf_dataset"      : "range3/cc100-ja",
            "hf_config"       : None,
            "split"           : "train",
            "streaming"       : True,
            "text_field"      : "text",
            "title_field"     : None,
            "sample_size"     : 1000
        },

        # CC-News英語ニュース - 動作確認済み
        "cc_news": {
            "name"            : "CC-News（英語ニュース）",
            "icon"            : "🌐",
            "required_columns": ["title", "text"],
            "description"     : "Common Crawl英語ニュース記事",
            "hf_dataset"      : "cc_news",
            "hf_config"       : None,
            "split"           : "train",
            "streaming"       : True,
            "text_field"      : "text",
            "title_field"     : "title",
            "sample_size"     : 500
        },

        # Livedoorニュースコーパス - 日本語ニュース記事
        "livedoor": {
            "name"            : "Livedoorニュースコーパス",
            "icon"            : "📰",
            "required_columns": ["url", "title", "content", "category"],
            "description"     : "Livedoorニュース日本語記事（9カテゴリ）",
            "hf_dataset"      : None,  # 直接ダウンロード
            "download_url"    : "https://www.rondhuit.com/download/ldcc-20140209.tar.gz",
            "hf_config"       : None,
            "split"           : None,
            "streaming"       : False,
            "text_field"      : "content",
            "title_field"     : "title",
            "sample_size"     : 7376  # 全記事数
        }
    }

    @classmethod
    def get_config(cls, dataset_type: str) -> Dict[str, Any]:
        """データセット設定の取得"""
        return cls.DATASET_CONFIGS.get(dataset_type, {
            "name"            : "未知のデータセット",
            "icon"            : "❓",
            "required_columns": [],
            "description"     : "未知のデータセット",
            "hf_dataset"      : None,
            "hf_config"       : None,
            "split"           : "train",
            "port"            : 8500,
            "text_field"      : "text",
            "title_field"     : None,
            "sample_size"     : 1000
        })

    @classmethod
    def get_all_datasets(cls) -> List[str]:
        """全データセットタイプのリストを取得"""
        return list(cls.DATASET_CONFIGS.keys())


# ===================================================================
# データセット別検証関数
# ===================================================================

def validate_wikipedia_data_specific(df: pd.DataFrame) -> List[str]:
    """Wikipedia特有の検証"""
    issues = []

    if 'text' in df.columns:
        # テキストの長さチェック
        text_lengths = df['text'].str.len()
        avg_length = text_lengths.mean()

        if avg_length < 100:
            issues.append(f"⚠️ 平均テキスト長が短い: {avg_length:.0f}文字")
        else:
            issues.append(f"✅ 適切なテキスト長: 平均{avg_length:.0f}文字")

        # Wikipedia特有のマークアップチェック
        wiki_markup = df['text'].str.contains('==|\\[\\[|\\]\\]', regex=True, na=False).sum()
        if wiki_markup > 0:
            percentage = (wiki_markup / len(df)) * 100
            issues.append(f"💡 Wikiマークアップ含む記事: {wiki_markup}件 ({percentage:.1f}%)")

    if 'title' in df.columns:
        # タイトルの重複チェック
        duplicates = df['title'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"⚠️ 重複タイトル: {duplicates}件")

    return issues


def validate_news_data_specific(df: pd.DataFrame, dataset_type: str) -> List[str]:
    """ニュースデータ特有の検証"""
    issues = []

    # テキストフィールドの特定（livedoorはcontent、その他はtext）
    if 'content' in df.columns:
        text_field = 'content'
    elif 'body' in df.columns:
        text_field = 'body'
    else:
        text_field = 'text'

    if text_field in df.columns:
        # 記事の長さ分析
        text_lengths = df[text_field].str.len()
        avg_length = text_lengths.mean()

        issues.append(f"📊 平均記事長: {avg_length:.0f}文字")

        # 短すぎる記事の検出
        short_articles = (text_lengths < 100).sum()
        if short_articles > 0:
            percentage = (short_articles / len(df)) * 100
            issues.append(f"⚠️ 短い記事（<100文字）: {short_articles}件 ({percentage:.1f}%)")

    # カテゴリ情報（livedoorの場合）
    if 'category' in df.columns:
        categories = df['category'].value_counts()
        issues.append(f"📂 カテゴリ数: {len(categories)}種類")
        top_3 = categories.head(3)
        for cat, count in top_3.items():
            issues.append(f"  - {cat}: {count}件")

    return issues


def validate_scientific_data_specific(df: pd.DataFrame, dataset_type: str) -> List[str]:
    """学術論文データ特有の検証"""
    issues = []

    if 'abstract' in df.columns:
        # 要旨の長さ分析
        abstract_lengths = df['abstract'].str.len()
        avg_length = abstract_lengths.mean()

        issues.append(f"📄 平均要旨長: {avg_length:.0f}文字")

        # 学術用語の検出
        academic_keywords = ['research', 'study', 'method', 'result', 'conclusion',
                             '研究', '方法', '結果', '考察']
        has_keywords = df['abstract'].str.contains('|'.join(academic_keywords),
                                                   case=False, na=False).sum()
        percentage = (has_keywords / len(df)) * 100
        issues.append(f"📚 学術的キーワード含む: {has_keywords}件 ({percentage:.1f}%)")

    # PubMed特有
    if dataset_type == "pubmed" and 'abstract' in df.columns:
        # 医学用語の検出
        medical_terms = ['patient', 'treatment', 'disease', 'clinical',
                         '患者', '治療', '疾患', '臨床']
        has_medical = df['abstract'].str.contains('|'.join(medical_terms),
                                                  case=False, na=False).sum()
        percentage = (has_medical / len(df)) * 100
        issues.append(f"🏥 医学用語含む: {has_medical}件 ({percentage:.1f}%)")

    # arXiv特有
    if dataset_type == "arxiv" and 'article' in df.columns:
        # 本文が存在するか
        has_article = df['article'].notna().sum()
        percentage = (has_article / len(df)) * 100
        issues.append(f"📖 本文あり: {has_article}件 ({percentage:.1f}%)")

    return issues


def validate_code_data_specific(df: pd.DataFrame) -> List[str]:
    """コードデータ特有の検証"""
    issues = []

    if 'code' in df.columns:
        # コードの長さ分析
        code_lengths = df['code'].str.len()
        avg_length = code_lengths.mean()
        issues.append(f"💻 平均コード長: {avg_length:.0f}文字")

        # ドキュメント文字列の存在確認
        if 'func_documentation_string' in df.columns:
            has_docs = df['func_documentation_string'].notna().sum()
            percentage = (has_docs / len(df)) * 100
            issues.append(f"📝 ドキュメントあり: {has_docs}件 ({percentage:.1f}%)")

    # プログラミング言語キーワードの検出
    if 'code' in df.columns:
        code_keywords = ['def ', 'class ', 'import ', 'function', 'return']
        has_keywords = df['code'].str.contains('|'.join(code_keywords),
                                               case=False, na=False).sum()
        percentage = (has_keywords / len(df)) * 100
        issues.append(f"🔧 コードキーワード含む: {has_keywords}件 ({percentage:.1f}%)")

    return issues


def validate_stackoverflow_data_specific(df: pd.DataFrame) -> List[str]:
    """Stack Overflow特有の検証"""
    issues = []

    if 'body' in df.columns:
        # 質問の長さ分析
        body_lengths = df['body'].str.len()
        avg_length = body_lengths.mean()
        issues.append(f"❓ 平均質問長: {avg_length:.0f}文字")

    # タグ情報
    if 'tags' in df.columns:
        has_tags = df['tags'].notna().sum()
        percentage = (has_tags / len(df)) * 100
        issues.append(f"🏷️ タグ付き: {has_tags}件 ({percentage:.1f}%)")

        # 人気タグの分析
        if has_tags > 0:
            all_tags = []
            for tags in df['tags'].dropna():
                if isinstance(tags, str):
                    all_tags.extend(tags.split(','))

            if all_tags:
                from collections import Counter
                top_tags = Counter(all_tags).most_common(5)
                issues.append("🔝 人気タグTop5:")
                for tag, count in top_tags:
                    issues.append(f"  - {tag.strip()}: {count}件")

    # 技術キーワードの検出
    if 'body' in df.columns:
        tech_keywords = ['python', 'javascript', 'java', 'error', 'function', 'code']
        has_tech = df['body'].str.contains('|'.join(tech_keywords),
                                           case=False, na=False).sum()
        percentage = (has_tech / len(df)) * 100
        issues.append(f"💡 技術キーワード含む: {has_tech}件 ({percentage:.1f}%)")

    return issues


# ===================================================================
# Livedoorコーパス用関数
# ===================================================================

def download_livedoor_corpus(save_dir: str = "datasets") -> str:
    """Livedoorニュースコーパスをダウンロード

    Args:
        save_dir: ダウンロード先ディレクトリ

    Returns:
        解凍先ディレクトリのパス
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # ダウンロードURL
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    tar_filename = "ldcc-20140209.tar.gz"
    tar_path = save_path / tar_filename

    # ダウンロード
    if not tar_path.exists():
        logger.info(f"Livedoorニュースコーパスをダウンロード中: {url}")
        urllib.request.urlretrieve(url, tar_path)
        logger.info(f"ダウンロード完了: {tar_path}")

    # 解凍
    extract_dir = save_path / "livedoor"
    text_dir = extract_dir / "text"

    if not text_dir.exists():
        logger.info(f"アーカイブを解凍中: {tar_path}")
        with tarfile.open(tar_path, 'r:gz') as tar:
            # セキュリティ: filterパラメータを追加
            tar.extractall(extract_dir, filter='data')
        logger.info(f"解凍完了: {extract_dir}")

    return str(extract_dir)


def load_livedoor_corpus(data_dir: str) -> pd.DataFrame:
    """Livedoorニュースコーパスを読み込み

    Args:
        data_dir: Livedoorコーパスの解凍ディレクトリ

    Returns:
        記事データのDataFrame
    """
    # カテゴリリスト
    categories = [
        'dokujo-tsushin',
        'it-life-hack',
        'kaden-channel',
        'livedoor-homme',
        'movie-enter',
        'peachy',
        'smax',
        'sports-watch',
        'topic-news'
    ]

    articles = []
    text_dir = Path(data_dir) / "text"

    for category in categories:
        category_path = text_dir / category
        if not category_path.exists():
            logger.warning(f"カテゴリディレクトリが見つかりません: {category_path}")
            continue

        # カテゴリ内の全txtファイルを取得
        txt_files = list(category_path.glob("*.txt"))

        for file_path in txt_files:
            # LICENSE.txtなどを除外
            if file_path.name in ['LICENSE.txt', 'README.txt']:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # ファイル形式: 1行目=URL, 2行目=日付, 3行目=タイトル, 4行目以降=本文
                if len(lines) >= 3:
                    url = lines[0].strip()
                    date = lines[1].strip() if len(lines) > 1 else ""
                    title = lines[2].strip() if len(lines) > 2 else ""
                    content = ''.join(lines[3:]).strip() if len(lines) > 3 else ""

                    articles.append({
                        'url': url,
                        'date': date,
                        'title': title,
                        'content': content,
                        'category': category
                    })
            except Exception as e:
                logger.error(f"ファイル読み込みエラー {file_path}: {e}")

    df = pd.DataFrame(articles)
    logger.info(f"Livedoorコーパス読み込み完了: {len(df)}記事")

    return df


# ===================================================================
# データ処理関数
# ===================================================================

@safe_execute
def extract_text_content(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """データセットからテキストコンテンツを抽出"""
    config = NonQARAGConfig.get_config(dataset_type)
    text_field = config["text_field"]
    title_field = config["title_field"]

    df_processed = df.copy()

    # タイトルとテキストを結合
    if title_field and title_field in df.columns and text_field in df.columns:
        # タイトルがある場合は結合
        df_processed['Combined_Text'] = df_processed.apply(
            lambda row: f"{clean_text(str(row.get(title_field, '')))} {clean_text(str(row.get(text_field, '')))}".strip(),
            axis=1
        )
    elif text_field in df.columns:
        # タイトルがない場合はテキストのみ
        df_processed['Combined_Text'] = df_processed[text_field].apply(
            lambda x: clean_text(str(x)) if x is not None else ""
        )
    else:
        # フィールドが見つからない場合のフォールバック
        # 利用可能なテキスト系フィールドを探す
        text_candidates = ['text', 'content', 'body', 'document', 'abstract', 'description']
        found_field = None
        for field in text_candidates:
            if field in df.columns:
                found_field = field
                break

        if found_field:
            df_processed['Combined_Text'] = df_processed[found_field].apply(
                lambda x: clean_text(str(x)) if x is not None else ""
            )
        else:
            # テキストフィールドが見つからない場合は全カラムを結合
            df_processed['Combined_Text'] = df_processed.apply(
                lambda row: " ".join([str(v) for v in row.values if v is not None]),
                axis=1
            )

    # 空のテキストを除外
    df_processed = df_processed[df_processed['Combined_Text'].str.strip() != '']

    return df_processed


# ===================================================================
# メイン処理
# ===================================================================

def main():
    """メイン処理関数"""

    # 初期設定（デフォルトのデータセットタイプ）

    # ページ設定
    try:
        st.set_page_config(
            page_title="非Q&A型RAGデータ処理",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except st.errors.StreamlitAPIException:
        pass

    # サイドバー設定
    with st.sidebar:
        st.title("📚 非Q&A型データ処理")
        st.markdown("---")

        # データセットタイプ選択
        st.subheader("📊 データセットタイプ選択")

        dataset_options = NonQARAGConfig.get_all_datasets()
        dataset_labels = {
            dt: f"{NonQARAGConfig.get_config(dt)['icon']} {NonQARAGConfig.get_config(dt)['name']}"
            for dt in dataset_options
        }

        selected_dataset = st.selectbox(
            "処理するデータセットタイプ",
            options=dataset_options,
            format_func=lambda x: dataset_labels[x],
            help="処理したいデータセットのタイプを選択してください"
        )

        # データセット設定を取得
        dataset_config = NonQARAGConfig.get_config(selected_dataset)

        # データセット情報表示
        st.info(f"""
        **選択中のデータセット:**
        - タイプ: {dataset_config['name']}
        - 主要フィールド: {dataset_config['text_field']}
        - HuggingFace: {dataset_config['hf_dataset']}
        """)

        # モデル選択
        st.divider()
        selected_model = select_model()
        show_model_info(selected_model)

        # データセット固有のオプション
        st.divider()
        st.subheader("⚙️ データセット固有設定")

        dataset_specific_options = {}

        if selected_dataset == "wikipedia_ja":
            dataset_specific_options['remove_markup'] = st.checkbox(
                "Wikiマークアップを除去",
                value=True,
                help="[[リンク]]や==見出し==などを除去"
            )
            dataset_specific_options['min_text_length'] = st.number_input(
                "最小テキスト長",
                min_value=50,
                value=200,
                help="この長さ未満の記事を除外"
            )

        elif selected_dataset == "japanese_text":
            dataset_specific_options['remove_urls'] = st.checkbox(
                "URLを除去",
                value=True,
                help="テキスト中のURLを除去"
            )
            dataset_specific_options['min_text_length'] = st.number_input(
                "最小テキスト長",
                min_value=10,
                value=10,
                help="この長さ未満のテキストを除外"
            )

        elif selected_dataset == "cc_news":
            dataset_specific_options['remove_urls'] = st.checkbox(
                "URLを除去",
                value=True,
                help="テキスト中のURLを除去"
            )
            dataset_specific_options['min_text_length'] = st.number_input(
                "最小テキスト長",
                min_value=50,
                value=100,
                help="この長さ未満のテキストを除外"
            )

    # メインコンテンツ
    st.title(f"{dataset_config['icon']} {dataset_config['name']}前処理アプリ")
    st.caption("RAG（Retrieval-Augmented Generation）用データ前処理 - 非Q&A型データセット対応")
    st.markdown("---")

    # 使い方例をExpanderで表示
    with st.expander("📖 **使い方例**", expanded=False):
        st.markdown(f"""
        ### 🎯 基本的な使い方

        1. **左ペインで設定**
           - 📊 **データセットタイプを選択** ({dataset_config['name']})
           - 🤖 使用するモデルを選択
           - ⚙️ データセット固有の設定を調整

        2. **右ペインで処理実行**

           **📁 データアップロード**
           - CSVファイルをアップロード、または
           - HuggingFaceから自動ダウンロード
             - 推奨データセット: `{dataset_config['hf_dataset']}`
             - Split: `{dataset_config['split']}`
             - サンプル数: {dataset_config['sample_size']}件程度

           **🔍 データ検証**
           - データの品質をチェック
           - 必須フィールドの確認
           - データセット固有の検証結果を確認

           **⚙️ 前処理実行**
           - テキスト抽出・結合
           - クレンジング処理
           - 「🚀 前処理を実行」をクリック

           **📊 結果・ダウンロード**
           - 処理済みデータをCSV、TXT、JSON形式でダウンロード
           - OUTPUTフォルダに保存

        ### 💡 ヒント
        - 非Q&A型データなので、タイトル・本文・要旨などを適切に結合します
        - データセットによっては大量のテキストが含まれるため、サンプル数を調整してください
        - HuggingFaceからダウンロードしたデータは`datasets/`フォルダに保存されます
        """)

    # タブ設定
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 データアップロード",
        "🔍 データ検証",
        "⚙️ 前処理実行",
        "📊 結果・ダウンロード"
    ])

    # Tab 1: データアップロード
    with tab1:
        st.header("データファイルのアップロード")

        # ファイルアップロード
        uploaded_file = st.file_uploader(
            f"{dataset_config['name']}のCSVファイルを選択",
            type=['csv'],
            help=f"主要フィールド: {dataset_config['text_field']}"
        )

        # HuggingFaceから自動ロード
        st.divider()

        if selected_dataset == "custom":
            st.info("📁 カスタムデータセットが選択されています。上記のファイルアップロードを使用してください。")
        else:
            st.subheader("または、HuggingFaceから自動ロード")

            # デフォルト設定を表示
            st.info(f"""
            **推奨設定:**
            - データセット名: `{dataset_config['hf_dataset']}`
            - Config: `{dataset_config['hf_config'] or 'なし'}`
            - Split: `{dataset_config.get('split', 'train')}`
            - サンプル数: {dataset_config['sample_size']}件
            """)

            dataset_name = st.text_input(
                "HuggingFaceデータセット名",
                value=dataset_config['hf_dataset'] or "",
                placeholder="例: wikimedia/wikipedia または range3/cc100-ja"
            )

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                split_name = st.text_input("Split名",
                                          value=dataset_config.get('split', 'train'),
                                          placeholder="train")
            with col2:
                sample_size = st.number_input("サンプル数", min_value=10,
                                            value=dataset_config['sample_size'])
            with col3:
                config_name = st.text_input("Config名",
                                          value=dataset_config['hf_config'] or "",
                                          placeholder="任意")

            if st.button("📥 HuggingFaceからロード", type="primary"):
                try:
                    from datasets import load_dataset as hf_load_dataset

                    # 入力値の検証
                    if not dataset_name:
                        st.error("データセット名を入力してください")
                        st.stop()

                    if not split_name:
                        split_name = "train"

                    with st.spinner(f"HuggingFaceから{dataset_name}をダウンロード中..."):
                        # ストリーミングモードで確実にロード
                        samples = []

                        # Livedoorコーパスの特別処理
                        if selected_dataset == "livedoor":
                            st.info("📥 Livedoorニュースコーパスをダウンロード中...")

                            # ダウンロードと解凍
                            with st.spinner("ダウンロードと解凍中..."):
                                data_dir = download_livedoor_corpus("datasets")

                            # データ読み込み
                            with st.spinner("データを読み込み中..."):
                                df = load_livedoor_corpus(data_dir)

                            # サンプリング（必要に応じて）
                            if sample_size < len(df):
                                df = df.sample(n=sample_size, random_state=42)
                                st.info(f"📊 {len(df)}件にサンプリングしました")

                        # 動作確認済みのデータセットのみ処理
                        elif dataset_name == "wikimedia/wikipedia" or dataset_name == "wikipedia":
                            # Wikipedia日本語版
                            actual_dataset = "wikimedia/wikipedia"
                            actual_config = config_name if config_name else "20231101.ja"

                            st.info(f"📥 {actual_dataset}をロード中 (config: {actual_config})...")
                            dataset = hf_load_dataset(actual_dataset, actual_config, split=split_name, streaming=True)

                            # サンプリング
                            progress_bar = st.progress(0)
                            for i, item in enumerate(dataset):
                                if i >= sample_size:
                                    break
                                samples.append(item)
                                progress_bar.progress((i + 1) / sample_size)

                            df = pd.DataFrame(samples)
                            progress_bar.empty()

                        elif dataset_name == "range3/cc100-ja":
                            # CC100日本語
                            st.info(f"📥 {dataset_name}をロード中...")
                            dataset = hf_load_dataset(dataset_name, split=split_name, streaming=True)

                            # サンプリング
                            progress_bar = st.progress(0)
                            for i, item in enumerate(dataset):
                                if i >= sample_size:
                                    break
                                samples.append(item)
                                progress_bar.progress((i + 1) / sample_size)

                            df = pd.DataFrame(samples)
                            progress_bar.empty()

                        elif dataset_name == "cc_news":
                            # CC-News（動作確認済み）
                            st.info(f"📥 {dataset_name}をロード中...")
                            if config_name:
                                dataset = hf_load_dataset(dataset_name, config_name, split=split_name, streaming=True)
                            else:
                                dataset = hf_load_dataset(dataset_name, split=split_name, streaming=True)

                            # サンプリング
                            progress_bar = st.progress(0)
                            for i, item in enumerate(dataset):
                                if i >= sample_size:
                                    break
                                samples.append(item)
                                progress_bar.progress((i + 1) / sample_size)

                            df = pd.DataFrame(samples)
                            progress_bar.empty()

                        else:
                            # その他のデータセット（非推奨）
                            st.warning("⚠️ このデータセットは動作保証外です")
                            if config_name:
                                dataset = hf_load_dataset(dataset_name, config_name, split=split_name, streaming=True)
                            else:
                                dataset = hf_load_dataset(dataset_name, split=split_name, streaming=True)

                            # サンプリング
                            progress_bar = st.progress(0)
                            for i, item in enumerate(dataset):
                                if i >= sample_size:
                                    break
                                samples.append(item)
                                progress_bar.progress((i + 1) / sample_size)

                            df = pd.DataFrame(samples)
                            progress_bar.empty()

                    # datasetsフォルダに保存
                    if df is not None and len(df) > 0:
                        datasets_dir = Path("datasets")
                        datasets_dir.mkdir(exist_ok=True)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_dataset_name = dataset_name.replace("/", "_").replace("-", "_")
                        csv_filename = f"{safe_dataset_name}_{split_name}_{sample_size}_{timestamp}.csv"
                        csv_path = datasets_dir / csv_filename

                        df.to_csv(csv_path, index=False)
                        st.info(f"📂 データをdatasets/{csv_filename}に保存しました")

                        # メタデータ保存
                        metadata = {
                            'dataset_name' : dataset_name,
                            'dataset_type' : selected_dataset,
                            'config'       : config_name,
                            'split'        : split_name,
                            'sample_size'  : sample_size,
                            'actual_size'  : len(df),
                            'downloaded_at': datetime.now().isoformat(),
                            'columns'      : df.columns.tolist()
                        }

                        metadata_filename = f"{safe_dataset_name}_{split_name}_{sample_size}_{timestamp}_metadata.json"
                        metadata_path = datasets_dir / metadata_filename

                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2)

                        st.session_state['uploaded_data'] = df
                        st.session_state['uploaded_columns'] = df.columns.tolist()
                        st.success(f"✅ {len(df)}件のデータをロードし、datasets/フォルダに保存しました")

                except Exception as e:
                    error_msg = str(e)
                    if "Dataset scripts are no longer supported" in error_msg:
                        st.error("❌ このデータセットはスクリプトベースで廃止されています")
                        st.info("""
                        💡 **動作確認済みのデータセットをご利用ください：**
                        - `wikimedia/wikipedia` (Config: 20231101.ja)
                        - `range3/cc100-ja`
                        """)
                    elif "doesn't exist on the Hub" in error_msg:
                        st.error("❌ データセットが見つかりません")
                        st.info("""
                        💡 **データセット名を確認してください。推奨：**
                        - `wikimedia/wikipedia`
                        - `range3/cc100-ja`
                        """)
                    else:
                        st.error(f"データセットのロードに失敗しました: {error_msg}")
                        st.info("💡 データセット名、config名、split名を確認してください")

        # アップロードされたデータの処理
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_data'] = df
            st.session_state['uploaded_columns'] = df.columns.tolist()
            st.success(f"✅ ファイルをアップロードしました: {uploaded_file.name}")

        # データプレビュー
        if 'uploaded_data' in st.session_state:
            df = st.session_state['uploaded_data']
            st.subheader("📋 データプレビュー")
            st.info(f"データ件数: {len(df)}件 | カラム数: {len(df.columns)}列")
            st.dataframe(df.head(10), width='stretch')

            # カラム情報
            with st.expander("📊 カラム詳細"):
                col_info = pd.DataFrame({
                    'カラム名'  : df.columns,
                    'データ型'  : df.dtypes.astype(str),
                    '非NULL数'  : df.count(),
                    'NULL数'    : df.isnull().sum(),
                    'ユニーク数': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, width='stretch')

    # Tab 2: データ検証
    with tab2:
        st.header("データ品質チェック")

        if 'uploaded_data' not in st.session_state:
            st.warning("⚠️ まずデータをアップロードしてください")
        else:
            df = st.session_state['uploaded_data']

            # 基本検証
            st.subheader("📋 基本検証")
            issues = validate_data(df, selected_dataset)

            # データセット固有の検証
            st.subheader(f"🔍 {dataset_config['name']}固有の検証")

            if selected_dataset == "wikipedia_ja":
                specific_issues = validate_wikipedia_data_specific(df)
            elif selected_dataset in ["japanese_text", "cc_news", "livedoor"]:
                specific_issues = validate_news_data_specific(df, selected_dataset)
            else:
                # その他のデータセットの検証
                specific_issues = []
                if 'text' in df.columns:
                    specific_issues.append("✅ テキストフィールドを検出しました")

            issues.extend(specific_issues)

            # 検証結果の表示
            if issues:
                for issue in issues:
                    if "⚠️" in issue or "❌" in issue:
                        st.warning(issue)
                    elif "✅" in issue:
                        st.success(issue)
                    else:
                        st.info(issue)

            # テキストフィールドのサンプル表示
            st.subheader("📝 テキストコンテンツのサンプル")
            text_field = dataset_config['text_field']
            if text_field in df.columns:
                with st.expander(f"{text_field} のサンプル（最初の3件）"):
                    for i, value in enumerate(df[text_field].head(3), 1):
                        st.text(f"[{i}] {str(value)[:500]}...")  # 最初の500文字のみ表示

    # Tab 3: 前処理実行
    with tab3:
        st.header("RAG用データ前処理")

        if 'uploaded_data' not in st.session_state:
            st.warning("⚠️ まずデータをアップロードしてください")
        else:
            df = st.session_state['uploaded_data']

            # 前処理オプション
            st.subheader("⚙️ 前処理設定")

            col1, col2 = st.columns(2)
            with col1:
                remove_short_text = st.checkbox(
                    "短いテキストを除外",
                    value=True,
                    help="指定文字数未満のテキストを除外"
                )
                if remove_short_text:
                    min_length = st.number_input(
                        "最小文字数",
                        min_value=10,
                        value=100,
                        help="この文字数未満は除外"
                    )

            with col2:
                remove_duplicates = st.checkbox(
                    "重複を除去",
                    value=True,
                    help="完全に同じテキストを除外"
                )

            # 処理実行ボタン
            if st.button("🚀 前処理を実行", type="primary"):
                with st.spinner("処理中..."):
                    try:
                        # テキスト抽出
                        df_processed = extract_text_content(df, selected_dataset)

                        # 短いテキストの除外
                        if remove_short_text:
                            before_len = len(df_processed)
                            df_processed = df_processed[
                                df_processed['Combined_Text'].str.len() >= min_length
                                ]
                            removed = before_len - len(df_processed)
                            if removed > 0:
                                st.info(f"📊 {removed}件の短いテキストを除外しました")

                        # 重複除去
                        if remove_duplicates:
                            before_len = len(df_processed)
                            df_processed = df_processed.drop_duplicates(subset=['Combined_Text'])
                            removed = before_len - len(df_processed)
                            if removed > 0:
                                st.info(f"📊 {removed}件の重複テキストを除外しました")

                        # インデックスリセット
                        df_processed = df_processed.reset_index(drop=True)

                        # セッションに保存
                        st.session_state['processed_data'] = df_processed
                        st.session_state['processing_config'] = {
                            'dataset_type'     : selected_dataset,
                            'options'          : dataset_specific_options,
                            'remove_short_text': remove_short_text,
                            'min_length'       : min_length if remove_short_text else 0,
                            'remove_duplicates': remove_duplicates
                        }

                        st.success(f"✅ 前処理が完了しました！（{len(df_processed)}件）")

                    except Exception as e:
                        st.error(f"前処理エラー: {str(e)}")
                        logger.error(f"前処理エラー: {e}")

            # 処理済みデータのプレビュー
            if 'processed_data' in st.session_state:
                df_processed = st.session_state['processed_data']

                st.subheader("📋 処理済みデータプレビュー")
                st.dataframe(df_processed[['Combined_Text']].head(10), width='stretch')

                # トークン使用量推定
                st.subheader("💰 トークン使用量推定")
                estimate_token_usage(df_processed, selected_model)

                # テキスト長の分布
                st.subheader("📊 テキスト長の分布")
                text_lengths = df_processed['Combined_Text'].str.len()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("平均", f"{text_lengths.mean():.0f}文字")
                with col2:
                    st.metric("最小", f"{text_lengths.min():,}文字")
                with col3:
                    st.metric("最大", f"{text_lengths.max():,}文字")
                with col4:
                    st.metric("中央値", f"{text_lengths.median():.0f}文字")

    # Tab 4: 結果・ダウンロード
    with tab4:
        st.header("処理結果とダウンロード")

        if 'processed_data' not in st.session_state:
            st.warning("⚠️ まず前処理を実行してください")
        else:
            df_processed = st.session_state['processed_data']
            config = st.session_state['processing_config']

            # 処理サマリー
            st.subheader("📊 処理サマリー")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("処理件数", f"{len(df_processed):,}件")
            with col2:
                original_count = len(st.session_state.get('uploaded_data', []))
                removed = original_count - len(df_processed)
                st.metric("除外件数", f"{removed:,}件")
            with col3:
                retention_rate = (len(df_processed) / original_count * 100) if original_count > 0 else 0
                st.metric("残存率", f"{retention_rate:.1f}%")

            # ファイルダウンロード
            st.subheader("📥 ファイルダウンロード")

            # CSVデータ作成
            csv_buffer = io.StringIO()
            df_processed.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            # テキストデータ作成
            text_data = '\n'.join(df_processed['Combined_Text'].dropna().astype(str))

            # メタデータ作成
            metadata = {
                'dataset_type'  : config['dataset_type'],
                'dataset_name'  : dataset_config['name'],
                'processed_at'  : datetime.now().isoformat(),
                'row_count'     : len(df_processed),
                'original_count': original_count,
                'removed_count' : removed,
                'config'        : config
            }

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    label="📄 CSVファイル",
                    data=csv_data,
                    file_name=f"preprocessed_{config['dataset_type']}.csv",
                    mime="text/csv"
                )

            with col2:
                st.download_button(
                    label="📝 テキストファイル",
                    data=text_data,
                    file_name=f"{config['dataset_type']}.txt",
                    mime="text/plain"
                )

            with col3:
                st.download_button(
                    label="📋 メタデータ(JSON)",
                    data=json.dumps(metadata, ensure_ascii=False, indent=2),
                    file_name=f"metadata_{config['dataset_type']}.json",
                    mime="application/json"
                )

            # OUTPUTフォルダへの保存
            st.divider()
            if st.button("💾 OUTPUTフォルダに保存", type="primary"):
                saved_files = save_files_to_output(
                    df_processed,
                    config['dataset_type'],
                    csv_data,
                    text_data
                )

                if saved_files:
                    st.success("✅ ファイルを保存しました：")
                    for file_type, file_path in saved_files.items():
                        st.write(f"• {file_path}")
                else:
                    st.error("❌ ファイル保存に失敗しました")

            # データサンプル表示
            st.divider()
            st.subheader("📝 処理済みテキストのサンプル")
            for i, text in enumerate(df_processed['Combined_Text'].head(3), 1):
                with st.expander(f"サンプル {i}"):
                    st.text(str(text)[:1000] + "..." if len(str(text)) > 1000 else str(text))


if __name__ == "__main__":
    main()
