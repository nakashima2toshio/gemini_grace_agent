#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qdrant_search_page.py - Qdrant検索ページ
========================================
Qdrantベクトルデータベースを使用した意味検索

機能:
- コレクション検索
- 埋め込みベクトル生成
- AI応答生成
"""

import warnings
import pandas as pd
import streamlit as st
from helper_llm import create_llm_client
from qdrant_client import QdrantClient

# サービスモジュールからインポート
from services.qdrant_service import (
    QdrantDataFetcher,
    embed_query_for_search,
    get_dynamic_collection_mapping,
    get_collection_embedding_params,
)
from services.file_service import load_source_qa_data
from qdrant_client_wrapper import search_collection, embed_sparse_query_unified # Import search_collection and embed_sparse_query_unified

def show_qdrant_search_page():
    """画面5: Qdrant検索"""
    st.title("🔎 Qdrant検索")
    st.caption("Qdrantベクトルデータベースを使用した意味検索")

    # Qdrant接続確認
    qdrant_url = "http://localhost:6333"
    client = None
    available_collections = []

    try:
        client = QdrantClient(url=qdrant_url)
        collections_response = client.get_collections()
        available_collections = [col.name for col in collections_response.collections]
    except Exception:
        st.error(f"❌ Qdrantサーバーに接続できません: {qdrant_url}")
        st.warning("Qdrantサーバーが起動していることを確認してください")
        st.code("python server.py", language="bash")
        st.caption("または")
        st.code("docker run -p 6333:6333 qdrant/qdrant", language="bash")
        return

    # コレクションとCSVファイルの対応表を表示（動的取得）
    st.subheader("📊 コレクションとCSVファイルの対応")
    
    # 動的マッピングの取得
    dynamic_mapping = get_dynamic_collection_mapping(client)
    
    if dynamic_mapping:
        mapping_data = []
        for collection, csv_file in dynamic_mapping.items():
            mapping_data.append(
                {
                    "コレクション名": collection,
                    "CSVファイル": csv_file,
                    "ファイルパス": f"qa_output/{csv_file}",
                }
            )
        mapping_df = pd.DataFrame(mapping_data)
        st.table(mapping_df)
    else:
        st.info("コレクションとCSVファイルの対応情報はありません（コレクションが存在しないか、命名規則が一致しません）")

    st.divider()

    if not available_collections:
        st.warning("利用可能なコレクションがありません")
        st.info("先に「Qdrant登録」でデータを登録してください")
        return

    # サイドバー：検索設定
    with st.sidebar:
        st.header("🔧 検索設定")

        # コレクション選択
        collection = st.selectbox(
            "コレクション",
            options=available_collections,
            help="検索対象のコレクションを選択",
        )

        # コレクション情報表示
        if client and collection:
            col_info = get_collection_embedding_params(client, collection)
            st.info(f"📊 {col_info['model']} ({col_info['dims']}次元)")

        # Top-K設定
        topk = st.slider(
            "検索結果数（Top-K）", min_value=1, max_value=20, value=5, step=1
        )
        
        # ハイブリッド検索の有効化トグル
        use_hybrid_search = st.checkbox("⚙️ ハイブリッド検索を有効にする (Sparse + Dense)", value=False)

        # デバッグモード
        debug_mode = st.checkbox("🐛 デバッグモード", value=False)

    # メインエリア
    # セッション状態の初期化
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    # コレクションデータプレビューセクション
    with st.expander("📋 コレクションデータプレビュー", expanded=False):
        # QdrantDataFetcherインスタンスを作成
        try:
            client = QdrantClient(url=qdrant_url)
            data_fetcher = QdrantDataFetcher(client)

            # fetch_collection_source_infoを使用してソース情報を取得
            source_info = data_fetcher.fetch_collection_source_info(collection)

            if "error" not in source_info:
                sources = source_info.get("sources", {})

                if sources:
                    st.caption(f"コレクション: **{collection}**")

                    # 各ソースファイルごとにエキスパンダーを作成
                    for source, stats in sorted(sources.items()):
                        with st.expander(f"📄 {source}", expanded=False):
                            st.markdown(
                                f"- 推定データ数: {stats['estimated_total']:,}件 ({stats['percentage']:.1f}%)"
                            )
                            st.markdown(f"- 生成方法: `{stats['method']}`")
                            st.markdown(f"- ドメイン: `{stats['domain']}`")

                            # question, answerテーブルを表示
                            df_qa = load_source_qa_data(source, num_rows=20)
                            if df_qa is not None:
                                st.dataframe(
                                    df_qa, width='stretch', hide_index=True
                                )
                            else:
                                st.info(f"データを読み込めません: qa_output/{source}")
                else:
                    st.info("データソース情報が見つかりません")
            else:
                st.error(f"エラー: {source_info['error']}")
        except Exception as e:
            st.error(f"データ取得エラー: {str(e)}")

    st.divider()

    # 検索入力
    st.subheader("🔍 検索")
    query = st.text_input(
        "検索クエリを入力してください",
        value=st.session_state.search_query,
        placeholder="検索したい質問を入力してください",
    )

    col_search, col_clear = st.columns([4, 1])
    with col_search:
        do_search = st.button("🔍 検索実行", type="primary", width='stretch')
    with col_clear:
        if st.button("🗑️ クリア", width='stretch'):
            st.session_state.search_query = ""
            st.rerun()

    # 検索実行
    if do_search and query.strip():
        try:
            client = QdrantClient(url=qdrant_url)

            # コレクションに対応した埋め込み設定を取得
            collection_config = get_collection_embedding_params(client, collection)
            embedding_model = collection_config["model"]
            embedding_dims = collection_config.get("dims")

            if debug_mode:
                st.info(f"🔍 使用モデル: {embedding_model} ({embedding_dims}次元)")
                try:
                    # コレクション設定のデバッグ表示
                    col_info_debug = client.get_collection(collection)
                    st.markdown("**📋 コレクション設定 (Debug):**")
                    st.json(col_info_debug.model_dump() if hasattr(col_info_debug, 'model_dump') else col_info_debug.dict())
                except Exception as e:
                    st.error(f"コレクション設定の取得に失敗: {e}")

            # クエリを埋め込みベクトルに変換
            with st.spinner("埋め込みベクトルを生成中..."):
                qvec = embed_query_for_search(query, embedding_model, embedding_dims)
                if debug_mode:
                    st.success(f"✅ {len(qvec)}次元のベクトルを生成しました")

            # Qdrantで検索
            with st.spinner("検索中..."):
                sparse_vector = None
                if use_hybrid_search:
                    with st.spinner("Sparseベクトルを生成中..."):
                        # sparse_vector生成
                        sparse_vector = embed_sparse_query_unified(query)
                        if debug_mode:
                            st.success("✅ Sparseベクトルを生成しました")
                
                # search_collection関数を呼び出し
                hits_dict_list = search_collection( # search_collection returns List[Dict[str, Any]]
                    client=client,
                    collection_name=collection,
                    query_vector=qvec,
                    sparse_vector=sparse_vector if use_hybrid_search else None, # ハイブリッド検索が有効な場合のみSparseベクトルを渡す
                    limit=topk
                )
            
            # search_collectionの戻り値はDictのリストなので、QdrantのPointStructに変換 (UI表示のため)
            class MockHit: # 既存のUI表示ロジックに合わせるため
                def __init__(self, hit_dict):
                    self.score = hit_dict.get("score", 0.0)
                    self.id = hit_dict.get("id")
                    self.payload = hit_dict.get("payload")
            
            hits = [MockHit(h) for h in hits_dict_list]

            

            # 検索結果を表示
            st.divider()
            st.subheader(f"📊 検索結果 (Top {len(hits)})")

            if not hits:
                st.warning("検索結果が見つかりませんでした")
                return

            # 結果をDataFrameに変換
            rows = []
            for h in hits:
                row_data = {
                    "スコア": f"{h.score:.4f}",
                    "質問": h.payload.get("question", "N/A") if h.payload else "N/A",
                    "回答": h.payload.get("answer", "N/A") if h.payload else "N/A",
                    "ソース": h.payload.get("source", "N/A") if h.payload else "N/A",
                }
                rows.append(row_data)

            df_results = pd.DataFrame(rows)
            st.dataframe(df_results, width='stretch', hide_index=True)

            # 最高スコアの結果を詳細表示
            if hits:
                best_hit = hits[0]
                st.divider()
                st.subheader("🏆 最高スコアの結果")

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("スコア", f"{best_hit.score:.4f}")
                with col2:
                    if best_hit.payload:
                        source = best_hit.payload.get("source", "N/A")
                        st.caption(f"ソース: {source}")

                if best_hit.payload:
                    question = best_hit.payload.get("question", "")
                    answer = best_hit.payload.get("answer", "")

                    st.markdown("**質問:**")
                    st.info(question)

                    st.markdown("**回答:**")
                    st.success(answer)

                    # Geminiによる日本語応答生成
                    st.divider()
                    st.subheader("🧠 AI応答（Gemini）")

                    qa_prompt = (
                        "以下の検索結果とユーザーの質問を踏まえて、日本語で簡潔かつ正確に回答してください。\n\n"
                        f"ユーザーの質問:\n{query}\n\n"
                        f"検索結果のスコア: {best_hit.score:.4f}\n"
                        f"検索結果の質問: {question}\n"
                        f"検索結果の回答: {answer}\n"
                    )

                    with st.expander("📝 プロンプト詳細", expanded=False):
                        st.code(qa_prompt)

                    try:
                        with st.spinner("Gemini AIが回答を生成中..."):
                            llm_client = create_llm_client(provider="gemini")
                            generated_answer = llm_client.generate_content(
                                prompt=qa_prompt,
                                model="gemini-2.0-flash"
                            )

                        if generated_answer and generated_answer.strip():
                            st.markdown("**AI応答:**")
                            st.write(generated_answer)
                        else:
                            st.info("応答テキストを取得できませんでした")
                    except Exception as gen_err:
                        st.error(f"AI応答生成に失敗しました: {str(gen_err)}")
                        if debug_mode:
                            st.exception(gen_err)

        except Exception as e:
            st.error(f"❌ エラーが発生しました: {str(e)}")
            if debug_mode:
                st.exception(e)

            if "Connection refused" in str(e):
                st.warning("Qdrantサーバーが起動していることを確認してください")
                st.code("python server.py", language="bash")
            elif "collection" in str(e).lower() and "not found" in str(e).lower():
                st.warning(f"コレクション '{collection}' が見つかりません")
                st.info("「Qdrant登録」でデータを登録してください")