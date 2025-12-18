#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a50_rag_search_local_qdrant.py — Qdrant RAG検索用Streamlit UI
------------------------------------------------------------------------------
機能概要:
  - 複数コレクション対応（product_embeddings, qa_corpus, qa_cc_news_*, qa_livedoor_*, raw_*等）
  - ドメイン別検索（customer, medical, sciq, legal, trivia）
  - Named Vectors切替（ada-002, 3-small等）
  - 動的な埋め込み次元対応（384次元、1536次元）
  - リアルタイム類似度スコア表示
  - OpenAI GPT-4o-miniによる日本語回答生成

対応データセット:
  - カスタマーサポート・FAQ (customer)
  - 医療QAデータ (medical)
  - 科学・技術QA (sciq)
  - 法律・判例QA (legal)
  - TriviaQA（トリビアQA） (trivia)
  - CC News データセット (raw_cc_news, qa_cc_news_a02_llm, qa_cc_news_a03_rule, qa_cc_news_a10_hybrid)
    - raw_cc_news: CC Newsの生データ（7,376件）
    - qa_cc_news_a02_llm: LLM生成方式によるQ&Aペア（1,344件）
    - qa_cc_news_a03_rule: ルールベース生成方式によるQ&Aペア（1,638件）
    - qa_cc_news_a10_hybrid: ハイブリッド生成方式によるQ&Aペア（1,638件）
  - Livedoor News データセット (raw_livedoor, qa_livedoor_a02_20_llm, qa_livedoor_a03_rule, qa_livedoor_a10_hybrid)
    - raw_livedoor: Livedoorの生データ（7,376件）
    - qa_livedoor_a02_20_llm: LLM生成方式によるQ&Aペア（1,317件）
    - qa_livedoor_a03_rule: ルールベース生成方式によるQ&Aペア（1,638件）
    - qa_livedoor_a10_hybrid: ハイブリッド生成方式によるQ&Aペア（1,638件）

起動: streamlit run a50_rag_search_local_qdrant.py --server.port=8504
"""
import os
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st

try:
    import yaml
except Exception:
    yaml = None

from qdrant_client import QdrantClient
from qdrant_client.http import models
from helper_embedding import create_embedding_client
from helper_llm import create_llm_client

# デフォルト設定の定義（config.ymlが存在しない場合のフォールバック）
DEFAULTS = {
    "rag": {"collection": "product_embeddings"},  # デフォルトコレクション
    "embeddings": {
        "primary": {"provider": "openai", "model": "text-embedding-3-small", "dims": 1536},
        "ada-002": {"provider": "openai", "model": "text-embedding-ada-002", "dims": 1536},
        "3-small": {"provider": "openai", "model": "text-embedding-3-small", "dims": 1536},
    },
    "qdrant": {"url": "http://localhost:6333"},  # Qdrantサーバーのデフォルトアドレス
}

# コレクション固有の埋め込み設定（モデルと次元数を指定）
COLLECTION_EMBEDDINGS = {
    "product_embeddings": {"model": "text-embedding-3-small", "dims": 384},  # 製品情報用：384次元で高速処理
    "qa_corpus": {"model": "text-embedding-3-small", "dims": 1536},  # Q&Aコーパス用：1536次元で高精度
    "raw_cc_news": {"model": "text-embedding-3-small", "dims": 1536},  # CC News 生データ
    "raw_livedoor": {"model": "text-embedding-3-small", "dims": 1536},  # Livedoor 生データ
    "qa_cc_news_a02_llm": {"model": "text-embedding-3-small", "dims": 1536},  # CC News LLM生成方式
    "qa_cc_news_a03_rule": {"model": "text-embedding-3-small", "dims": 1536},  # CC News ルールベース生成方式
    "qa_cc_news_a10_hybrid": {"model": "text-embedding-3-small", "dims": 1536},  # CC News ハイブリッド生成方式
    "qa_livedoor_a02_20_llm": {"model": "text-embedding-3-small", "dims": 1536},  # Livedoor LLM生成方式
    "qa_livedoor_a03_rule": {"model": "text-embedding-3-small", "dims": 1536},  # Livedoor ルールベース生成方式
    "qa_livedoor_a10_hybrid": {"model": "text-embedding-3-small", "dims": 1536},  # Livedoor ハイブリッド生成方式
}

def load_config(path="config.yml") -> Dict[str, Any]:
    """
    設定ファイルを読み込み、DEFAULTSとマージする

    Args:
        path: 設定ファイルのパス（デフォルトはconfig.yml）

    Returns:
        マージされた設定辞書
    """
    cfg = {}
    if yaml and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    full = DEFAULTS.copy()
    # 浅いマージ（第1階層の辞書は更新、それ以外は上書き）
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(full.get(k), dict):
            full[k].update(v)
        else:
            full[k] = v
    return full

def embed_query(text: str, model: str = "gemini-embedding-001", dims: Optional[int] = None) -> List[float]:
    """
    クエリテキストを埋め込みベクトルに変換（Gemini API使用）

    Args:
        text: 埋め込むテキスト
        model: 使用する埋め込みモデル（デフォルト: gemini-embedding-001）
        dims: ベクトルの次元数（Geminiでは3072次元）

    Returns:
        埋め込みベクトル（float配列）
    """
    embedding_client = create_embedding_client(provider="gemini")
    return embedding_client.embed_text(text)

st.set_page_config(page_title="Qdrant RAG UI", page_icon="🔎", layout="wide")
st.title("🔎 Qdrant RAG UI (domain filter / named vectors)")

cfg = load_config("config.yml")
default_collection = cfg.get("rag", {}).get("collection", "product_embeddings")
embeddings_cfg: Dict[str, Dict[str, Any]] = cfg.get("embeddings", {})
qdrant_url = (cfg.get("qdrant", {}) or {}).get("url", "http://localhost:6333")

# Fetch available collections from Qdrant
available_collections = []
try:
    temp_client = QdrantClient(url=qdrant_url)
    collections_response = temp_client.get_collections()
    available_collections = [col.name for col in collections_response.collections]
    # Sort collections with default_collection first
    if default_collection in available_collections:
        available_collections.remove(default_collection)
        available_collections.insert(0, default_collection)
except Exception:
    available_collections = [default_collection]  # Fallback to default if can't connect

# 各ドメイン・コレクション用のサンプル質問
SAMPLE_QUESTIONS = {
    "customer": [
        "返金は可能ですか？",
        "配送にはどのくらい時間がかかりますか？",
        "アカウントを作成するにはどうすればよいですか？"
    ],
    "medical": [
        "副作用はありますか？",
        "心房細動の症状について教えてください",
        "糖尿病の管理方法は何ですか？"
    ],
    "legal": [
        "Googleは私が作成したコンテンツに基づいて新しいコンテンツを作成することが許可されていますか？",
        "ユーザー生成コンテンツの修正からなる派生作品を作成することはGoogleの法的権利内ですか？",
        "Googleは常に私のコンテンツをGoogleアカウントから転送することを許可しますか？"
    ],
    "sciq": [
        "チーズやヨーグルトなどの食品の調製に一般的に使用される生物のタイプは何ですか？",
        "放射性崩壊の最も危険性の低いタイプは何ですか？",
        "物質が酸素と迅速に反応するときに起こる反応の種類は何ですか？"
    ],
    "trivia": [
        "日本で一番高い山は何ですか？",
        "アメリカの初代大統領は誰ですか？",
        "太陽系で最も大きな惑星は何ですか？",
        "東京オリンピックは何年に開催されましたか？",
        "世界で最も長い川は何ですか？"
    ],
    "cc_news": [
        "Which Boston Ballet dancers starred in their Super Bowl video?",
        "What role does The Nutcracker play for ballet companies beyond being a performance piece?",
        "What insight did the speaker gain from seeing Robert's McLaren documentary?",
        "Under what circumstance can a firm still charge a fee for an SAR under GDPR?",
        "Which two technologies did Vyas use to illustrate a 5G use case?"
    ],
    "livedoor": [
        "ライブドアニュースについて教えてください",
        "最新のテクノロジーニュースは？",
        "スポーツニュースで話題になっていることは？",
        "エンタメ関連のニュースを知りたい",
        "経済ニュースの最新情報は？"
    ]
}

with st.sidebar:
    st.header("Settings")
    
    # Collection selector
    if available_collections:
        # Default collection is already at index 0 due to sorting
        collection = st.selectbox("Collection", options=available_collections, index=0)
    else:
        collection = st.text_input("Collection", value=default_collection)
    
    # Show collection info
    if collection in COLLECTION_EMBEDDINGS:
        col_info = COLLECTION_EMBEDDINGS[collection]
        st.info(f"📊 Collection '{collection}' uses {col_info['model']} with {col_info['dims']} dimensions")
    
    vec_name = st.selectbox("Using vector (named)", options=list(embeddings_cfg.keys()))
    model_for_using = embeddings_cfg[vec_name]["model"]
    
    # Check if collection supports domain filtering
    supports_domain = collection in ["qa_corpus"]  # Add collections that support domain filtering
    
    # Show domain selector based on collection support
    if supports_domain:
        domain = st.selectbox("Domain", options=["ALL", "customer", "medical", "legal", "sciq", "trivia"])
    else:
        # For collections without domain field, force ALL
        st.info(f"ℹ️ Collection '{collection}' doesn't support domain filtering. Using ALL.")
        domain = "ALL"
    
    topk = st.slider("TopK", min_value=1, max_value=20, value=5, step=1)
    qdrant_url_input = st.text_input("Qdrant URL", value=qdrant_url)
    debug_mode = st.checkbox("🐛 Debug Mode", value=False)
    
    # サンプル質問セクション
    st.markdown("---")
    st.subheader("💡 質問例")

    # CC Newsコレクションかどうかをチェック
    is_cc_news = collection.startswith("qa_cc_news_") or collection == "raw_cc_news"
    # Livedoorコレクションかどうかをチェック
    is_livedoor = collection.startswith("qa_livedoor_") or collection == "raw_livedoor"

    if is_cc_news:
        # CC Newsコレクション用のサンプル質問を表示
        st.write("**CC News サンプル検索:**")
        collection_label = ""
        if "raw_cc_news" in collection:
            collection_label = " (生データ)"
        elif "a02" in collection:
            collection_label = " (LLM生成)"
        elif "a03" in collection:
            collection_label = " (ルールベース)"
        elif "a10" in collection:
            collection_label = " (ハイブリッド)"
        st.caption(f"Collection: {collection}{collection_label}")

        for i, q in enumerate(SAMPLE_QUESTIONS.get("cc_news", []), 1):
            if st.button(f"{i}. {q}", key=f"sample_cc_news_{i}"):
                st.session_state['selected_query'] = q
    elif is_livedoor:
        # Livedoorコレクション用のサンプル質問を表示
        st.write("**Livedoor サンプル検索:**")
        collection_label = ""
        if "raw_livedoor" in collection:
            collection_label = " (生データ)"
        elif "a02" in collection:
            collection_label = " (LLM生成)"
        elif "a03" in collection:
            collection_label = " (ルールベース)"
        elif "a10" in collection:
            collection_label = " (ハイブリッド)"
        st.caption(f"Collection: {collection}{collection_label}")

        for i, q in enumerate(SAMPLE_QUESTIONS.get("livedoor", []), 1):
            if st.button(f"{i}. {q}", key=f"sample_livedoor_{i}"):
                st.session_state['selected_query'] = q
    elif domain != "ALL":
        st.write(f"**{domain.upper()}ドメインの質問例:**")
        for i, question in enumerate(SAMPLE_QUESTIONS.get(domain, []), 1):
            if st.button(f"{i}. {question[:30]}...", key=f"sample_{domain}_{i}"):
                st.session_state['selected_query'] = question
    else:
        # Show two examples for each domain when ALL is selected
        st.write("**ALLドメインの質問例（各ドメイン2件）**")
        for dom in ["customer", "medical", "legal", "sciq", "trivia"]:
            st.caption(f"{dom.upper()} ドメイン")
            examples = SAMPLE_QUESTIONS.get(dom, [])[:2]
            for i, q in enumerate(examples, 1):
                if st.button(f"{dom} {i}. {q[:30]}...", key=f"sample_all_{dom}_{i}"):
                    st.session_state['selected_query'] = q

        # Additionally show product_embeddings samples if that collection is selected
        if collection == "product_embeddings":
            st.markdown("---")
            st.write("**Product Embeddings サンプル検索:**")
            sample_queries = [
                "製品の特徴",
                "価格について",
                "使い方を教えて",
                "サポート情報"
            ]
            for i, q in enumerate(sample_queries, 1):
                if st.button(f"{i}. {q}", key=f"sample_product_{i}"):
                    st.session_state['selected_query'] = q

# Initialize session state for query
if 'selected_query' not in st.session_state:
    st.session_state['selected_query'] = "返金は可能ですか？"

st.code("""
機能・データセット説明:
  - collection「qa_corpus」: 5種類のドメイン別データセット（customer, medical, legal, sciq, trivia）
    - Domainセレクタで特定ドメインを選択可能
    - Domain=ALLは5つのデータセットの統合版

  - CC News collections: 3つの異なるQ&A生成手法で比較可能
    - raw_cc_news: CC Newsの生データ（7,376件）
    - qa_cc_news_a02_llm: LLM生成方式（1,344件）
      - AIによる自然な質問・回答ペアの生成
    - qa_cc_news_a03_rule: ルールベース生成方式（1,638件）
      - テンプレートベースの構造化されたQ&A生成
    - qa_cc_news_a10_hybrid: ハイブリッド生成方式（1,638件）
      - LLMとルールベースを組み合わせた生成

  - Livedoor News collections: 3つの異なるQ&A生成手法で比較可能
    - raw_livedoor: Livedoorの生データ（7,376件）
    - qa_livedoor_a02_20_llm: LLM生成方式（1,317件）
      - AIによる自然な質問・回答ペアの生成
    - qa_livedoor_a03_rule: ルールベース生成方式（1,638件）
      - テンプレートベースの構造化されたQ&A生成
    - qa_livedoor_a10_hybrid: ハイブリッド生成方式（1,638件）
      - LLMとルールベースを組み合わせた生成

  - 多言語対応埋め込み:
    - OpenAI embedding modelが多言語対応のため、日英間での意味的検索が可能
    - 例: 日本語「返金は可能ですか？」と英語「Can I get a refund?」の高い類似度（0.4957）
    - 翻訳なしで日英クロスリンガル検索が実現

  - 類似度スコア（Score）の目安:
    - 0.8以上: 非常に関連性が高い（ほぼ一致）
    - 0.6-0.8: 関連性がある（有用な結果）
    - 0.4-0.6: 部分的に関連（参考程度）
    - 0.4未満: 関連性が低い（フィルタリング推奨）
""")
query = st.text_input("Enter your query", value=st.session_state['selected_query'])
do_search = st.button("Search")

if do_search and query.strip():
    try:
        # Use the updated URL from input if provided
        current_qdrant_url = qdrant_url_input if 'qdrant_url_input' in locals() else qdrant_url
        client = QdrantClient(url=current_qdrant_url)
        # Test connection
        try:
            client.get_collections()
        except Exception as conn_err:
            st.error(f"❌ Qdrantサーバーに接続できません: {current_qdrant_url}")
            st.error("以下を確認してください:")
            st.error("1. Qdrantサーバーが起動しているか確認: `docker ps` または `qdrant` コマンド")
            st.error("2. URLが正しいか確認 (デフォルト: http://localhost:6333)")
            st.error(f"エラー詳細: {str(conn_err)}")
            st.stop()
        
        # コレクションに対応した埋め込み設定を取得（各コレクションは異なる次元数を持つ可能性がある）
        collection_config = COLLECTION_EMBEDDINGS.get(collection, {"model": model_for_using, "dims": None})
        embedding_model = collection_config["model"]
        embedding_dims = collection_config.get("dims")

        # デバッグモード：使用する埋め込み設定を表示
        if debug_mode:
            st.info(f"🔍 Using model: {embedding_model} with dims: {embedding_dims}")
        
        # クエリテキストを埋め込みベクトルに変換（コレクションごとの次元数に対応）
        try:
            qvec = embed_query(query, embedding_model, embedding_dims)
            if debug_mode:
                st.success(f"✅ Generated embedding with {len(qvec)} dimensions")
        except Exception as embed_err:
            st.error(f"❌ Embedding generation failed: {str(embed_err)}")
            st.error(f"Model: {embedding_model}, Requested dims: {embedding_dims}")
            st.stop()

        # ドメインフィルタの設定（qa_corpusコレクションのみ対応）
        qfilter = None
        if domain != "ALL":
            qfilter = models.Filter(must=[models.FieldCondition(key="domain", match=models.MatchValue(value=domain))])

        # Qdrantでベクトル類似検索を実行（deprecation警告を抑制）
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            hits = client.search(
                collection_name=collection,
                query_vector=qvec,
                limit=topk,
                query_filter=qfilter
            )
        rows = []
        for h in hits:
            # Debug: Show the actual payload structure
            if debug_mode:
                st.write(f"Debug - Payload keys: {h.payload.keys() if h.payload else 'No payload'}")
                st.write(f"Debug - Full payload: {h.payload}")
            
            # Try different field names that might be used
            row_data = {
                "score": h.score,
                "domain": h.payload.get("domain") if h.payload else None,
                "question": h.payload.get("question") or h.payload.get("text") or h.payload.get("content") if h.payload else None,
                "answer": h.payload.get("answer") or h.payload.get("response") or h.payload.get("metadata") if h.payload else None,
                "source": h.payload.get("source") or h.payload.get("file") if h.payload else None,
            }
            
            # If still no question/answer, try to extract from any text field
            if not row_data["question"] and h.payload:
                # Look for any text-like field
                for key in h.payload.keys():
                    if isinstance(h.payload[key], str) and len(h.payload[key]) > 10:
                        row_data["question"] = h.payload[key][:200]  # Limit to 200 chars
                        break
            
            rows.append(row_data)
        
        st.subheader("Results")
        st.dataframe(pd.DataFrame(rows))
        
        # Display the highest score result
        if rows:
            best_result = max(rows, key=lambda x: x["score"])
            st.subheader("🏆 Highest Score Result")
            st.write(f"**Score:** {best_result['score']:.4f}")
            st.write(f"**Question:** {best_result['question']}")
            st.write(f"**Answer:** {best_result['answer']}")

            # Ask Gemini using the result + original query (Japanese output)
            st.subheader("🧠 Gemini 応答（日本語）")
            try:
                br_q = best_result.get("question") or ""
                br_a = best_result.get("answer") or ""
                br_score = best_result.get("score") or 0.0

                qa_prompt_jp = (
                    "以下の検索結果（スコア・質問・回答）とユーザーの元の質問を踏まえて、" \
                    "日本語で簡潔かつ正確に回答してください。必要に応じて箇条書きを用いてください。\n\n"
                    f"ユーザーの元の質問（query）:\n{query}\n\n"
                    f"検索結果のスコア: {br_score:.4f}\n"
                    f"検索結果の質問: {br_q}\n"
                    f"検索結果の回答: {br_a}\n"
                )

                st.markdown("**質問（プロンプト）**")
                st.code(qa_prompt_jp)

                with st.spinner("Gemini AIに問い合わせ中..."):
                    llm_client = create_llm_client(provider="gemini")
                    generated_answer = llm_client.generate_content(
                        prompt=qa_prompt_jp,
                        model="gemini-2.0-flash"
                    )

                st.markdown("**回答（日本語）**")
                if generated_answer and generated_answer.strip():
                    st.write(generated_answer)
                else:
                    st.info("応答テキストを取得できませんでした。")
            except Exception as gen_err:
                st.error(f"Gemini応答生成に失敗しました: {str(gen_err)}")
    except ConnectionRefusedError:
        st.error(f"❌ Qdrantサーバーへの接続が拒否されました: {qdrant_url}")
        st.error("Qdrantサーバーが起動していることを確認してください:")
        st.code("cd docker-compose && docker-compose up -d", language="bash")
    except Exception as e:
        if "Connection refused" in str(e):
            st.error(f"❌ Qdrantサーバーに接続できません: {current_qdrant_url}")
            st.error("Qdrantサーバーが起動していることを確認してください:")
            st.code("cd docker-compose && docker-compose up -d", language="bash")
        elif "collection" in str(e).lower() and "not found" in str(e).lower():
            st.error(f"❌ コレクション '{collection}' が見つかりません")
            st.error("先に a42_qdrant_registration.py を実行してデータを登録してください:")
            st.code("python a42_qdrant_registration.py", language="bash")
        else:
            st.error(f"❌ エラーが発生しました: {str(e)}")
            st.error("エラーの詳細:")
            st.exception(e)
