#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
a42_qdrant_registration.py — 3つの異なるコレクションへの分離登録版 (Gemini対応)
--------------------------------------------------------------------------------
cc_newsドメインのQ&Aデータを、生成方法ごとに別々のコレクションに登録します。

【コレクション構成】
- qa_cc_news_a02_llm    : a02_qa_pairs_cc_news.csv (LLM生成方式)
- qa_cc_news_a03_rule   : a03_qa_pairs_cc_news.csv (ルールベース生成方式)
- qa_cc_news_a10_hybrid : a10_qa_pairs_cc_news.csv (ハイブリッド生成方式)

【主な変更点】
- OpenAI APIの使用を廃止し、Gemini API (helper_embedding) に移行
- 単一コレクション → 3つの独立したコレクション
- 各CSVファイルが専用のコレクションを持つ

使い方：
  # 1. CSVファイルを生成（a20_output_qa_csv.pyを実行）
  python a20_output_qa_csv.py

  # 2. Qdrantサーバーを起動
  export GOOGLE_API_KEY=...
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

  # 3. 3つのコレクションにデータを登録
  python a42_qdrant_registration.py --recreate --include-answer

主要引数：
  --recreate          : コレクション削除→新規作成
  --collection        : 特定コレクションのみ処理（指定なしで全コレクション）
  --input-file        : ローカルファイルを登録（CSV/TXT/JSON/JSONL）
  --qdrant-url        : 既定は http://localhost:6333
  --batch-size        : Embeddings/Upsert バッチサイズ（既定 32）
  --limit             : データ件数上限（開発用、0=無制限）
  --include-answer    : 埋め込み入力に answer も結合（question + "\n" + answer）
  --search            : クエリ指定で検索のみ実行
  --topk              : 上位件数（既定5）
"""
import argparse
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Any
from pathlib import Path

import pandas as pd

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

try:
    import helper_rag as hrag
except Exception:
    hrag = None

# Gemini対応のためのインポート
from helper_embedding import create_embedding_client

from qdrant_client import QdrantClient
from qdrant_client.http import models

# ------------------ デフォルト設定 ------------------
DEFAULTS = {
    "rag": {
        "include_answer_in_embedding": False,
    },
    "embeddings": {
        "primary": {"provider": "gemini", "model": "gemini-embedding-001", "dims": 3072},
    },
    "qdrant": {"url": "http://localhost:6333"},
}

# ------------------ コレクション定義 ------------------
# CSVファイル/TXTファイル → コレクション名 → 生成方法のマッピング
COLLECTION_MAPPINGS = [
    {
        "csv_file": "qa_output/a02_qa_pairs_cc_news.csv",
        "collection": "qa_cc_news_a02_llm",
        "generation_method": "a02_make_qa",
        "domain": "cc_news",
        "description": "LLM生成方式（a02_make_qa_para.py）",
        "type": "qa"
    },
    {
        "csv_file": "qa_output/a03_qa_pairs_cc_news.csv",
        "collection": "qa_cc_news_a03_rule",
        "generation_method": "a03_coverage",
        "domain": "cc_news",
        "description": "ルールベース生成方式（a03_rag_qa_coverage_improved.py）",
        "type": "qa"
    },
    {
        "csv_file": "qa_output/a10_qa_pairs_cc_news.csv",
        "collection": "qa_cc_news_a10_hybrid",
        "generation_method": "a10_hybrid",
        "domain": "cc_news",
        "description": "ハイブリッド生成方式（a10_qa_optimized_hybrid_batch.py）",
        "type": "qa"
    },
    {
        "csv_file": "qa_output/a02_qa_pairs_livedoor.csv",
        "collection": "qa_livedoor_a02_20_llm",
        "generation_method": "a02_make_qa",
        "domain": "livedoor",
        "description": "LLM生成方式（a02_make_qa_para.py）- livedoorデータ",
        "type": "qa"
    },
    {
        "csv_file": "qa_output/a03_qa_pairs_livedoor.csv",
        "collection": "qa_livedoor_a03_rule",
        "generation_method": "a03_coverage",
        "domain": "livedoor",
        "description": "ルールベース生成方式（a03_rag_qa_coverage_improved.py）- livedoorデータ",
        "type": "qa"
    },
    {
        "csv_file": "qa_output/a10_qa_pairs_livedoor.csv",
        "collection": "qa_livedoor_a10_hybrid",
        "generation_method": "a10_hybrid",
        "domain": "livedoor",
        "description": "ハイブリッド生成方式（a10_qa_optimized_hybrid_batch.py）- livedoorデータ",
        "type": "qa"
    },
    {
        "csv_file": "OUTPUT/cc_news.txt",
        "collection": "raw_cc_news",
        "generation_method": "raw_text",
        "domain": "cc_news",
        "description": "生テキストデータ（cc_news）",
        "type": "raw"
    },
    {
        "csv_file": "OUTPUT/livedoor.txt",
        "collection": "raw_livedoor",
        "generation_method": "raw_text",
        "domain": "livedoor",
        "description": "生テキストデータ（livedoor）",
        "type": "raw"
    }
]


# ------------------ 設定ロード ------------------
def load_config(path: str = "config.yml") -> Dict[str, Any]:
    cfg = {}
    if yaml and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    # マージ（浅いマージで十分。必要なら深いマージに差し替え）
    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst.setdefault(k, v)
    full = {}
    merge(full, DEFAULTS)
    merge(full, cfg)
    return full

# ------------------ 小道具 ------------------
def batched(seq: Iterable, size: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

# ------------------ 埋め込み実装（helper優先） ------------------
def embed_texts(texts: List[str], model: str, batch_size: int = 128, provider: str = "gemini") -> List[List[float]]:
    """
    テキストをバッチ処理でEmbeddingに変換
    helper_embeddingを使用して、Gemini/OpenAI/FastEmbedに対応
    """
    # 空文字列の除去とインデックス保持
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text)
            valid_indices.append(i)

    if not valid_texts:
        print("\r   [WARN] 全てのテキストが空文字列です。ダミーベクトルを返します。", flush=True)
        # 次元数は仮に3072とする（Gemini）
        return [[0.0] * 3072] * len(texts)

    # クライアント生成
    client = create_embedding_client(provider=provider, model=model)
    
    # 埋め込み生成
    print(f"\r   埋め込み生成中: {len(valid_texts)}件 (Provider: {provider})... ", end="", flush=True)
    valid_vecs = client.embed_texts(valid_texts, batch_size=batch_size)
    
    # ベクトルの再構築
    vecs = []
    dims = client.dimensions
    valid_vec_idx = 0
    
    for i in range(len(texts)):
        if i in valid_indices:
            if valid_vec_idx < len(valid_vecs):
                vecs.append(valid_vecs[valid_vec_idx])
                valid_vec_idx += 1
            else:
                # エラーなどで数が合わない場合のフォールバック
                vecs.append([0.0] * dims)
        else:
            vecs.append([0.0] * dims)
            
    return vecs

# ------------------ 入力テキスト構築 ------------------
def build_inputs(df: pd.DataFrame, include_answer: bool) -> List[str]:
    if include_answer:
        return (df["question"].astype(str) + "\n" + df["answer"].astype(str)).tolist()
    return df["question"].astype(str).tolist()

# ------------------ CSVロード ------------------
def load_csv(path: str, required=("question", "answer"), limit: int = 0) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    # 列名マッピングが必要ならここで調整（例: 'Question'->'question'）
    column_mappings = {
        'Question': 'question',
        'Response': 'answer',
        'Answer': 'answer',
        'correct_answer': 'answer'
    }
    df = df.rename(columns=column_mappings)
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path} には '{col}' 列が必要です（列: {list(df.columns)}）")
    df = df.fillna("").drop_duplicates(subset=list(required)).reset_index(drop=True)
    if limit and limit > 0:
        df = df.head(limit).copy()
    return df

# ------------------ TXTロード（生テキスト） ------------------
# (detect_language, chunk_japanese_text, chunk_english_text は helper_rag / helper_text にあるものを使用すべきだが、
# ここでは独立性を保つため、または helper_rag を使うように変更するのが望ましい)
# 簡易的に helper_rag があればそちらを使い、なければ簡易実装を使う

def load_txt(path: str, limit: int = 0, chunk_size: int = 500) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"TXT not found: {path}")

    # 簡易チャンク分割（固定長）
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    with open(path, "r", encoding="utf-8") as f:
        full_text = f.read()
    tokens = enc.encode(full_text)
    
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i+chunk_size]
        chunks.append(enc.decode(chunk_tokens))

    if limit and limit > 0:
        chunks = chunks[:limit]

    return pd.DataFrame({"text": chunks})

# ------------------ Qdrant: コレクション作成（Named Vectors対応） ------------------
def create_or_recreate_collection(client: QdrantClient, name: str, recreate: bool,
                                  embeddings_cfg: Dict[str, Dict[str, Any]]):
    # embeddings_cfg: dict[name] = {"model": "...", "dims": int}
    if len(embeddings_cfg) == 1:
        dims = list(embeddings_cfg.values())[0]["dims"]
        vectors_config = models.VectorParams(size=dims, distance=models.Distance.COSINE)
    else:
        # Named vectors
        vectors_config = {
            k: models.VectorParams(size=v["dims"], distance=models.Distance.COSINE)
            for k, v in embeddings_cfg.items()
        }
    if recreate:
        client.recreate_collection(collection_name=name, vectors_config=vectors_config)
    else:
        # 無ければ作成
        try:
            client.get_collection(name)
        except Exception:
            client.create_collection(collection_name=name, vectors_config=vectors_config)
    # よく使うpayloadの索引（任意）
    try:
        client.create_payload_index(name, field_name="domain", field_schema=models.PayloadSchemaType.KEYWORD)
    except Exception:
        pass
    try:
        client.create_payload_index(name, field_name="generation_method", field_schema=models.PayloadSchemaType.KEYWORD)
    except Exception:
        pass

# ------------------ ポイント構築（Named Vectors対応） ------------------
def build_points(df: pd.DataFrame, vectors_by_name: Dict[str, List[List[float]]], domain: str, source_file: str,
                 generation_method: str = None, data_type: str = "qa") -> List[models.PointStruct]:
    n = len(df)
    for name, vecs in vectors_by_name.items():
        if len(vecs) != n:
            raise ValueError(f"vectors length mismatch for '{name}': df={n}, vecs={len(vecs)}")
    now_iso = datetime.now(timezone.utc).isoformat()
    points: List[models.PointStruct] = []

    for i, row in enumerate(df.itertuples(index=False)):
        if data_type == "qa":
            payload = {
                "domain": domain,
                "generation_method": generation_method or "unknown",
                "question": getattr(row, "question"),
                "answer": getattr(row, "answer"),
                "source": os.path.basename(source_file),
                "created_at": now_iso,
                "schema": "qa:v1",
            }
        else:  # raw text
            payload = {
                "domain": domain,
                "generation_method": generation_method or "unknown",
                "text": getattr(row, "text"),
                "source": os.path.basename(source_file),
                "created_at": now_iso,
                "schema": "raw:v1",
            }

        pid = abs(hash(f"{domain}-{generation_method}-{source_file}-{i}")) & 0x7FFFFFFFFFFFFFFF
        if len(vectors_by_name) == 1:
            vec = list(vectors_by_name.values())[0][i]
            points.append(models.PointStruct(id=pid, vector=vec, payload=payload))
        else:
            vecs_dict = {name: vecs[i] for name, vecs in vectors_by_name.items()}
            points.append(models.PointStruct(id=pid, vector=vecs_dict, payload=payload))
    return points

def upsert_points(client: QdrantClient, collection: str, points: List[models.PointStruct], batch_size: int = 128) -> int:
    count = 0
    for chunk in batched(points, batch_size):
        client.upsert(collection_name=collection, points=chunk)
        count += len(chunk)
    return count

# ------------------ 検索（Named Vectors対応） ------------------
def embed_one(text: str, model: str, provider: str = "gemini") -> List[float]:
    return embed_texts([text], model=model, batch_size=1, provider=provider)[0]

def search(client: QdrantClient, collection: str, query: str, using_vec: str, model_for_using: str,
           topk: int = 5, domain: Optional[str] = None, generation_method: Optional[str] = None, provider: str = "gemini"):
    qvec = embed_one(query, model=model_for_using, provider=provider)
    qfilter = None
    filter_conditions = []
    if domain:
        filter_conditions.append(models.FieldCondition(key="domain", match=models.MatchValue(value=domain)))
    if generation_method:
        filter_conditions.append(models.FieldCondition(key="generation_method", match=models.MatchValue(value=generation_method)))
    if filter_conditions:
        qfilter = models.Filter(must=filter_conditions)
    
    try:
        hits = client.search(collection_name=collection, query_vector=qvec, limit=topk,
                           query_filter=qfilter)
    except Exception as e:
        print(f"検索エラー: {e}")
        hits = []
    
    return hits

# ------------------ メイン ------------------
def main():
    cfg = load_config("config.yml")
    rag_cfg = cfg.get("rag", {})
    
    # 埋め込み設定をGemini向けに上書きまたは取得
    embeddings_cfg: Dict[str, Dict[str, Any]] = cfg.get("embeddings", {})
    if not embeddings_cfg or "primary" not in embeddings_cfg:
        embeddings_cfg = DEFAULTS["embeddings"]
    
    # primaryプロバイダーがopenaiならgeminiに書き換える（安全策）
    if embeddings_cfg["primary"].get("provider") == "openai":
        print("[INFO] 設定ファイルのOpenAIプロバイダーをGeminiにオーバーライドします。")
        embeddings_cfg["primary"] = DEFAULTS["embeddings"]["primary"]

    qdrant_url = (cfg.get("qdrant", {}) or {}).get("url", "http://localhost:6333")

    ap = argparse.ArgumentParser(
        description="3つのQ&Aデータセットをそれぞれ独立したQdrantコレクションに登録"
    )
    ap.add_argument("--recreate", action="store_true",
                    help="Drop & create collection before upsert.")
    ap.add_argument("--collection", default=None,
                    help="特定コレクションのみ処理（指定なしで全コレクション）")
    ap.add_argument("--input-file", default=None,
                    help="ローカルファイルを登録（CSV/TXT/JSON/JSONL）")
    ap.add_argument("--qdrant-url", default=qdrant_url)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                    help="Row limit per CSV for development (0=all)")
    ap.add_argument("--include-answer", action="store_true",
                    default=rag_cfg.get("include_answer_in_embedding", False),
                    help="Use 'question\nanswer' as embedding input.")
    ap.add_argument("--search", default=None, help="Run search only.")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    # Qdrant client
    client = QdrantClient(url=args.qdrant_url, timeout=300)

    # プロバイダーとモデルの取得
    provider = embeddings_cfg["primary"].get("provider", "gemini")
    model = embeddings_cfg["primary"].get("model", "gemini-embedding-001")

    # 検索のみ
    if args.search:
        if not args.collection:
            print("[ERROR] 検索には --collection の指定が必要です")
            return

        hits = search(client, args.collection, args.search, "primary", model, topk=args.topk, provider=provider)

        print(f"\n[Search] collection={args.collection} query={args.search!r}")
        for h in hits:
            method = h.payload.get('generation_method', 'unknown')
            question = h.payload.get('question', '')[:80]
            answer = h.payload.get('answer', '')[:80]
            print(f"score={h.score:.4f}  method={method}  Q: {question}  A: {answer}...")
        return

    # 処理対象のコレクションを決定
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"[ERROR] ファイルが見つかりません: {args.input_file}")
            return

        file_path = Path(args.input_file)
        file_basename = file_path.stem
        file_ext = file_path.suffix.lower().lstrip('.')

        clean_name = file_basename.replace('qa_pairs_', '').replace('a02_qa_pairs_', '').replace('a03_qa_pairs_', '').replace('a10_qa_pairs_', '')
        collection_name = f"qa_{clean_name}"

        if 'cc_news' in file_basename.lower():
            domain = 'cc_news'
        elif 'livedoor' in file_basename.lower():
            domain = 'livedoor'
        elif 'upload' in file_basename.lower():
            domain = 'custom_upload'
        else:
            domain = 'custom'

        if file_ext in ['txt', 'text']:
            data_type = 'raw'
            generation_method = 'raw_text'
        else:
            data_type = 'qa'
            generation_method = 'custom_upload'

        target_mappings = [{
            "csv_file": args.input_file,
            "collection": collection_name,
            "generation_method": generation_method,
            "domain": domain,
            "description": f"カスタムアップロード ({file_basename})",
            "type": data_type
        }]

        print("\n[INFO] ローカルファイル登録モード")
        print(f"  ファイル: {args.input_file}")
        print(f"  コレクション: {collection_name}")
        print(f"  ドメイン: {domain}")
        print(f"  タイプ: {data_type}")

    elif args.collection:
        target_mappings = [m for m in COLLECTION_MAPPINGS if m["collection"] == args.collection]
        if not target_mappings:
            print(f"[ERROR] コレクション '{args.collection}' は定義されていません")
            print(f"利用可能なコレクション: {[m['collection'] for m in COLLECTION_MAPPINGS]}")
            return
    else:
        target_mappings = COLLECTION_MAPPINGS

    # インジェスト処理
    print(f"\n[INFO] 処理対象: {len(target_mappings)} コレクション")
    print("=" * 80)

    total = 0
    for mapping in target_mappings:
        csv_file = mapping["csv_file"]
        collection_name = mapping["collection"]
        generation_method = mapping["generation_method"]
        domain = mapping["domain"]
        description = mapping["description"]

        print(f"\n📦 コレクション: {collection_name}")
        print(f"   説明: {description}")
        print(f"   ソース: {csv_file}")
        print("-" * 80)

        if not os.path.exists(csv_file):
            print(f"[WARN] ファイルが見つかりません: {csv_file} (スキップ)")
            continue

        # コレクション作成
        create_or_recreate_collection(client, collection_name, args.recreate, embeddings_cfg)

        # データタイプに応じてロード
        data_type = mapping.get("type", "qa")
        if data_type == "raw":
            df = load_txt(csv_file, limit=args.limit)
            print(f"   データ件数: {len(df):,}件")
            texts = df["text"].tolist()
        else:
            df = load_csv(csv_file, limit=args.limit)
            print(f"   データ件数: {len(df):,}件")
            texts = build_inputs(df, include_answer=args.include_answer)

        # 埋め込み生成
        vectors_by_name: Dict[str, List[List[float]]] = {}
        for name, vcfg in embeddings_cfg.items():
            current_provider = vcfg.get("provider", "gemini")
            current_model = vcfg.get("model", "gemini-embedding-001")
            print(f"   埋め込み生成中: {name} (provider={current_provider}, model={current_model})... ", end="", flush=True)
            
            vectors_by_name[name] = embed_texts(
                texts, 
                model=current_model, 
                batch_size=args.batch_size,
                provider=current_provider
            )
            print("✓")

        # ポイント構築とアップサート
        points = build_points(df, vectors_by_name, domain, csv_file, generation_method, data_type=data_type)
        print("   アップサート中... ", end="", flush=True)
        n = upsert_points(client, collection_name, points, batch_size=args.batch_size)
        print(f"✓ {n:,}件")

        total += n

    print("\n" + "=" * 80)
    print(f"✅ 完了: 総登録件数 {total:,}件")

    # 登録されたコレクション一覧を表示
    print("\n[INFO] 登録されたコレクション一覧:")
    print("-" * 80)
    all_collections = client.get_collections()
    for col in all_collections.collections:
        print(f"  • {col.name}")
    print("-" * 80)

    # 検証検索
    print("\n[INFO] 検証検索を実行中...")
    
    for mapping in target_mappings:
        collection_name = mapping["collection"]
        try:
            info = client.get_collection(collection_name)
            print(f"\n  {collection_name}: {info.points_count:,}件登録済み")

            hits = search(client, collection_name, "気候変動", "primary", model, topk=2, provider=provider)
            if hits:
                for h in hits[:1]:
                    q = h.payload.get('question', '')[:50]
                    print(f"    サンプル検索結果: score={h.score:.4f}  Q: {q}...")
        except Exception as e:
            print(f"    検証エラー: {e}")

if __name__ == "__main__":
    main()