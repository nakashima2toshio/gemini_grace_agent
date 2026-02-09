# ユーザークエリ入力〜Embedding処理: 改善点・問題点分析

**Version 1.0** | 分析日: 2026-02-09

---

## 分析対象ファイル

| ファイル | 役割 |
|---------|------|
| `grace_chat_page.py` | UI層（チャット入力・表示） |
| `agent_service.py` | ReActAgent（LLM制御・ツール呼び出し） |
| `agent_tools.py` | 検索エントリポイント（キャッシュ・並列・Rerank） |
| `qdrant_client_wrapper.py` | Embedding生成・Qdrant検索実行 |
| `agent_parallel_search.py` | 並列検索エンジン |
| `regex_mecab.py` | キーワード抽出 |
| `config.py` | 設定・定数 |

---

## 1. Embedding生成の重複呼び出し（🔴 重大）

### 問題

並列検索時に、**同一クエリに対してEmbedding APIが最大N回（コレクション数分）呼び出される**。

**根拠コード:**

```
agent_tools.py: search_rag_knowledge_base_cached()
  → parallel_search_engine.search_all_collections(query, all_collections, search_func_with_hybrid)
    → 各コレクションごとに search_rag_knowledge_base_structured(query, col) を並列実行
      → embed_query(query)            ← コレクションごとに毎回呼ばれる
      → embed_sparse_query_unified(query) ← コレクションごとに毎回呼ばれる
```

`search_rag_knowledge_base_structured()` (agent_tools.py L398) で `embed_query(query)` を呼んでおり、`ParallelSearchEngine._search_single_collection()` はコレクション単位で `search_func` を呼ぶため、4コレクションあれば Dense Embedding が4回、Sparse Embedding も4回生成される。

### 改善案

Embeddingを**事前に1回だけ生成**し、各コレクション検索にはベクトルのみを渡す。

```python
# 改善案: search_rag_knowledge_base_cached() 内
query_vector = embed_query(query)                          # 1回だけ
sparse_vector = embed_sparse_query_unified(query) if use_hybrid_search else None  # 1回だけ

# 各コレクションにはベクトルを渡す
def search_func(q, col):
    return search_collection_with_vectors(client, col, query_vector, sparse_vector, limit=20)
```

### 影響

- **API コスト削減**: Gemini Embedding API の呼び出し回数が N → 1 に
- **レイテンシ改善**: Embedding生成（約100-300ms）× N回 が1回に
- **レート制限リスク低減**: API呼び出し集中による429エラーの回避

---

## 2. Qdrant ヘルスチェック・コレクション存在確認の過剰呼び出し（🔴 重大）

### 問題

`search_rag_knowledge_base_structured()` が呼ばれるたびに、以下の処理が**毎回**実行される:

```python
# agent_tools.py L389-396
if not check_qdrant_health():          # ← client.get_collections() を実行
    raise QdrantConnectionError(...)

existing_collections = [c.name for c in client.get_collections().collections]  # ← 2回目
if collection_name not in existing_collections:
    raise CollectionNotFoundError(...)
```

加えて `search_collection()` (qdrant_client_wrapper.py L1016) 内でも:

```python
collection_info = client.get_collection(collection_name)  # ← 3回目（ベクトル設定確認用）
```

並列4コレクション検索時、Qdrant への `get_collections()` / `get_collection()` が **少なくとも12回**発生する。

### 改善案

- ヘルスチェックは**起動時またはセッション開始時に1回**実行し、結果をキャッシュ
- コレクション存在確認は `search_rag_knowledge_base_cached()` レベルで1回行い、子関数には verified フラグを渡す
- `search_collection()` 内のベクトル設定は**初回のみ取得してキャッシュ**する（コレクションのスキーマは起動中に変わらない）

```python
# 例: コレクション設定キャッシュ
_collection_config_cache: Dict[str, dict] = {}

def get_collection_vector_config(client, collection_name):
    if collection_name not in _collection_config_cache:
        info = client.get_collection(collection_name)
        _collection_config_cache[collection_name] = {
            "is_named_vector": isinstance(info.config.params.vectors, dict),
            "dense_vector_name": "default" if isinstance(info.config.params.vectors, dict) else None
        }
    return _collection_config_cache[collection_name]
```

---

## 3. EmbeddingClient の毎回インスタンス化（🟡 中程度）

### 問題

`embed_query_unified()` (qdrant_client_wrapper.py L596-599) が呼ばれるたびに:

```python
def embed_query_unified(text, provider=None):
    provider = provider or DEFAULT_EMBEDDING_PROVIDER
    embedding_client = create_embedding_client(provider=provider)  # ← 毎回新規作成
    return embedding_client.embed_text(text, task_type="retrieval_query")
```

`create_embedding_client()` の実装は `helper_embedding.py`（未提供）にあるが、毎回クライアントを生成するのはオーバーヘッド。特にモデルのロードや接続初期化が含まれる場合は大きい。

### 改善案

モジュールレベルでシングルトンとして保持する:

```python
_embedding_clients: Dict[str, EmbeddingClient] = {}

def get_embedding_client(provider: str = None) -> EmbeddingClient:
    provider = provider or DEFAULT_EMBEDDING_PROVIDER
    if provider not in _embedding_clients:
        _embedding_clients[provider] = create_embedding_client(provider=provider)
    return _embedding_clients[provider]
```

※ `embed_sparse_query_unified()` の `get_sparse_embedding_client()` も同様。

---

## 4. QdrantClient のグローバルインスタンス管理（🟡 中程度）

### 問題

`QdrantClient` が複数箇所で**個別に生成**されている:

| 場所 | コード |
|------|--------|
| `agent_tools.py` L35 | `client = QdrantClient(url=qdrant_url)` （モジュールレベル） |
| `grace_chat_page.py` L55 | `client = QdrantClient(url=os.getenv("QDRANT_URL", ...))` |
| `agent_service.py` L442 | `client = QdrantClient(url=qdrant_url)` (`get_available_collections_from_qdrant_helper`) |

各インスタンスは別の接続プールを持ち、リソースが無駄になる。また設定の不一致リスクもある（`os.getenv` vs `QdrantConfig.URL`）。

### 改善案

`qdrant_client_wrapper.py` にファクトリ関数を用意し、シングルトンで管理:

```python
_qdrant_client: Optional[QdrantClient] = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QdrantConfig.URL, timeout=QdrantConfig.DEFAULT_TIMEOUT)
    return _qdrant_client
```

---

## 5. キーワード抽出とLLMクエリ生成の二重処理（🟡 中程度）

### 問題

`execute_turn()` で以下の流れになっている:

1. `KeywordExtractor.extract(user_input)` → キーワード抽出
2. 抽出キーワードをプロンプトに追加: `「重要キーワード: X, Y, Z」`
3. LLM(Gemini)が `Thought:` で思考し、`Action: search_rag_knowledge_base(query="...")` を生成
4. **LLMが生成した `query` で検索実行**（キーワードが含まれる保証なし）

つまり、KeywordExtractor で抽出したキーワードは**LLMへのヒント**に過ぎず、LLMが生成する検索クエリに反映される保証がない。キーワードが検索クエリに含まれなかった場合、抽出処理が無駄になる。

### 改善案

2つのアプローチを検討:

**案A: LLM生成クエリにキーワードを強制付加**
```python
# LLMが生成したクエリに抽出キーワードを追加
final_query = f"{llm_generated_query} {' '.join(keywords)}"
```

**案B: Embedding検索とは別にキーワードフィルタリングに活用**（現在の `filter_results_by_keywords` を強化）
```python
# MeCab抽出キーワードで検索結果をフィルタ
filtered = filter_results_by_extracted_keywords(results, mecab_keywords)
```

---

## 6. Sparse Embedding のフォールバック処理が多重化（🟡 中程度）

### 問題

Sparse Vector がサポートされないコレクションへのフォールバックが **3段階** で重複実装されている:

1. **agent_tools.py L403-412**: `embed_sparse_query_unified()` の例外キャッチ → `sparse_vector = None`
2. **agent_tools.py L426-443**: `search_collection()` のスパースエラーキャッチ → `sparse_vector=None` で再試行
3. **qdrant_client_wrapper.py L1052-1060**: `search_collection()` 内の `UnexpectedResponse` キャッチ → `sparse_vector = None` で再試行

同一エラーに対して3段階のtry-exceptが走り、コードの可読性が低く、ログも混乱しやすい。

### 改善案

フォールバックの責務を `search_collection()` に一元化する:

```python
def search_collection(client, collection_name, query_vector, sparse_vector=None, limit=5):
    # sparse_vector があれば Hybrid Search を試行、失敗なら Dense のみ
    # フォールバックロジックはここだけ
    ...
```

呼び出し側 (`search_rag_knowledge_base_structured`) では try-except 不要にする。

---

## 7. `search_rag_knowledge_base` と `search_rag_knowledge_base_structured` の責務重複（🟡 中程度）

### 問題

`search_rag_knowledge_base()` (Legacy版) は内部で `search_rag_knowledge_base_structured()` を呼び、さらに独自のフォールバック検索ロジックを持っている。一方 `search_rag_knowledge_base_cached()` も `search_rag_knowledge_base_structured()` を呼ぶ。

```
search_rag_knowledge_base_cached → search_rag_knowledge_base → search_rag_knowledge_base_structured
search_rag_knowledge_base_cached → parallel_search_engine → search_rag_knowledge_base_structured
search_rag_knowledge_base_cached → (直接) → search_rag_knowledge_base_structured
```

3つのエントリポイントが存在し、フォーマット処理も `search_rag_knowledge_base()` と `_format_results()` で重複。

### 改善案

- `search_rag_knowledge_base()` の独自フォールバックロジックを削除（`search_rag_knowledge_base_cached` が並列検索で全コレクションをカバーしているため不要）
- エントリポイントを `search_rag_knowledge_base_cached()` に統一し、Legacy版は薄いラッパーにする

---

## 8. `filter_results_by_keywords` が未使用（⚪ 軽微）

### 問題

`agent_tools.py` L144 に `filter_results_by_keywords()` が定義されているが、現在の検索パイプラインのどこからも呼ばれていない。デッドコードになっている。

### 改善案

活用するか削除する。活用する場合は `rerank_results()` の後に配置するのが効果的。

---

## 9. Embedding次元数の不整合リスク（⚪ 軽微だが潜在的）

### 問題

`config.py` で `QdrantConfig.DEFAULT_VECTOR_SIZE = 3072` (gemini-embedding-001) だが、設計ドキュメント (`grace_chat_input_query.md`) では「768次元」と記載されている箇所がある。

また、`COLLECTION_EMBEDDINGS` にはOpenAI用 (1536次元) のコレクションも混在しており、検索時に次元数不整合でエラーになるリスクがある。

現在 `embed_query()` は常に `provider="gemini"` (3072次元) を使うため、OpenAI用コレクション (1536次元) に対して検索すると次元不一致エラーが発生する。

### 改善案

- コレクションごとにどのプロバイダー/次元数で作成されたかをメタデータとして保持
- 検索時にコレクションの次元数に合致するプロバイダーで Embedding を生成する

```python
def embed_query_for_collection(query: str, collection_name: str) -> List[float]:
    config = get_collection_embedding_config(collection_name)
    return embed_query_unified(query, provider=config["provider"])
```

---

## 10. print() がプロダクションコードに残存（⚪ 軽微）

### 問題

`regex_mecab.py` L102-103:
```python
print("✅ MeCabが利用可能です（複合名詞抽出モード）")
print("⚠️ MeCabが利用できません（正規表現モード）")
```

Streamlit環境ではコンソールに出力されるだけで、ユーザーには見えない。`logger` を使うべき。

---

## 改善優先度まとめ

| 優先度 | # | 項目 | 効果 |
|--------|---|------|------|
| 🔴 高 | 1 | Embedding生成の重複呼び出し | API コスト・レイテンシ大幅削減 |
| 🔴 高 | 2 | ヘルスチェック・存在確認の過剰呼び出し | Qdrant負荷・レイテンシ削減 |
| 🟡 中 | 3 | EmbeddingClient の毎回インスタンス化 | オーバーヘッド削減 |
| 🟡 中 | 4 | QdrantClient の分散管理 | リソース統一・設定不一致防止 |
| 🟡 中 | 5 | キーワード抽出とLLMクエリの二重処理 | 検索精度向上 |
| 🟡 中 | 6 | Sparse フォールバックの多重化 | コード可読性・保守性向上 |
| 🟡 中 | 7 | 検索関数の責務重複 | アーキテクチャ整理 |
| ⚪ 低 | 8 | デッドコード | コード整理 |
| ⚪ 低 | 9 | Embedding次元数の不整合リスク | 将来的なバグ防止 |
| ⚪ 低 | 10 | print()残存 | ログ統一 |
