# agent_tools.py

import os
import time
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from qdrant_client_wrapper import search_collection, embed_query, embed_sparse_query_unified, QDRANT_CONFIG
from config import AgentConfig, CohereConfig

try:
    import cohere
except ImportError:
    cohere = None

logger = logging.getLogger(__name__) # Configure logger for this module

# Initialize Client
qdrant_url: str = QDRANT_CONFIG.get("url", "http://localhost:6333")
client: QdrantClient = QdrantClient(url=qdrant_url)


# ============ カスタム例外 ============ 
class RAGToolError(Exception):
    """RAGツール固有のエラー基底クラス"""
    pass

class QdrantConnectionError(RAGToolError):
    """Qdrant接続エラー"""
    pass

class CollectionNotFoundError(RAGToolError):
    """コレクション未存在エラー"""
    pass

class EmbeddingError(RAGToolError):
    """埋め込み生成エラー"""
    pass


# ============ 評価用メトリクス ============ 
@dataclass
class SearchMetrics:
    """検索結果のメトリクス（評価用）"""
    query: str
    collection_name: str
    latency_ms: float
    total_results: int
    filtered_results: int
    top_score: float
    scores: List[float] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

# Global metrics log (in-memory for evaluation session)
_search_metrics_log: List[SearchMetrics] = []

def get_search_metrics() -> List[SearchMetrics]:
    """評価用: 収集したメトリクスを取得"""
    return _search_metrics_log.copy()

def clear_search_metrics() -> None:
    """評価用: メトリクスをクリア"""
    _search_metrics_log.clear()

def export_metrics_to_dict() -> List[Dict[str, Any]]:
    """メトリクスを辞書形式でエクスポート"""
    from dataclasses import asdict
    return [asdict(m) for m in _search_metrics_log]


# ============ ヘルスチェック ============ 
def check_qdrant_health() -> bool:
    """Qdrantサーバーの接続確認"""
    try:
        client.get_collections()
        logger.info("Qdrant health check: OK")
        return True
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        return False


# ============ ツール関数 ============ 
def list_rag_collections() -> str:
    """
    利用可能なRAGのコレクション一覧（ナレッジベースの種類）を取得します。
    ユーザーが「どのような知識があるか」「コレクション一覧を教えて」と質問した場合に使用してください。
    Returns:
        str: 利用可能なコレクション名のリスト。
    """
    logger.info("ツールアクション: コレクション一覧を取得中...")
    try:
        collections_response = client.get_collections()
        collections: List[str] = [c.name for c in collections_response.collections]

        if not collections:
            logger.info("Qdrantに利用可能なコレクションがありません。")
            return "現在、利用可能なコレクションはありません。"

        result_lines: List[str] = ["利用可能なコレクション一覧:"]
        for c in collections:
            try:
                info = client.get_collection(c)
                count: int = info.points_count
                result_lines.append(f"- {c} ({count} documents)")
            except (UnexpectedResponse, ResponseHandlingException) as e:
                logger.warning(f"コレクション '{c}' の情報取得エラー: {e}")
                result_lines.append(f"- {c} (情報取得エラー)")
            except Exception as e:
                logger.error(f"不明なエラー: コレクション '{c}' の情報取得中にエラーが発生しました: {e}", exc_info=True)
                result_lines.append(f"- {c} (不明なエラー)")

        logger.info(f"コレクション一覧取得完了: {len(collections)}件")
        return "\n".join(result_lines)

    except Exception as e:
        logger.error(f"コレクション一覧取得エラー: {e}", exc_info=True)
        raise QdrantConnectionError(f"Qdrant接続エラー、またはコレクション一覧の取得に失敗しました: {str(e)}")


def filter_results_by_keywords(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    検索結果をクエリのキーワードでフィルタリングする（共通ロジック）
    Legacy Agentと同じく、スペース区切りのトークンを必須キーワードとして扱う。
    """
    import re
    
    # 必須キーワードの抽出（Legacyと同一ロジック: スペース区切り）
    tokens = query.split()
    required_keywords = []
    
    for t in tokens:
        # 2文字以上で、かつ記号のみでないものを採用
        if len(t) >= 2:
             required_keywords.append(t)

    required_keywords = list(set(required_keywords))
    logger.info(f"Filtering Logic - Required keywords: {required_keywords}")

    filtered_results = []
    for res in results:
        payload = res.get("payload", {})
        content = (str(payload.get("question", "")) + " " + 
                   str(payload.get("answer", "")) + " " + 
                   str(payload.get("content", "")))

        is_relevant = True
        if required_keywords:
            # キーワードが1つでも含まれていればOKとする（緩やかなAND条件）
            # Legacy Agentでは「キーワードを含めてください」と指示しているため、
            # 検索結果にそれらが含まれることを期待するが、
            # 全てが含まれるとは限らないため、ヒット数で判定。
            hit_count = sum(1 for k in required_keywords if k in content)
            
            # 1つもヒットしない場合は除外
            if hit_count == 0:
                is_relevant = False
                logger.debug(f"Keyword miss (score={res.get('score', 0):.3f}): Filtering out.")

        if is_relevant:
            filtered_results.append(res)
            
    return filtered_results


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 3,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    検索結果をCohere Rerank APIで再評価し、スコアを更新してソートする。
    
    Args:
        query: ユーザーの検索クエリ
        results: Qdrantからの検索結果リスト
        top_k: 最終的に残す件数
        threshold: スコアの足切りライン
        
    Returns:
        再ランク付けされた結果リスト
    """
    if not results or not CohereConfig.API_KEY or cohere is None:
        return results[:top_k]

    try:
        co = cohere.Client(api_key=CohereConfig.API_KEY)
        
        # ドキュメントのテキストリストを作成
        documents = []
        for res in results:
            payload = res.get("payload", {})
            # QuestionとAnswerを組み合わせて文脈を作る
            doc_text = f"Question: {payload.get('question', '')}\nAnswer: {payload.get('answer', '')}"
            documents.append(doc_text)

        # Rerank実行
        rerank_response = co.rerank(
            model=CohereConfig.RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=len(documents)
        )

        # スコアを更新
        reranked_results = []
        for r in rerank_response.results:
            # 元の結果を取得 (indexで対応)
            original_result = results[r.index]
            new_score = r.relevance_score
            
            # スコアを更新した新しい辞書を作成
            new_result = original_result.copy()
            new_result["score"] = new_score
            
            # 閾値判定
            if new_score >= threshold:
                reranked_results.append(new_result)

        # スコア順はCohereが保証しているはずだが、念のためソート
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Re-ranking completed: {len(results)} -> {len(reranked_results)} results (Top score: {reranked_results[0]['score'] if reranked_results else 0.0:.4f})")
        
        return reranked_results[:top_k]

    except Exception as e:
        logger.error(f"Re-ranking failed: {e}")
        # 失敗時は元の結果をスコア順（RRF）で返す
        # RRFスコアは低い可能性があるため、警告を出すか、またはそのまま返す
        return results[:top_k]


def search_rag_knowledge_base(
    query: str,
    collection_name: Optional[str] = None
) -> str:
    """
    Qdrantデータベースから専門的な知識を検索します（Legacy String Output版）。
    """
    # デフォルトコレクションの解決（表示用）
    effective_collection = collection_name if collection_name else AgentConfig.RAG_DEFAULT_COLLECTION

    results = search_rag_knowledge_base_structured(query, collection_name)
    
    if isinstance(results, str): # Error or No Result strings
        return results
        
    formatted_results: List[str] = []
    for i, res in enumerate(results, 1):
        score: float = res.get("score", 0.0)
        payload: Dict[str, Any] = res.get("payload", {})
        q: str = payload.get("question", "N/A")
        a: str = payload.get("answer", "N/A")
        # source: str = payload.get("source", "unknown") # ファイル名は使用しない

        formatted_results.append(
            f"--- Result {i} [Score: {score:.4f}] ---\n"
            f"Q: {q}\n"
            f"A: {a}\n"
            f"Source: {effective_collection}\n"
        )

    if not formatted_results:
        return "[[NO_RAG_RESULT_LOW_SCORE]] 検索結果は見つかりましたが、関連性スコアが低すぎたため採用しませんでした。"

    return "\n".join(formatted_results)


def search_rag_knowledge_base_structured(
    query: str,
    collection_name: Optional[str] = None
) -> Union[List[Dict[str, Any]], str]:
    """
    Qdrantデータベースから専門的な知識を検索します（構造化データ版）。
    """
    if collection_name is None:
        collection_name = AgentConfig.RAG_DEFAULT_COLLECTION

    start_time: float = time.time()
    logger.info(f"ツールアクション(Structured): RAG検索を実行: query='{query}', collection='{collection_name}'")

    metrics: SearchMetrics = SearchMetrics(
        query=query,
        collection_name=collection_name,
        latency_ms=0.0,
        total_results=0,
        filtered_results=0,
        top_score=0.0
    )

    try:
        if not check_qdrant_health():
            raise QdrantConnectionError("Qdrantサーバーに接続できません。")

        existing_collections: List[str] = [c.name for c in client.get_collections().collections]
        if collection_name not in existing_collections:
            error_msg: str = f"コレクション '{collection_name}' はQdrantサーバーに存在しません。"
            logger.warning(error_msg)
            raise CollectionNotFoundError(error_msg)

        query_vector: List[float] = embed_query(query)
        if query_vector is None:
            raise EmbeddingError("クエリの埋め込み生成に失敗しました。")
        sparse_vector = embed_sparse_query_unified(query)

        # 1. Retrieval (Broad Search)
        # Re-rankingの効果を高めるため、最終的に欲しい数より多く取得する
        # Hybrid Search (RRF) を使用
        candidates: List[Dict[str, Any]] = search_collection(
            client=client,
            collection_name=collection_name,
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=20 # 候補を広げる
        )

        metrics.total_results = len(candidates) if candidates else 0

        if not candidates:
            metrics.latency_ms = (time.time() - start_time) * 1000.0
            _search_metrics_log.append(metrics)
            return f"[[NO_RAG_RESULT]] 検索結果が見つかりませんでした。コレクション: '{collection_name}'."

        # 2. Re-ranking (Cohere)
        # ここでスコアが「順位スコア(0.66...)」から「確率スコア(0.902...)」に変わる
        # Cohere APIキーがない場合は、ここでの変更は行われず、RRFスコアのままフィルタリングに進む
        # (ただし、RRFスコアは低いので threshold=0.5 で足切りされるリスクがあるため、
        #  rerank_results内でAPIキーがない場合のフォールバックを考慮する必要があるが、
        #  今回はAPI利用前提の設計となっている)
        reranked_results = rerank_results(query, candidates, top_k=AgentConfig.RAG_SEARCH_LIMIT)

        # 3. Metrics & Return
        scores: List[float] = [res.get("score", 0.0) for res in reranked_results]
        metrics.scores = scores
        metrics.top_score = max(scores) if scores else 0.0
        metrics.filtered_results = len(reranked_results)
        
        metrics.latency_ms = (time.time() - start_time) * 1000.0
        _search_metrics_log.append(metrics)

        if not reranked_results:
            return f"[[NO_RAG_RESULT_LOW_SCORE]] スコア閾値未満の結果のみでした。最高スコア: {metrics.top_score:.2f}"

        return reranked_results

    except Exception as e:
        logger.error(f"RAGツールエラー: {e}", exc_info=True)
        return f"[[RAG_TOOL_ERROR]] エラーが発生しました: {str(e)}"
