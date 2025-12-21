# agent_tools.py

import os
import time
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from qdrant_client_wrapper import search_collection, embed_query, embed_sparse_query_unified, QDRANT_CONFIG
from config import AgentConfig

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

        results: List[Dict[str, Any]] = search_collection(
            client=client,
            collection_name=collection_name,
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=AgentConfig.RAG_SEARCH_LIMIT
        )

        metrics.total_results = len(results) if results else 0

        if not results:
            metrics.latency_ms = (time.time() - start_time) * 1000.0
            _search_metrics_log.append(metrics)
            return f"[[NO_RAG_RESULT]] 検索結果が見つかりませんでした。コレクション: '{collection_name}'."

        scores: List[float] = [res.get("score", 0.0) for res in results]
        metrics.scores = scores
        metrics.top_score = max(scores) if scores else 0.0

        # ============ 共通フィルタリングロジックの適用 ============
        filtered_results = filter_results_by_keywords(results, query)

        # スコア閾値適用
        final_results = [r for r in filtered_results if r.get("score", 0.0) >= AgentConfig.RAG_SCORE_THRESHOLD]

        metrics.filtered_results = len(final_results)
        metrics.latency_ms = (time.time() - start_time) * 1000.0
        _search_metrics_log.append(metrics)

        if not final_results:
            if filtered_results:
                return f"[[NO_RAG_RESULT_LOW_SCORE]] スコア閾値未満の結果のみでした。最高スコア: {metrics.top_score:.2f}"
            return f"[[RAG_SEARCH_FAILED]] 必須キーワードを含む結果が見つかりませんでした。"

        return final_results

    except Exception as e:
        logger.error(f"RAGツールエラー: {e}", exc_info=True)
        return f"[[RAG_TOOL_ERROR]] エラーが発生しました: {str(e)}"
