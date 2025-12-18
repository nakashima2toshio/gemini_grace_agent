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


def search_rag_knowledge_base(
    query: str,
    collection_name: Optional[str] = None
) -> str:
    """
    Qdrantデータベースから専門的な知識を検索します。
    ユーザーが「仕様」「設定」「Wikipediaの知識」「事実確認」など、
    外部知識が必要な詳細について質問した場合にこのツールを使用してください。
    
    **重要: 一般的なプログラミング言語の文法や使い方に関する質問には、このツールを使用しないでください。**
    Args:
        query: 検索したいキーワードや質問文。
        collection_name: 検索対象のQdrantコレクション名。
    Returns:
        str: 検索されたドキュメントの内容（質問と回答のペア）。
    """
    if collection_name is None:
        collection_name = AgentConfig.RAG_DEFAULT_COLLECTION

    start_time: float = time.time()
    logger.info(f"ツールアクション: RAG検索を実行: query='{query}', collection='{collection_name}'")

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
            error_msg: str = f"コレクション '{collection_name}' はQdrantサーバーに存在しません。利用可能なコレクション: {existing_collections}"
            logger.warning(error_msg)
            raise CollectionNotFoundError(error_msg)

        query_vector: List[float] = embed_query(query) # Assuming embed_query returns List[float]
        if query_vector is None:
            raise EmbeddingError("クエリの埋め込み生成に失敗しました。")
            
        # Sparse Vector生成 (Hybrid Search用)
        # 常に生成するが、検索時にコレクション側が対応していなければ無視される可能性がある
        # エラーハンドリングは qdrant_client_wrapper 側で吸収することを期待
        sparse_vector = embed_sparse_query_unified(query)

        results: List[Dict[str, Any]] = search_collection( # Assuming search_collection returns List[Dict[str, Any]]
            client=client,
            collection_name=collection_name,
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=AgentConfig.RAG_SEARCH_LIMIT
        )

        metrics.total_results = len(results) if results else 0

        # 結果がない場合の詳細フィードバック
        if not results:
            metrics.latency_ms = (time.time() - start_time) * 1000.0
            _search_metrics_log.append(metrics)
            logger.info("検索結果: 0件")
            return (
                f"[[NO_RAG_RESULT]] 検索結果が見つかりませんでした。"
                f"コレクション: '{collection_name}'。"
                f"クエリ: '{query}'。"
            )

        scores: List[float] = [res.get("score", 0.0) for res in results]
        metrics.scores = scores
        metrics.top_score = max(scores) if scores else 0.0

        formatted_results: List[str] = []
        for i, res in enumerate(results, 1):
            score: float = res.get("score", 0.0)

            if score < AgentConfig.RAG_SCORE_THRESHOLD:
                continue

            payload: Dict[str, Any] = res.get("payload", {})
            q: str = payload.get("question", "N/A")
            a: str = payload.get("answer", "N/A")
            source: str = payload.get("source", "unknown")

            formatted_results.append(
                f"Result {i} (Score: {score:.2f}):\n"
                f"Q: {q}\n"
                f"A: {a}\n"
                f"Source: {source}"
            )

        metrics.filtered_results = len(formatted_results)
        metrics.latency_ms = (time.time() - start_time) * 1000.0
        _search_metrics_log.append(metrics)

        logger.info(
            f"検索完了: {metrics.filtered_results}/{metrics.total_results} results, "
            f"top_score={metrics.top_score:.2f}, latency={metrics.latency_ms:.1f}ms"
        )

        # 閾値以下の結果しかなかった場合の詳細フィードバック
        if not formatted_results:
            first_q = results[0].get("payload", {}).get("question", "N/A") if results else "N/A"
            return (
                f"[[NO_RAG_RESULT_LOW_SCORE]] 検索結果は見つかりましたが、関連性スコアが低すぎたため採用しませんでした。"
                f"コレクション: '{collection_name}'。"
                f"ヒット数 (閾値未満): {metrics.total_results}件。"
                f"最高スコア: {metrics.top_score:.2f}。"
                f"参考 (最高スコアのQ): '{first_q[:50]}...'。"
                f"クエリ: '{query}'。"
            )

        return "\n".join(formatted_results)

    except (QdrantConnectionError, CollectionNotFoundError, EmbeddingError) as e:
        logger.error(f"RAGツールエラー: {e}", exc_info=True)
        metrics.error = str(e)
        metrics.latency_ms = (time.time() - start_time) * 1000.0
        _search_metrics_log.append(metrics)
        return f"[[RAG_TOOL_ERROR]] エラーが発生しました: {str(e)}"
    except UnexpectedResponse as e:
        error_msg: str = f"Qdrantサーバーからの予期せぬ応答: {str(e)}"
        logger.error(error_msg, exc_info=True)
        metrics.error = error_msg
        metrics.latency_ms = (time.time() - start_time) * 1000.0
        _search_metrics_log.append(metrics)
        return f"[[RAG_TOOL_ERROR]] 検索中にQdrantサーバーエラーが発生しました: {str(e)}"
    except Exception as e:
        error_msg: str = f"予期せぬエラーが発生しました: {str(e)}"
        logger.error(error_msg, exc_info=True)
        metrics.error = error_msg
        metrics.latency_ms = (time.time() - start_time) * 1000.0
        _search_metrics_log.append(metrics)
        return f"[[RAG_TOOL_ERROR]] 検索中に予期せぬエラーが発生しました: {str(e)}"