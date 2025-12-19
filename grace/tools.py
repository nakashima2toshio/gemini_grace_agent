"""
GRACE Tools - ツール定義

エージェントが使用するツール（RAG検索、推論、ask_user等）を定義
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from google import genai
from google.genai import types

from .config import get_config, GraceConfig

logger = logging.getLogger(__name__)


# =============================================================================
# ツール結果データクラス
# =============================================================================

@dataclass
class ToolResult:
    """ツール実行結果"""
    success: bool
    output: Any
    confidence_factors: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


# =============================================================================
# ツール基底クラス
# =============================================================================

class BaseTool(ABC):
    """ツール基底クラス"""

    name: str = "base_tool"
    description: str = "Base tool"

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """ツールを実行"""
        pass


# =============================================================================
# RAG検索ツール
# =============================================================================

class RAGSearchTool(BaseTool):
    """RAG検索ツール（Qdrant）"""

    name = "rag_search"
    description = "ベクトルDBから関連情報を検索"

    def __init__(
        self,
        config: Optional[GraceConfig] = None,
        qdrant_url: Optional[str] = None
    ):
        self.config = config or get_config()
        self.qdrant_url = qdrant_url or self.config.qdrant.url
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """Qdrantクライアントを取得（遅延初期化）"""
        if self._client is None:
            self._client = QdrantClient(url=self.qdrant_url)
        return self._client

    def execute(
        self,
        query: str,
        collection: Optional[str] = None,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> ToolResult:
        """
        RAG検索を実行

        Args:
            query: 検索クエリ
            collection: 検索対象コレクション
            limit: 取得件数上限
            score_threshold: スコア閾値

        Returns:
            ToolResult: 検索結果とConfidence計算用の統計情報
        """
        import time
        start_time = time.time()

        collection = collection or self.config.qdrant.collection_name
        limit = limit or self.config.qdrant.search_limit
        score_threshold = score_threshold or self.config.qdrant.score_threshold

        logger.info(f"RAG search: query='{query[:50]}...', collection={collection}")

        try:
            # 既存のqdrant_serviceを使用してクエリをベクトル化
            from services.qdrant_service import (
                embed_query_for_search,
                get_collection_embedding_params
            )

            # コレクションのEmbedding設定を取得
            params = get_collection_embedding_params(self.client, collection)
            model = params.get("model", "gemini-embedding-001")
            dims = params.get("dims", 3072)

            # クエリをベクトル化
            query_vector = embed_query_for_search(query, model=model, dims=dims)

            # 検索実行
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold
            )

            # 結果を整形
            search_results = []
            scores = []

            for result in results:
                scores.append(result.score)
                search_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })

            # Confidence計算用の統計情報
            confidence_factors = self._calculate_confidence_factors(scores)

            execution_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"RAG search completed: {len(search_results)} results, "
                f"avg_score={confidence_factors['avg_score']:.3f}"
            )

            return ToolResult(
                success=True,
                output=search_results,
                confidence_factors=confidence_factors,
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                confidence_factors={
                    "result_count": 0,
                    "avg_score": 0.0,
                    "score_variance": 1.0
                }
            )

    def _calculate_confidence_factors(self, scores: List[float]) -> Dict[str, Any]:
        """Confidence計算用の統計情報を算出"""
        if not scores:
            return {
                "result_count": 0,
                "avg_score": 0.0,
                "score_variance": 1.0,
                "max_score": 0.0,
                "min_score": 0.0
            }

        avg_score = sum(scores) / len(scores)

        # 分散計算
        if len(scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        else:
            variance = 0.0

        return {
            "result_count": len(scores),
            "avg_score": avg_score,
            "score_variance": variance,
            "max_score": max(scores),
            "min_score": min(scores)
        }


# =============================================================================
# 推論ツール
# =============================================================================

class ReasoningTool(BaseTool):
    """LLM推論ツール"""

    name = "reasoning"
    description = "収集した情報を分析・統合して回答を生成"

    def __init__(
        self,
        config: Optional[GraceConfig] = None,
        model_name: Optional[str] = None
    ):
        self.config = config or get_config()
        self.model_name = model_name or self.config.llm.model
        self.client = genai.Client()

    def execute(
        self,
        query: str,
        context: Optional[str] = None,
        sources: Optional[List[Dict]] = None,
        **kwargs
    ) -> ToolResult:
        """
        LLM推論を実行

        Args:
            query: 元のクエリ
            context: 追加コンテキスト
            sources: 参照ソース（RAG検索結果など）

        Returns:
            ToolResult: 生成された回答
        """
        import time
        start_time = time.time()

        logger.info(f"Reasoning: query='{query[:50]}...'")

        try:
            # プロンプト構築
            prompt = self._build_prompt(query, context, sources)

            # LLM呼び出し
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.config.llm.temperature,
                    max_output_tokens=self.config.llm.max_tokens,
                )
            )

            answer = response.text
            execution_time = int((time.time() - start_time) * 1000)

            # トークン使用量（利用可能な場合）
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                token_usage = {
                    "input_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "output_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                }

            logger.info(f"Reasoning completed: {len(answer)} chars")

            return ToolResult(
                success=True,
                output=answer,
                confidence_factors={
                    "has_sources": bool(sources),
                    "source_count": len(sources) if sources else 0,
                    "answer_length": len(answer),
                    "token_usage": token_usage
                },
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )

    def _build_prompt(
        self,
        query: str,
        context: Optional[str],
        sources: Optional[List[Dict]]
    ) -> str:
        """推論用プロンプトを構築"""
        prompt_parts = []

        # システム指示
        prompt_parts.append(
            "あなたは正確で役立つ回答を生成するアシスタントです。\n"
            "提供された情報を元に、ユーザーの質問に対して明確で具体的な回答を生成してください。\n"
        )

        # ソース情報
        if sources:
            prompt_parts.append("\n【参照情報】")
            for i, source in enumerate(sources, 1):
                payload = source.get("payload", {})
                score = source.get("score", 0)

                # ペイロードから関連情報を抽出
                question = payload.get("question", "")
                answer = payload.get("answer", "")
                content = payload.get("content", "")

                prompt_parts.append(f"\n--- ソース {i} (関連度: {score:.2f}) ---")
                if question:
                    prompt_parts.append(f"Q: {question}")
                if answer:
                    prompt_parts.append(f"A: {answer}")
                if content and not (question or answer):
                    prompt_parts.append(content[:500])

        # コンテキスト
        if context:
            prompt_parts.append(f"\n【追加コンテキスト】\n{context}")

        # ユーザーの質問
        prompt_parts.append(f"\n【ユーザーの質問】\n{query}")

        # 回答指示
        prompt_parts.append(
            "\n【回答指示】\n"
            "上記の情報を元に、ユーザーの質問に対して明確で具体的な回答を生成してください。"
            "情報が不足している場合は、その旨を伝えてください。"
        )

        return "\n".join(prompt_parts)


# =============================================================================
# Ask User ツール（HITL用）
# =============================================================================

class AskUserTool(BaseTool):
    """ユーザーに質問するツール（HITL）"""

    name = "ask_user"
    description = "ユーザーに追加情報や確認を求める"

    # Gemini Function Calling用の定義
    FUNCTION_DECLARATION = {
        "name": "ask_user_for_clarification",
        "description": """
ユーザーに追加情報を求めるツール。
以下の場合にのみ使用:
- 質問の意図が曖昧で、複数の解釈が可能
- 必要な情報が検索で見つからない
- 矛盾する情報があり、どちらを優先すべきか不明
""",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "ユーザーへの質問文（明確かつ簡潔に）"
                },
                "reason": {
                    "type": "string",
                    "description": "なぜこの質問が必要か（ユーザーに表示）"
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "選択肢がある場合のリスト（任意）"
                },
                "urgency": {
                    "type": "string",
                    "enum": ["blocking", "optional"],
                    "description": "blocking: 回答がないと進めない, optional: 推測で進めることも可能"
                }
            },
            "required": ["question", "reason", "urgency"]
        }
    }

    def execute(
        self,
        question: str,
        reason: str,
        urgency: str = "blocking",
        options: Optional[List[str]] = None,
        **kwargs
    ) -> ToolResult:
        """
        ユーザーに質問（実際のUIとの連携はExecutorで行う）

        Args:
            question: ユーザーへの質問
            reason: 質問の理由
            urgency: 緊急度（blocking/optional）
            options: 選択肢リスト

        Returns:
            ToolResult: 質問情報（回答はExecutorで処理）
        """
        logger.info(f"Ask user: {question} (urgency={urgency})")

        return ToolResult(
            success=True,
            output={
                "question": question,
                "reason": reason,
                "urgency": urgency,
                "options": options,
                "awaiting_response": True
            },
            confidence_factors={
                "requires_user_input": True,
                "urgency": urgency
            }
        )


# =============================================================================
# ツールレジストリ
# =============================================================================

class ToolRegistry:
    """ツールレジストリ"""

    def __init__(self, config: Optional[GraceConfig] = None):
        self.config = config or get_config()
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """デフォルトツールを登録"""
        enabled_tools = self.config.tools.enabled

        if "rag_search" in enabled_tools:
            self.register(RAGSearchTool(config=self.config))

        if "reasoning" in enabled_tools:
            self.register(ReasoningTool(config=self.config))

        if "ask_user" in enabled_tools:
            self.register(AskUserTool())

        logger.info(f"ToolRegistry initialized with: {list(self._tools.keys())}")

    def register(self, tool: BaseTool):
        """ツールを登録"""
        self._tools[tool.name] = tool
        logger.debug(f"Tool registered: {tool.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        """ツールを取得"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """登録済みツール名のリスト"""
        return list(self._tools.keys())

    def execute(self, name: str, **kwargs) -> ToolResult:
        """ツールを実行"""
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {name}"
            )
        return tool.execute(**kwargs)


# =============================================================================
# ファクトリ関数
# =============================================================================

def create_tool_registry(config: Optional[GraceConfig] = None) -> ToolRegistry:
    """ToolRegistryインスタンスを作成"""
    return ToolRegistry(config=config)


# =============================================================================
# エクスポート
# =============================================================================

__all__ = [
    # Data classes
    "ToolResult",

    # Base class
    "BaseTool",

    # Tools
    "RAGSearchTool",
    "ReasoningTool",
    "AskUserTool",

    # Registry
    "ToolRegistry",
    "create_tool_registry",
]