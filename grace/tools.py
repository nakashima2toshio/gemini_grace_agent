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

# Import wrappers for robust execution
from qdrant_client_wrapper import search_collection, embed_query_unified, embed_sparse_query_unified
from services.qdrant_service import get_collection_embedding_params

from .config import get_config, GraceConfig
from regex_mecab import KeywordExtractor

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
        
        # KeywordExtractorの初期化
        try:
            self.keyword_extractor = KeywordExtractor(prefer_mecab=True)
            logger.info("RAGSearchTool: KeywordExtractor initialized")
        except Exception as e:
            logger.warning(f"RAGSearchTool: Failed to initialize KeywordExtractor: {e}")
            self.keyword_extractor = None

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
        RAG検索を実行（Legacy Proven Logic委譲版）

        Args:
            query: 検索クエリ
            collection: 検索対象コレクション
            limit: 取得件数上限（現在はLegacy側に固定されているが、将来的に拡張可能）
            score_threshold: スコア閾値（現在はLegacy側のAgentConfigを参照）

        Returns:
            ToolResult: 検索結果
        """
        import time
        from agent_tools import search_rag_knowledge_base_structured
        
        start_time = time.time()
        logger.info(f"RAG search (Native): query='{query[:50]}...', collection={collection}")

        try:
            # 実績のあるLegacyロジック（キーワードフィルタリング等を含む）を呼び出し
            # results は List[Dict] または str (エラー/メッセージ)
            results = search_rag_knowledge_base_structured(query, collection)
            
            execution_time = int((time.time() - start_time) * 1000)

            if isinstance(results, str):
                # 結果なし、またはエラーメッセージの場合
                logger.info(f"RAG search returned message: {results}")
                
                # エラー文字列が含まれているかチェック
                is_error = "ERROR" in results or "FAILED" in results
                
                return ToolResult(
                    success=False,
                    output=[],
                    error=results if is_error else None,
                    confidence_factors={
                        "result_count": 0,
                        "avg_score": 0.0,
                        "message": results
                    },
                    execution_time_ms=execution_time
                )

            # 成功時
            scores = [r.get("score", 0.0) for r in results]
            confidence_factors = self._calculate_confidence_factors(scores)
            
            logger.info(f"RAG search SUCCESS: {len(results)} results found via Legacy Logic.")

            return ToolResult(
                success=True,
                output=results,
                confidence_factors=confidence_factors,
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"RAG search failed in Tool wrapper: {e}", exc_info=True)
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

            # --- [IPO LOG] PROCESS INPUT (GRACE REASONING) ---
            logger.info(f"\n{'='*20} [GRACE REASONING IPO: INPUT] {'='*20}\n{prompt}\n{'='*60}")

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

            # --- [IPO LOG] PROCESS OUTPUT (GRACE REASONING) ---
            logger.info(f"\n{'='*20} [GRACE REASONING IPO: OUTPUT] {'='*20}\n{answer}\n{'='*60}")

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
        """
        推論用プロンプトを構築（高度化版）
        Legacy Agentの知見を活かした指示セットを使用。
        """
        prompt_parts = []

        # システム指示
        prompt_parts.append(
            "あなたは社内ドキュメント検索システムと連携した「ハイブリッド・ナレッジ・エージェント」です。\n"
            "提供された【参照情報】を元に、ユーザーの質問に対して正確で誠実な回答を生成してください。\n"
        )

        # ソース情報（RAG結果）
        if sources:
            prompt_parts.append("\n### 【参照情報】")
            for i, source in enumerate(sources, 1):
                payload = source.get("payload", {})
                score = source.get("score", 0)
                col = source.get("collection", "unknown")

                question = payload.get("question", "")
                answer = payload.get("answer", "")
                content = payload.get("content", "")
                src_file = payload.get("source", "unknown")

                prompt_parts.append(f"\n--- 情報源 {i} (信頼度: {score:.2f}, コレクション: {col}) ---")
                if question:
                    prompt_parts.append(f"Q: {question}")
                if answer:
                    prompt_parts.append(f"A: {answer}")
                if content and not (question or answer):
                    prompt_parts.append(content[:1000])
                prompt_parts.append(f"出典: {src_file}")

        # 追加コンテキスト（他ステップの結果など）
        if context:
            prompt_parts.append(f"\n### 【補足コンテキスト】\n{context}")

        # ユーザーの質問
        prompt_parts.append(f"\n### 【ユーザーの質問】\n{query}")

        # 回答のルール
        prompt_parts.append(
            "\n### 【回答の構成ルール（最重要）】\n"
            "1. **正確性と誠実さ**: 参照情報にある事実のみを述べてください。情報がない場合は「提供された情報源には見当たりませんでした」と正直に回答してください。\n"
            "2. **判明した事実を優先**: 質問に対する直接的な回答が見つかった場合は、それを最初に簡潔に述べてください。\n"
            "3. **出典の明示**: 回答の根拠となった情報がある場合、「社内ナレッジ（出典ファイル名）によると...」の形式で出典を明示してください。\n"
            "4. **丁寧な日本語**: です・ます調で、読みやすく構造化（箇条書き等）して回答してください。\n"
            "5. **捏造禁止**: あなた自身の事前知識で情報を補完したり、勝手な推測で回答を作成したりしないでください。\n"
            "\n上記のルールに従い、プロフェッショナルな回答を生成してください。"
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