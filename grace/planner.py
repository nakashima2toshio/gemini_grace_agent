"""
GRACE Planner - 計画生成エージェント

ユーザーの質問を分析し、実行計画を生成
"""

import logging
from typing import Optional, List
from google import genai
from google.genai import types

from .schemas import (
    ExecutionPlan,
    PlanStep,
    create_plan_id,
    validate_plan_dependencies,
)
from .config import get_config, GraceConfig
from services.qdrant_service import get_all_collections
from qdrant_client import QdrantClient
from services.prompts import SEARCH_QUERY_INSTRUCTION
from regex_mecab import KeywordExtractor

logger = logging.getLogger(__name__)


# =============================================================================
# プロンプト定義
# =============================================================================

PLAN_GENERATION_PROMPT = f"""
あなたは計画策定の専門家です。ユーザーの質問を分析し、回答を生成するための実行計画を作成してください。

【利用可能なアクション】
- rag_search: ベクトルDB（Qdrant）から関連情報を検索
- reasoning: 収集した情報を分析・統合して回答を生成
- ask_user: ユーザーに追加情報や確認を求める

【利用可能なコレクション (rag_search用)】
{{available_collections}}

【コレクション選択のルール (重要)】
- **rag_search の collection 引数は、最初に "wikipedia_ja" を指定してください。**
- 一般的な知識・人物・事実の検索には "wikipedia_ja" が最適です。
- 他のコレクション（livedoor, cc_news, japanese_text）は、wikipedia_jaで見つからない場合にのみ使用してください。

【計画作成のルール】
1. 最小限のステップで目標を達成すること（通常2-5ステップ）
2. 各ステップには明確な期待出力を設定
3. 依存関係を正しく設定（depends_onは先行ステップのIDのみ）
4. 失敗時の代替手段（fallback）を検討
5. 最後のステップは必ず "reasoning" で回答を生成
6. コレクションは上記リストから最も適切なものを選択すること（存在しないコレクション名は使用不可）

{SEARCH_QUERY_INSTRUCTION}

【計画の複雑度(complexity)の目安】
- 0.0-0.3: 単純な質問（1-2ステップ）
- 0.4-0.6: 中程度の質問（2-3ステップ）
- 0.7-1.0: 複雑な質問（4ステップ以上）

【requires_confirmationをtrueにする条件】
- 質問が曖昧で複数の解釈が可能な場合
- 実行に時間がかかる可能性がある場合
- 外部リソースへのアクセスが必要な場合

ユーザーの質問: {{query}}

JSON形式で実行計画を出力してください。
"""

COMPLEXITY_ESTIMATION_PROMPT = """
以下の質問の複雑度を0.0から1.0の数値で評価してください。

評価基準:
- 0.0-0.2: 非常に単純（事実確認、定義の質問）
- 0.3-0.4: 単純（1つのトピックについての説明）
- 0.5-0.6: 中程度（比較、分析が必要）
- 0.7-0.8: 複雑（複数のソースからの情報統合が必要）
- 0.9-1.0: 非常に複雑（専門知識、多段階の推論が必要）

質問: {query}

数値のみを回答してください（例: 0.5）
"""


# =============================================================================
# Planner クラス
# =============================================================================

class Planner:
    """計画生成エージェント"""

    def __init__(
        self,
        config: Optional[GraceConfig] = None,
        model_name: Optional[str] = None
    ):
        """
        Args:
            config: GRACE設定（Noneの場合はデフォルト設定を使用）
            model_name: 使用するモデル名（Noneの場合は設定から取得）
        """
        self.config = config or get_config()
        self.model_name = model_name or self.config.llm.model
        self.client = genai.Client()
        
        # KeywordExtractorの初期化（Legacy Agentと同一）
        try:
            self.keyword_extractor = KeywordExtractor(prefer_mecab=True)
            logger.info("Planner: KeywordExtractor initialized")
        except Exception as e:
            logger.warning(f"Planner: Failed to initialize KeywordExtractor: {e}")
            self.keyword_extractor = None

        logger.info(f"Planner initialized with model: {self.model_name}")

    def create_plan(self, query: str) -> ExecutionPlan:
        """
        質問から実行計画を生成（LLM使用版 - 本来のロジック）

        Args:
            query: ユーザーの質問

        Returns:
            ExecutionPlan: LLMが生成した実行計画
        """
        logger.info(f"Creating execution plan for: {query[:50]}...")

        # --- Legacy Agentと同一の入力加工 ---
        augmented_query = query
        if self.keyword_extractor:
            try:
                keywords = self.keyword_extractor.extract(query, top_n=5)
                if keywords:
                    keywords_str = ", ".join(keywords)
                    augmented_query = f"{query}\n\n【重要: 検索クエリ作成の指示】\n以下の抽出された重要キーワードを、必ず検索クエリに含めてください。\n重要キーワード: {keywords_str}"
                    logger.info(f"Augmented query with keywords: {keywords_str}")
            except Exception as e:
                logger.warning(f"Keyword extraction failed: {e}")
        # ------------------------------------

        try:
            # 利用可能なコレクションを取得
            available_collections = self._get_available_collections()
            collections_str = ", ".join(available_collections) if available_collections else "(コレクションなし)"

            # 複雑度を推定
            complexity = self.estimate_complexity(query)

            # プロンプトを構築
            prompt = PLAN_GENERATION_PROMPT.format(
                available_collections=collections_str,
                query=augmented_query  # 加工済みクエリを使用
            ) + "\n\nIMPORTANT: Ensure the output is a valid, complete JSON object. Do not truncate the response."

            # --- [IPO LOG] PROCESS INPUT (GRACE PLANNER) ---
            logger.info(f"\n{'='*20} [GRACE PLANNER IPO: INPUT] {'='*20}\n{prompt}\n{'='*60}")

            # LLMで計画生成（JSON出力）
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExecutionPlan,
                    temperature=self.config.llm.temperature,
                    max_output_tokens=8192,
                )
            )

            # --- [IPO LOG] PROCESS OUTPUT (GRACE PLANNER) ---
            logger.info(f"\n{'='*20} [GRACE PLANNER IPO: OUTPUT] {'='*20}\n{response.text}\n{'='*60}")

            # JSONをパースしてExecutionPlanに変換
            plan = ExecutionPlan.model_validate_json(response.text)

            # 計画IDを設定
            plan.plan_id = create_plan_id()

            # 依存関係を検証
            errors = validate_plan_dependencies(plan)
            if errors:
                logger.warning(f"Plan validation errors: {errors}")
                # エラーがあってもフォールバックせず、警告のみ

            logger.info(
                f"Plan created: {len(plan.steps)} steps, "
                f"complexity={plan.complexity:.2f}, "
                f"requires_confirmation={plan.requires_confirmation}"
            )

            return plan

        except Exception as e:
            logger.error(f"Failed to create plan with LLM: {e}")
            logger.info("Falling back to simple plan")
            return self._create_fallback_plan(query)

    def _create_plan_legacy(self, query: str) -> ExecutionPlan:
        """
        質問から実行計画を生成（Legacy Agent委譲版 - バックアップ）
        """
        return ExecutionPlan(
            original_query=query,
            complexity=0.1,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[
                PlanStep(
                    step_id=1,
                    action="run_legacy_agent",
                    description="Legacy Agent (ReAct) を実行して回答を生成",
                    query=query,
                    expected_output="ユーザーへの回答",
                    fallback=None
                )
            ],
            success_criteria="ユーザーの質問に適切に回答できている",
            plan_id=create_plan_id()
        )

    def _get_available_collections(self) -> list:
        """利用可能なQdrantコレクションを取得"""
        try:
            client = QdrantClient(url=self.config.qdrant.url)
            cols = get_all_collections(client)
            return [c["name"] for c in cols]
        except Exception as e:
            logger.warning(f"Failed to get collections: {e}")
            return self.config.qdrant.search_priority  # デフォルトリストを返す

    def _create_fallback_plan(self, query: str) -> ExecutionPlan:
        """
        フォールバック用の単純な計画を生成

        Args:
            query: ユーザーの質問

        Returns:
            ExecutionPlan: 単純な2ステップ計画
        """
        logger.info("Creating fallback plan")

        return ExecutionPlan(
            original_query=query,
            complexity=0.5,
            estimated_steps=2,
            requires_confirmation=False,
            steps=[
                PlanStep(
                    step_id=1,
                    action="rag_search",
                    description="wikipedia_jaから関連情報を検索",
                    query=query,
                    collection="wikipedia_ja",  # 明示的にwikipedia_jaを指定
                    expected_output="関連するドキュメントや情報",
                    fallback="reasoning"
                ),
                PlanStep(
                    step_id=2,
                    action="reasoning",
                    description="取得した情報を元に回答を生成",
                    depends_on=[1],
                    expected_output="ユーザーへの回答"
                )
            ],
            success_criteria="ユーザーの質問に適切に回答できている",
            plan_id=create_plan_id()
        )

    def estimate_complexity(self, query: str) -> float:
        """
        質問の複雑度を推定（0.0-1.0）

        Args:
            query: ユーザーの質問

        Returns:
            float: 複雑度スコア
        """
        # キーワードベースの簡易推定
        complexity_factors = [
            ("比較", 0.15),
            ("違い", 0.15),
            ("複数", 0.2),
            ("最新", 0.1),
            ("理由", 0.1),
            ("方法", 0.1),
            ("詳しく", 0.15),
            ("ステップ", 0.1),
            ("手順", 0.1),
            ("なぜ", 0.1),
            ("どのように", 0.15),
        ]

        score = 0.3  # ベーススコア

        for keyword, weight in complexity_factors:
            if keyword in query:
                score += weight

        # 質問の長さも考慮
        if len(query) > 100:
            score += 0.1
        if len(query) > 200:
            score += 0.1

        return min(1.0, score)

    def estimate_complexity_with_llm(self, query: str) -> float:
        """
        LLMを使用して質問の複雑度を推定

        Args:
            query: ユーザーの質問

        Returns:
            float: 複雑度スコア
        """
        try:
            prompt = COMPLEXITY_ESTIMATION_PROMPT.format(query=query)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=10,
                )
            )

            complexity = float(response.text.strip())
            return min(1.0, max(0.0, complexity))

        except Exception as e:
            logger.warning(f"LLM complexity estimation failed: {e}")
            return self.estimate_complexity(query)

    def refine_plan(
        self,
        plan: ExecutionPlan,
        feedback: str
    ) -> ExecutionPlan:
        """
        フィードバックに基づいて計画を修正

        Args:
            plan: 元の計画
            feedback: ユーザーからのフィードバック

        Returns:
            ExecutionPlan: 修正された計画
        """
        logger.info(f"Refining plan {plan.plan_id} with feedback")

        refine_prompt = f"""
以下の実行計画をユーザーのフィードバックに基づいて修正してください。

【元の計画】
クエリ: {plan.original_query}
ステップ数: {len(plan.steps)}
ステップ: {[s.description for s in plan.steps]}

【ユーザーのフィードバック】
{feedback}

修正された計画をJSON形式で出力してください。
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=refine_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExecutionPlan,
                    temperature=self.config.llm.temperature,
                )
            )

            refined_plan = ExecutionPlan.model_validate_json(response.text)
            refined_plan.plan_id = create_plan_id()

            logger.info(f"Plan refined: {refined_plan.plan_id}")
            return refined_plan

        except Exception as e:
            logger.error(f"Failed to refine plan: {e}")
            return plan


# =============================================================================
# ファクトリ関数
# =============================================================================

def create_planner(
    config: Optional[GraceConfig] = None,
    model_name: Optional[str] = None
) -> Planner:
    """
    Plannerインスタンスを作成

    Args:
        config: GRACE設定
        model_name: 使用するモデル名

    Returns:
        Planner: Plannerインスタンス
    """
    return Planner(config=config, model_name=model_name)


# =============================================================================
# エクスポート
# =============================================================================

__all__ = [
    "Planner",
    "create_planner",
    "PLAN_GENERATION_PROMPT",
]
