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

logger = logging.getLogger(__name__)


# =============================================================================
# プロンプト定義
# =============================================================================

PLAN_GENERATION_PROMPT = """
あなたは計画策定の専門家です。ユーザーの質問を分析し、回答を生成するための実行計画を作成してください。

【利用可能なアクション】
- rag_search: ベクトルDB（Qdrant）から関連情報を検索
- web_search: Web検索（最新情報が必要な場合）
- reasoning: 収集した情報を分析・統合して回答を生成
- ask_user: ユーザーに追加情報や確認を求める

【計画作成のルール】
1. 最小限のステップで目標を達成すること（通常2-5ステップ）
2. 各ステップには明確な期待出力を設定
3. 依存関係を正しく設定（depends_onは先行ステップのIDのみ）
4. 失敗時の代替手段（fallback）を検討
5. 最後のステップは必ず "reasoning" で回答を生成

【計画の複雑度(complexity)の目安】
- 0.0-0.3: 単純な質問（1-2ステップ）
- 0.4-0.6: 中程度の質問（2-3ステップ）
- 0.7-1.0: 複雑な質問（4ステップ以上）

【requires_confirmationをtrueにする条件】
- 質問が曖昧で複数の解釈が可能な場合
- 実行に時間がかかる可能性がある場合
- 外部リソースへのアクセスが必要な場合

ユーザーの質問: {query}

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

        logger.info(f"Planner initialized with model: {self.model_name}")

    def create_plan(self, query: str) -> ExecutionPlan:
        """
        質問から実行計画を生成

        Args:
            query: ユーザーの質問

        Returns:
            ExecutionPlan: 生成された実行計画

        Raises:
            ValueError: 計画生成に失敗した場合
        """
        logger.info(f"Creating plan for query: {query[:50]}...")

        try:
            # Gemini APIで構造化出力を生成
            prompt = PLAN_GENERATION_PROMPT.format(query=query)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExecutionPlan,
                    temperature=self.config.llm.temperature,
                    max_output_tokens=self.config.llm.max_tokens,
                )
            )

            # レスポンスをパース
            plan = ExecutionPlan.model_validate_json(response.text)

            # 計画IDを設定
            plan.plan_id = create_plan_id()

            # 依存関係を検証
            errors = validate_plan_dependencies(plan)
            if errors:
                logger.warning(f"Plan dependency warnings: {errors}")

            logger.info(
                f"Plan created: {plan.plan_id} with {len(plan.steps)} steps, "
                f"complexity={plan.complexity:.2f}"
            )

            return plan

        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            # フォールバック: 単純な計画を生成
            return self._create_fallback_plan(query)

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
                    description="関連情報をRAG検索で取得",
                    query=query,
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