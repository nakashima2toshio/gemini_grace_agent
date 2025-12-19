"""
GRACE Planner Tests
Plannerのテスト
"""

import pytest
from unittest.mock import MagicMock, patch
import json

from grace.planner import Planner, create_planner
from grace.schemas import ExecutionPlan, PlanStep
from grace.config import GraceConfig, reset_config


class TestPlanner:
    """Plannerのテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    @patch("grace.planner.genai.Client")
    def test_create_plan_success(self, mock_client_class):
        """計画生成の成功"""
        # モックレスポンスを設定
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "original_query": "Pythonについて教えて",
            "complexity": 0.5,
            "estimated_steps": 2,
            "requires_confirmation": False,
            "steps": [
                {
                    "step_id": 1,
                    "action": "rag_search",
                    "description": "関連情報を検索",
                    "query": "Python",
                    "expected_output": "検索結果"
                },
                {
                    "step_id": 2,
                    "action": "reasoning",
                    "description": "回答を生成",
                    "depends_on": [1],
                    "expected_output": "回答"
                }
            ],
            "success_criteria": "質問に回答できている"
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Plannerをテスト
        planner = Planner()
        plan = planner.create_plan("Pythonについて教えて")

        assert isinstance(plan, ExecutionPlan)
        assert plan.original_query == "Pythonについて教えて"
        assert len(plan.steps) == 2
        assert plan.plan_id is not None

    @patch("grace.planner.genai.Client")
    def test_create_plan_fallback(self, mock_client_class):
        """計画生成失敗時のフォールバック"""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        planner = Planner()
        plan = planner.create_plan("テスト")

        # フォールバック計画が返される
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "rag_search"
        assert plan.steps[1].action == "reasoning"

    def test_estimate_complexity_simple(self):
        """単純な質問の複雑度推定"""
        planner = Planner.__new__(Planner)  # __init__をスキップ
        planner.config = GraceConfig()

        # 単純な質問
        complexity = planner.estimate_complexity("Pythonとは何ですか")
        assert complexity < 0.5

    def test_estimate_complexity_complex(self):
        """複雑な質問の複雑度推定"""
        planner = Planner.__new__(Planner)
        planner.config = GraceConfig()

        # 複雑な質問
        complexity = planner.estimate_complexity(
            "PythonとJavaの違いを比較して、それぞれの利点と欠点を詳しく説明してください"
        )
        assert complexity > 0.5

    @patch("grace.planner.genai.Client")
    def test_refine_plan(self, mock_client_class):
        """計画の修正"""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "original_query": "テスト",
            "complexity": 0.5,
            "estimated_steps": 1,
            "requires_confirmation": False,
            "steps": [
                {
                    "step_id": 1,
                    "action": "reasoning",
                    "description": "修正された説明",
                    "expected_output": "結果"
                }
            ],
            "success_criteria": "テスト"
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        planner = Planner()

        original_plan = ExecutionPlan(
            original_query="テスト",
            complexity=0.5,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[
                PlanStep(
                    step_id=1,
                    action="rag_search",
                    description="元の説明",
                    expected_output="結果"
                )
            ],
            success_criteria="テスト"
        )

        refined_plan = planner.refine_plan(original_plan, "もっと詳しく")

        assert isinstance(refined_plan, ExecutionPlan)
        assert refined_plan.plan_id != original_plan.plan_id


class TestCreatePlanner:
    """create_planner関数のテスト"""

    @patch("grace.planner.genai.Client")
    def test_create_planner_default(self, mock_client_class):
        """デフォルト設定でのPlanner作成"""
        mock_client_class.return_value = MagicMock()

        planner = create_planner()

        assert isinstance(planner, Planner)
        assert planner.model_name == "gemini-2.0-flash"

    @patch("grace.planner.genai.Client")
    def test_create_planner_custom_model(self, mock_client_class):
        """カスタムモデルでのPlanner作成"""
        mock_client_class.return_value = MagicMock()

        planner = create_planner(model_name="custom-model")

        assert planner.model_name == "custom-model"