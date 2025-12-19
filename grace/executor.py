"""
GRACE Executor - 計画実行エージェント

生成された計画を順次実行し、結果を管理
"""

import logging
import time
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from .schemas import (
    ExecutionPlan,
    PlanStep,
    StepResult,
    ExecutionResult,
    StepStatus,
    create_plan_id,
)
from .tools import ToolRegistry, ToolResult, create_tool_registry
from .config import get_config, GraceConfig

logger = logging.getLogger(__name__)


# =============================================================================
# 実行状態管理
# =============================================================================

@dataclass
class ExecutionState:
    """実行状態管理"""

    plan: ExecutionPlan
    current_step_id: int = 0
    step_results: Dict[int, StepResult] = field(default_factory=dict)
    step_statuses: Dict[int, StepStatus] = field(default_factory=dict)
    overall_confidence: float = 0.0
    is_cancelled: bool = False
    is_paused: bool = False
    replan_count: int = 0
    max_replans: int = 3
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def __post_init__(self):
        """初期化後の処理"""
        # 全ステップをPENDINGで初期化
        for step in self.plan.steps:
            self.step_statuses[step.step_id] = StepStatus.PENDING

    def get_completed_outputs(self) -> Dict[int, str]:
        """完了済みステップの出力を取得"""
        return {
            step_id: result.output
            for step_id, result in self.step_results.items()
            if result.status == "success"
        }

    def get_completed_sources(self) -> List[str]:
        """完了済みステップのソースを取得"""
        sources = []
        for result in self.step_results.values():
            if result.status == "success" and result.sources:
                sources.extend(result.sources)
        return sources

    def can_replan(self) -> bool:
        """リプラン可能か判定"""
        return self.replan_count < self.max_replans and not self.is_cancelled

    def get_execution_time_ms(self) -> Optional[int]:
        """実行時間を取得（ミリ秒）"""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return int((end - self.start_time) * 1000)


# =============================================================================
# Executor クラス
# =============================================================================

class Executor:
    """計画実行エージェント"""

    def __init__(
        self,
        config: Optional[GraceConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        on_step_start: Optional[Callable[[PlanStep], None]] = None,
        on_step_complete: Optional[Callable[[StepResult], None]] = None,
        on_intervention_required: Optional[Callable[[str, Dict], Any]] = None,
    ):
        """
        Args:
            config: GRACE設定
            tool_registry: ツールレジストリ
            on_step_start: ステップ開始時のコールバック
            on_step_complete: ステップ完了時のコールバック
            on_intervention_required: 介入が必要な時のコールバック
        """
        self.config = config or get_config()
        self.tool_registry = tool_registry or create_tool_registry(self.config)

        # コールバック
        self.on_step_start = on_step_start
        self.on_step_complete = on_step_complete
        self.on_intervention_required = on_intervention_required

        logger.info("Executor initialized")

    def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        """
        計画全体を実行

        Args:
            plan: 実行計画

        Returns:
            ExecutionResult: 実行結果
        """
        logger.info(f"Executing plan: {plan.plan_id} with {len(plan.steps)} steps")

        # 状態を初期化
        state = ExecutionState(
            plan=plan,
            max_replans=self.config.replan.max_replans
        )
        state.start_time = time.time()

        # 各ステップを順次実行
        for step in plan.steps:
            if state.is_cancelled:
                logger.info("Execution cancelled")
                break

            if state.is_paused:
                logger.info("Execution paused")
                break

            # 依存ステップの完了確認
            if not self._check_dependencies(step, state):
                logger.warning(f"Step {step.step_id}: Dependencies not met, skipping")
                state.step_statuses[step.step_id] = StepStatus.SKIPPED
                continue

            # ステップ実行
            state.current_step_id = step.step_id
            state.step_statuses[step.step_id] = StepStatus.RUNNING

            # コールバック
            if self.on_step_start:
                self.on_step_start(step)

            # 実行
            result = self._execute_step(step, state)
            state.step_results[step.step_id] = result
            state.step_statuses[step.step_id] = StepStatus(result.status)

            # コールバック
            if self.on_step_complete:
                self.on_step_complete(result)

            # ask_user の場合は一時停止
            if step.action == "ask_user" and result.status == "success":
                output = result.output
                if isinstance(output, dict) and output.get("awaiting_response"):
                    state.is_paused = True
                    logger.info("Execution paused for user input")

                    # 介入コールバック
                    if self.on_intervention_required:
                        user_response = self.on_intervention_required(
                            "ask_user", output
                        )
                        if user_response:
                            # ユーザー応答を結果に追加
                            result.output = {**output, "user_response": user_response}
                            state.is_paused = False

            # 信頼度が低すぎる場合は早期終了
            if result.confidence < self.config.confidence.thresholds.confirm:
                logger.warning(
                    f"Step {step.step_id}: Low confidence ({result.confidence:.2f}), "
                    f"may need intervention"
                )

        # 完了処理
        state.end_time = time.time()

        # 全体の信頼度を計算
        state.overall_confidence = self._calculate_overall_confidence(state)

        # 結果を生成
        return self._create_execution_result(state)

    def _check_dependencies(self, step: PlanStep, state: ExecutionState) -> bool:
        """依存ステップの完了確認"""
        for dep_id in step.depends_on:
            if dep_id not in state.step_results:
                return False
            if state.step_results[dep_id].status == "failed":
                return False
        return True

    def _execute_step(self, step: PlanStep, state: ExecutionState) -> StepResult:
        """
        個別ステップの実行

        Args:
            step: 実行するステップ
            state: 現在の実行状態

        Returns:
            StepResult: ステップ実行結果
        """
        logger.info(f"Executing step {step.step_id}: {step.action} - {step.description}")
        start_time = time.time()

        try:
            # ツールを取得
            tool = self.tool_registry.get(step.action)
            if tool is None:
                raise ValueError(f"Unknown action: {step.action}")

            # ツール実行引数を準備
            kwargs = self._prepare_tool_kwargs(step, state)

            # 実行
            tool_result: ToolResult = tool.execute(**kwargs)

            # 実行時間
            execution_time = int((time.time() - start_time) * 1000)

            # 信頼度を計算
            confidence = self._calculate_step_confidence(tool_result, step)

            # ソースを抽出
            sources = self._extract_sources(tool_result)

            return StepResult(
                step_id=step.step_id,
                status="success" if tool_result.success else "failed",
                output=self._format_output(tool_result.output),
                confidence=confidence,
                sources=sources,
                error=tool_result.error,
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"Step {step.step_id} failed: {e}")
            execution_time = int((time.time() - start_time) * 1000)

            # フォールバック処理
            if step.fallback:
                logger.info(f"Attempting fallback: {step.fallback}")
                fallback_result = self._execute_fallback(step, state)
                if fallback_result.status == "success":
                    return fallback_result

            return StepResult(
                step_id=step.step_id,
                status="failed",
                output=None,
                confidence=0.0,
                error=str(e),
                execution_time_ms=execution_time
            )

    def _prepare_tool_kwargs(
        self,
        step: PlanStep,
        state: ExecutionState
    ) -> Dict[str, Any]:
        """ツール実行引数を準備"""
        kwargs = {
            "query": step.query or step.description,
        }

        if step.action == "rag_search":
            kwargs["collection"] = step.collection

        elif step.action == "reasoning":
            # 依存ステップの結果をコンテキストとして追加
            completed_outputs = state.get_completed_outputs()
            if completed_outputs:
                # RAG検索結果をソースとして渡す
                sources = []
                for dep_id in step.depends_on:
                    if dep_id in state.step_results:
                        dep_result = state.step_results[dep_id]
                        if isinstance(dep_result.output, list):
                            sources.extend(dep_result.output)
                        elif isinstance(dep_result.output, str):
                            # 文字列の場合はコンテキストとして追加
                            kwargs["context"] = dep_result.output
                if sources:
                    kwargs["sources"] = sources

        elif step.action == "ask_user":
            kwargs.update({
                "question": step.query or step.description,
                "reason": f"ステップ {step.step_id}: {step.description}",
                "urgency": "blocking"
            })

        return kwargs

    def _execute_fallback(
        self,
        step: PlanStep,
        state: ExecutionState
    ) -> StepResult:
        """フォールバックアクションを実行"""
        fallback_step = PlanStep(
            step_id=step.step_id,
            action=step.fallback,
            description=f"[Fallback] {step.description}",
            query=step.query,
            depends_on=step.depends_on,
            expected_output=step.expected_output,
            fallback=None  # 二重フォールバックは無し
        )
        return self._execute_step(fallback_step, state)

    def _calculate_step_confidence(
        self,
        tool_result: ToolResult,
        step: PlanStep
    ) -> float:
        """ステップの信頼度を計算"""
        if not tool_result.success:
            return 0.0

        factors = tool_result.confidence_factors

        # RAG検索の場合
        if step.action == "rag_search":
            result_count = factors.get("result_count", 0)
            avg_score = factors.get("avg_score", 0.0)

            if result_count == 0:
                return 0.2

            # 結果数とスコアに基づく信頼度
            count_factor = min(1.0, result_count / 5)
            return (avg_score * 0.7 + count_factor * 0.3)

        # 推論の場合
        elif step.action == "reasoning":
            has_sources = factors.get("has_sources", False)
            source_count = factors.get("source_count", 0)

            if not has_sources:
                return 0.5  # ソースなしの推論は信頼度中程度

            # ソース数に基づく信頼度
            return min(0.9, 0.6 + source_count * 0.1)

        # ask_user の場合
        elif step.action == "ask_user":
            return 0.5  # ユーザー入力待ちは中程度

        return 0.7  # デフォルト

    def _extract_sources(self, tool_result: ToolResult) -> List[str]:
        """ツール結果からソースを抽出"""
        sources = []

        if isinstance(tool_result.output, list):
            for item in tool_result.output:
                if isinstance(item, dict):
                    payload = item.get("payload", {})
                    source = payload.get("source", "")
                    if source and source not in sources:
                        sources.append(source)

        return sources

    def _format_output(self, output: Any) -> Optional[str]:
        """出力を文字列にフォーマット"""
        if output is None:
            return None
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            return str(output)
        if isinstance(output, list):
            # RAG検索結果の場合
            if output and isinstance(output[0], dict):
                return str(output)
            return "\n".join(str(item) for item in output)
        return str(output)

    def _calculate_overall_confidence(self, state: ExecutionState) -> float:
        """全体の信頼度を計算"""
        if not state.step_results:
            return 0.0

        confidences = [r.confidence for r in state.step_results.values()]
        return sum(confidences) / len(confidences)

    def _create_execution_result(self, state: ExecutionState) -> ExecutionResult:
        """実行結果を生成"""
        # 全体ステータスを判定
        statuses = [r.status for r in state.step_results.values()]

        if state.is_cancelled:
            overall_status = "cancelled"
        elif all(s == "success" for s in statuses):
            overall_status = "success"
        elif any(s == "success" for s in statuses):
            overall_status = "partial"
        else:
            overall_status = "failed"

        # 最終回答を取得（最後のreasoningステップの出力）
        final_answer = None
        for step in reversed(state.plan.steps):
            if step.action == "reasoning" and step.step_id in state.step_results:
                result = state.step_results[step.step_id]
                if result.status == "success":
                    final_answer = result.output
                    break

        return ExecutionResult(
            plan_id=state.plan.plan_id or create_plan_id(),
            original_query=state.plan.original_query,
            final_answer=final_answer,
            step_results=list(state.step_results.values()),
            overall_confidence=state.overall_confidence,
            overall_status=overall_status,
            replan_count=state.replan_count,
            total_execution_time_ms=state.get_execution_time_ms()
        )

    def cancel(self, state: ExecutionState):
        """実行をキャンセル"""
        state.is_cancelled = True
        logger.info("Execution cancelled")

    def resume(self, state: ExecutionState):
        """実行を再開"""
        state.is_paused = False
        logger.info("Execution resumed")


# =============================================================================
# ファクトリ関数
# =============================================================================

def create_executor(
    config: Optional[GraceConfig] = None,
    tool_registry: Optional[ToolRegistry] = None,
    **kwargs
) -> Executor:
    """Executorインスタンスを作成"""
    return Executor(
        config=config,
        tool_registry=tool_registry,
        **kwargs
    )


# =============================================================================
# エクスポート
# =============================================================================

__all__ = [
    "ExecutionState",
    "Executor",
    "create_executor",
]