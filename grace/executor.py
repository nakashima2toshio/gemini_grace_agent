"""
GRACE Executor - 計画実行エージェント

生成された計画を順次実行し、結果を管理
"""

import logging
import time
from typing import Dict, Optional, List, Callable, Any, Generator
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
from .confidence import (
    ConfidenceCalculator,
    ConfidenceFactors,
    ConfidenceScore,
    LLMSelfEvaluator,
    ConfidenceAggregator,
    ActionDecision,
    InterventionLevel,
    create_confidence_calculator,
    create_llm_evaluator,
    create_confidence_aggregator,
    create_query_coverage_calculator,
)
from .intervention import (
    InterventionHandler,
    InterventionRequest,
    InterventionResponse,
    InterventionAction,
    create_intervention_handler,
)

# === Legacy Agent Integration ===
try:
    from services.agent_service import ReActAgent, get_available_collections_from_qdrant_helper
    LEGACY_AGENT_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__) # Ensure logger exists before warning
    logger.warning("Failed to import services.agent_service. Legacy agent execution will fail.")
    LEGACY_AGENT_AVAILABLE = False
# ================================

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
    intervention_request: Optional[Any] = None  # InterventionRequest
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

from .replan import ReplanOrchestrator, create_replan_orchestrator

class Executor:
    """計画実行エージェント（GRACEネイティブ実装）"""

    def __init__(
        self,
        config: Optional[GraceConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        on_step_start: Optional[Callable[[PlanStep], None]] = None,
        on_step_complete: Optional[Callable[[StepResult], None]] = None,
        on_intervention_required: Optional[Callable[[str, Dict], Any]] = None,
        on_confidence_update: Optional[Callable[[ConfidenceScore, ActionDecision], None]] = None,
        on_replan: Optional[Callable[[str, int], None]] = None,
        replan_orchestrator: Optional[ReplanOrchestrator] = None,
        enable_replan: bool = True,
    ):
        self.config = config or get_config()

        # ToolRegistry（指定がなければデフォルト作成）
        self.tool_registry = tool_registry or create_tool_registry(config=self.config)

        # Confidence関連コンポーネント（Phase 2）
        self.confidence_calculator = create_confidence_calculator(config=self.config)
        self.llm_evaluator = create_llm_evaluator(config=self.config)
        self.query_coverage_calculator = create_query_coverage_calculator(config=self.config)
        self.confidence_aggregator = create_confidence_aggregator(config=self.config)

        # コールバック
        self.on_step_start = on_step_start
        self.on_step_complete = on_step_complete
        self.on_intervention_required = on_intervention_required
        self.on_confidence_update = on_confidence_update
        self.on_replan = on_replan  # リプラン発生時のコールバック

        # InterventionHandler（Phase 3）
        self.intervention_handler = create_intervention_handler(
            config=self.config,
            on_notify=self._handle_intervention_notify,
            on_confirm=self._handle_intervention_confirm,
            on_escalate=self._handle_intervention_escalate,
        )

        # ReplanOrchestrator（Phase 4）
        if replan_orchestrator is not None:
            self.replan_orchestrator = replan_orchestrator
        elif enable_replan:
            self.replan_orchestrator = create_replan_orchestrator(config=self.config)
        else:
            self.replan_orchestrator = None

        # ステップごとのConfidenceScoreを保持
        self.step_confidence_scores: Dict[int, ConfidenceScore] = {}

        replan_status = "enabled" if self.replan_orchestrator else "disabled"
        logger.info(
            f"Executor (GRACE Native) initialized: "
            f"tools={self.tool_registry.list_tools()}, replan={replan_status}"
        )

    def execute_plan_generator(
        self,
        plan: ExecutionPlan,
        state: Optional[ExecutionState] = None
    ) -> Generator[ExecutionState, None, ExecutionResult]:
        """
        計画をステップごとに実行（ジェネレータ版）
        
        UIなどで進捗をリアルタイム表示するために使用
        
        Args:
            plan: 実行する計画
            state: 既存の状態（再開時などに指定）
            
        Yields:
            ExecutionState: 各ステップ完了後の状態
            
        Returns:
            ExecutionResult: 最終実行結果
        """
        logger.info(f"Executing plan (generator): {plan.plan_id}, steps={len(plan.steps)}")

        # 実行状態を初期化（未指定の場合）
        if state is None:
            state = ExecutionState(plan=plan)
            state.start_time = time.time()
            
        try:
            # 各ステップを順次実行
            # 注: リプランなどでステップ数が増減する可能性があるため、インデックス管理が必要
            # ここでは簡易的に、現在のステップID以降を実行するロジック
            
            # 実行すべきステップのリストを取得（現在の計画に基づく）
            # 既に完了しているステップはスキップ
            steps_to_execute = [
                s for s in plan.steps 
                if state.step_statuses.get(s.step_id) != StepStatus.SUCCESS
            ]
            
            for step in steps_to_execute:
                # 状態更新: 現在のステップID
                state.current_step_id = step.step_id
                
                # キャンセルチェック
                if state.is_cancelled:
                    logger.info("Execution cancelled")
                    break

                # 依存関係チェック
                if not self._check_dependencies(step, state):
                    logger.warning(f"Step {step.step_id}: Dependencies not met, skipping")
                    state.step_statuses[step.step_id] = StepStatus.SKIPPED
                    yield state
                    continue

                # ステップ開始コールバック
                state.step_statuses[step.step_id] = StepStatus.RUNNING
                if self.on_step_start:
                    self.on_step_start(step)
                
                # ステップ実行
                # _execute_step は StepResult または Generator[Any, None, StepResult] を返す可能性がある
                step_execution = self._execute_step(step, state)
                
                result = None
                if isinstance(step_execution, Generator):
                    # ジェネレータの場合はイベントを中継し、最終結果(return value)を取得
                    # yield from は return value を返す
                    result = yield from step_execution
                else:
                    # 直接結果が返ってきた場合
                    result = step_execution

                # 結果を保存
                state.step_results[step.step_id] = result
                state.step_statuses[step.step_id] = (
                    StepStatus.SUCCESS if result.status == "success" else StepStatus.FAILED
                )

                # ステップ完了コールバック
                if self.on_step_complete:
                    self.on_step_complete(result)

                # 信頼度に基づく介入チェック (Phase 3)
                if step.step_id in self.step_confidence_scores:
                    confidence_score = self.step_confidence_scores[step.step_id]
                    action_decision = self.confidence_calculator.decide_action(confidence_score)
                    
                    # CONFIRM または ESCALATE の場合は一時停止
                    if action_decision.level in [InterventionLevel.CONFIRM, InterventionLevel.ESCALATE]:
                        logger.info(f"Pausing for intervention: {action_decision.level} (Step {step.step_id})")
                        
                        state.is_paused = True
                        
                        # 介入リクエストを作成
                        req_type = "confirm" if action_decision.level == InterventionLevel.CONFIRM else "escalate"
                        message = f"信頼度が低いため確認が必要です ({confidence_score.score:.2f})"
                        if action_decision.reason:
                            message += f"\n理由: {action_decision.reason}"

                        # InterventionRequestオブジェクトを作成
                        state.intervention_request = InterventionRequest(
                            level=action_decision.level,
                            step_id=step.step_id,
                            message=message,
                            reason=action_decision.reason,
                            confidence_score=confidence_score.score,
                            plan=plan
                        )
                        
                        # Yield: 一時停止状態を通知
                        yield state
                        
                        # ジェネレータを終了（再開時は新しいジェネレータを作成）
                        return self._create_execution_result(state)

                    # 通知のみ（SILENT/NOTIFY）
                    self._handle_intervention_if_needed(action_decision, step, state)

                # Yield: ステップ完了状態を通知
                yield state

                # ask_user の場合の処理（既存ロジック）
                if step.action == "ask_user" and result.status == "success":
                     # ...既存のask_user処理...
                     pass

                # 失敗時のリプラン
                if result.status == "failed" and self.replan_orchestrator:
                    replan_result = self.replan_orchestrator.handle_step_failure(
                        step_result=result,
                        current_plan=plan,
                        completed_results=state.step_results,
                        replan_count=state.replan_count
                    )
                    if replan_result and replan_result.success and replan_result.new_plan:
                        logger.info(f"Replanning: {replan_result.reason}")
                        state.replan_count += 1
                        
                        # 新しい計画に差し替え
                        # Generatorを再帰呼び出しするか、ループを再構成する必要がある
                        # ここでは、新しい計画で再帰的にGeneratorを作成し、その値をYieldする
                        state.plan = replan_result.new_plan
                        # 再帰呼び出し
                        yield from self.execute_plan_generator(replan_result.new_plan, state)
                        # 再帰から戻ったら終了（新しい計画が完了しているため）
                        return self._create_execution_result(state)

            # 全体の信頼度を計算
            state.overall_confidence = self._calculate_overall_confidence(state)
            state.end_time = time.time()

            # 最終結果
            return self._create_execution_result(state)

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            state.end_time = time.time()
            # エラー時も結果を返す
            return ExecutionResult(
                plan_id=plan.plan_id or create_plan_id(),
                original_query=plan.original_query,
                final_answer=f"実行エラー: {str(e)}",
                step_results=list(state.step_results.values()),
                overall_confidence=0.0,
                overall_status="failed",
                replan_count=state.replan_count,
                total_execution_time_ms=state.get_execution_time_ms()
            )

    def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        """
        計画を実行（GRACEネイティブ実装）

        Args:
            plan: 実行する計画

        Returns:
            ExecutionResult: 実行結果
        """
        logger.info(f"Executing plan: {plan.plan_id}, steps={len(plan.steps)}")

        # 実行状態を初期化
        state = ExecutionState(plan=plan)
        state.start_time = time.time()

        try:
            # 各ステップを順次実行
            for step in plan.steps:
                # キャンセルチェック
                if state.is_cancelled:
                    logger.info("Execution cancelled")
                    break

                # 依存関係チェック
                if not self._check_dependencies(step, state):
                    logger.warning(f"Step {step.step_id}: Dependencies not met, skipping")
                    state.step_statuses[step.step_id] = StepStatus.SKIPPED
                    continue

                # ステップ開始コールバック
                state.step_statuses[step.step_id] = StepStatus.RUNNING
                if self.on_step_start:
                    self.on_step_start(step)

                # ステップ実行
                result = self._execute_step(step, state)

                # 結果を保存
                state.step_results[step.step_id] = result
                state.step_statuses[step.step_id] = (
                    StepStatus.SUCCESS if result.status == "success" else StepStatus.FAILED
                )

                # ステップ完了コールバック
                if self.on_step_complete:
                    self.on_step_complete(result)

                # ask_user の場合、介入が必要
                if step.action == "ask_user" and result.status == "success":
                    if self.on_intervention_required and isinstance(result.output, str):
                        try:
                            output_data = eval(result.output) if result.output.startswith("{}") else {"question": result.output}
                        except Exception:
                            output_data = {"question": result.output}

                        user_response = self.on_intervention_required("ask_user", output_data)
                        if user_response:
                            # ユーザー応答を次のステップで利用可能にする
                            result.output = f"ユーザー応答: {user_response}"
                            state.step_results[step.step_id] = result

                # 失敗時のリプラン（Phase 4で有効化）
                if result.status == "failed" and self.replan_orchestrator:
                    replan_result = self.replan_orchestrator.handle_step_failure(
                        step_result=result,
                        current_plan=plan,
                        completed_results=state.step_results,
                        replan_count=state.replan_count
                    )
                    if replan_result and replan_result.success and replan_result.new_plan:
                        logger.info(f"Replanning: {replan_result.reason}")
                        state.replan_count += 1
                        # 新しい計画で再実行（再帰）
                        return self.execute_plan(replan_result.new_plan)

            # 全体の信頼度を計算
            state.overall_confidence = self._calculate_overall_confidence(state)
            state.end_time = time.time()

            # 実行結果を生成
            return self._create_execution_result(state)

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            state.end_time = time.time()

            return ExecutionResult(
                plan_id=plan.plan_id or create_plan_id(),
                original_query=plan.original_query,
                final_answer=f"実行エラー: {str(e)}",
                step_results=list(state.step_results.values()),
                overall_confidence=0.0,
                overall_status="failed",
                replan_count=state.replan_count,
                total_execution_time_ms=state.get_execution_time_ms()
            )

    def _check_dependencies(self, step: PlanStep, state: ExecutionState) -> bool:
        """依存ステップの完了確認"""
        for dep_id in step.depends_on:
            if dep_id not in state.step_results:
                return False
            if state.step_results[dep_id].status == "failed":
                return False
        return True

    def _execute_step(self, step: PlanStep, state: ExecutionState) -> Any:
        """
        個別ステップの実行

        Args:
            step: 実行するステップ
            state: 現在の実行状態

        Returns:
            StepResult or Generator: ステップ実行結果（またはジェネレータ）
        """
        logger.info(f"Executing step {step.step_id}: {step.action} - {step.description}")
        start_time = time.time()

        try:
            # ツールを取得
            tool = self.tool_registry.get(step.action)
            
            # --- 互換性維持のための特別なハンドリング ---
            if tool is None and step.action == "run_legacy_agent":
                # ツールとして登録されていないが、以前のLegacyプランが残っている場合
                return self._execute_legacy_agent_step(step, state, start_time)
            
            if tool is None:
                raise ValueError(f"Unknown action: {step.action}")

            # ツール実行引数を準備
            kwargs = self._prepare_tool_kwargs(step, state)

            # 実行
            tool_result: ToolResult = tool.execute(**kwargs)

            # 実行時間
            execution_time = int((time.time() - start_time) * 1000)

            # 信頼度を計算（state引数を渡す）
            confidence = self._calculate_step_confidence(tool_result, step, state)

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

    def _execute_legacy_agent_step(self, step: PlanStep, state: ExecutionState, start_time: float) -> Generator[Any, None, StepResult]:
        """Legacy ReActAgent を使用したステップ実行（ジェネレータ版）"""
        if not LEGACY_AGENT_AVAILABLE:
            raise ImportError("agent_service module not found")

        # 1. コレクション準備
        available_collections = get_available_collections_from_qdrant_helper()
        if not available_collections:
             available_collections = self.config.qdrant.search_priority

        # 2. Agent初期化
        agent = ReActAgent(
            selected_collections=available_collections,
            model_name=self.config.llm.model
        )

        query = step.query or step.description
        logger.info(f"Running Legacy Agent with query: {query}")

        final_answer = ""
        sources = []
        
        # 3. エージェント実行（ジェネレータ）
        # ストリーミングイベントを拾いながら、ツール結果からソースを収集
        for event in agent.execute_turn(query):
            # イベントをそのまま上位へ流す（UI表示用）
            yield event

            # ログ出力（デバッグ用）
            if event["type"] == "log":
                logger.info(f"[LegacyAgent] {event['content']}")
            elif event["type"] == "tool_call":
                logger.info(f"[LegacyAgent] Tool Call: {event['name']} args={event['args']}")
            elif event["type"] == "tool_result":
                logger.info(f"[LegacyAgent] Tool Result (len={len(event['content'])})")
                # ソース抽出 (簡易的な文字列解析)
                if "Source:" in event["content"]:
                     import re
                     # Source: filename.csv のパターンを抽出
                     found_sources = re.findall(r"Source:\s*([a-zA-Z0-9_.\-]+)", event["content"])
                     if found_sources:
                         sources.extend(found_sources)
            elif event["type"] == "final_answer":
                final_answer = event["content"]
        
        # 4. 結果構築
        execution_time = int((time.time() - start_time) * 1000)
        
        # ソースの重複排除
        sources = list(set(sources))
        
        # Confidence計算 (簡易版)
        confidence = 0.8 if final_answer and "申し訳ありません" not in final_answer else 0.3
        
        # ConfidenceScoreオブジェクトを作成して保存
        conf_score_obj = ConfidenceScore(
             score=confidence,
             factors=ConfidenceFactors(
                 source_count=len(sources),
                 search_result_count=len(sources), 
                 llm_self_confidence=confidence
             )
        )
        self.step_confidence_scores[step.step_id] = conf_score_obj
        
        # アクション判定
        if self.on_confidence_update:
             action = self.confidence_calculator.decide_action(conf_score_obj)
             self.on_confidence_update(conf_score_obj, action)

        return StepResult(
            step_id=step.step_id,
            status="success",
            output=final_answer,
            confidence=confidence,
            sources=sources,
            error=None,
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
            context_parts = []
            sources = []

            for dep_id in step.depends_on:
                if dep_id in state.step_results:
                    dep_result = state.step_results[dep_id]
                    dep_output = dep_result.output

                    if dep_output:
                        # 文字列化されたリストを復元する試み
                        if isinstance(dep_output, str):
                            try:
                                # RAG検索結果は "[{...}, {...}]" 形式で文字列化されている
                                if dep_output.startswith("[{") or dep_output.startswith("[{'"):
                                    import ast
                                    parsed = ast.literal_eval(dep_output)
                                    if isinstance(parsed, list):
                                        sources.extend(parsed)
                                        continue
                            except (ValueError, SyntaxError):
                                pass
                            # パースできない場合はコンテキストとして追加
                            context_parts.append(f"--- 参照情報 (Step {dep_id}) ---\n{dep_output}")
                        elif isinstance(dep_output, list):
                            sources.extend(dep_output)

            if sources:
                kwargs["sources"] = sources
            if context_parts:
                kwargs["context"] = "\n\n".join(context_parts)

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
        step: PlanStep,
        state: ExecutionState
    ) -> float:
        """
        ステップの信頼度を計算（ConfidenceCalculator使用）

        Args:
            tool_result: ツール実行結果
            step: 実行したステップ
            state: 現在の実行状態

        Returns:
            float: 信頼度スコア (0.0-1.0)
        """
        if not tool_result.success:
            return 0.0

        factors = tool_result.confidence_factors

        # ConfidenceFactorsを構築
        # source_countの決定: ツールが明示的に返した値を優先
        extracted_sources = self._extract_sources(tool_result)
        source_count = factors.get("source_count", len(extracted_sources))

        # 依存ステップからのスコア継承ロジック
        current_result_count = factors.get("result_count", 0)
        current_max_score = factors.get("max_score", factors.get("avg_score", 0.0))
        current_avg_score = factors.get("avg_score", 0.0)

        # 自身で検索しておらず、かつ推論ステップなどの場合、依存元のスコアを引き継ぐ
        if current_result_count == 0 and not (step.action in ["rag_search", "web_search"]):
            inherited_max = 0.0
            inherited_found = False
            for dep_id in step.depends_on:
                if dep_id in state.step_results:
                    dep_res = state.step_results[dep_id]
                    # 依存先の信頼度を継承
                    if dep_res.confidence > inherited_max:
                        inherited_max = dep_res.confidence
                        inherited_found = True
            
            if inherited_found:
                current_max_score = inherited_max
                current_avg_score = inherited_max
                current_result_count = 1  # 仮想的に1件あったとみなす

        confidence_factors = ConfidenceFactors(
            # RAG検索関連
            search_result_count=current_result_count,
            search_avg_score=current_avg_score,
            search_max_score=current_max_score,
            search_score_variance=factors.get("score_variance", 1.0),
            # ソース関連
            source_count=source_count,
            source_agreement=0.5,  # 単一ソースの場合はデフォルト
            # ツール実行関連
            tool_success_rate=1.0 if tool_result.success else 0.0,
            tool_execution_count=1,
            tool_success_count=1 if tool_result.success else 0,
            # ステップタイプ
            is_search_step=(step.action in ["rag_search", "web_search"])
        )

        # ConfidenceCalculatorで計算
        confidence_score = self.confidence_calculator.calculate(confidence_factors)

        # ステップごとのConfidenceScoreを保存
        self.step_confidence_scores[step.step_id] = confidence_score

        # アクション決定を取得
        action_decision = self.confidence_calculator.decide_action(confidence_score)

        # コールバックで通知（Phase 3のHITLと連携）
        if self.on_confidence_update:
            self.on_confidence_update(confidence_score, action_decision)

        logger.info(
            f"Step {step.step_id} confidence: {confidence_score.score:.2f} "
            f"(level={confidence_score.level}, action={action_decision.level.value})"
        )

        return confidence_score.score

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
        """
        全体の信頼度を計算（ConfidenceAggregator + LLMSelfEvaluator使用）

        Args:
            state: 実行状態

        Returns:
            float: 全体の信頼度スコア (0.0-1.0)
        """
        if not state.step_results:
            return 0.0

        # 各ステップのConfidenceScoreを収集
        step_scores = list(self.step_confidence_scores.values())
        
        # 最新のbreakdownを取得（ベースとして使用）
        current_breakdown = {}
        if step_scores:
            # 最後のステップのbreakdownをコピー
            current_breakdown = step_scores[-1].breakdown.copy()

        # 最終回答を取得
        final_answer = None
        # ... (中略) ...

        # LLMSelfEvaluatorで最終回答を評価（オプション）
        if final_answer:
            # 1. LLM自己評価 (Accuracy/Style etc.)
            try:
                eval_result = self.llm_evaluator.evaluate(
                    query=state.plan.original_query,
                    answer=final_answer,
                    sources=state.get_completed_sources()
                )
                
                score_val = 0.0
                if hasattr(eval_result, 'score'):
                    score_val = eval_result.score
                elif isinstance(eval_result, (int, float)):
                    score_val = float(eval_result)
                
                # breakdownを更新
                current_breakdown["llm_self_eval"] = score_val
                
                # 更新されたbreakdownを持つConfidenceScoreを作成
                llm_score = ConfidenceScore(
                    score=score_val,
                    factors=ConfidenceFactors(llm_self_confidence=score_val),
                    breakdown=current_breakdown.copy() # 全要素を含むbreakdown
                )
                step_scores.append(llm_score)
                logger.info(f"LLM self-evaluation: {score_val:.2f}")

            except Exception as e:
                logger.warning(f"LLM self-evaluation failed: {e}")

            # 2. クエリ網羅度評価 (Query Coverage)
            try:
                coverage_score = self.query_coverage_calculator.calculate(
                    query=state.plan.original_query,
                    answer=final_answer
                )
                
                # breakdownを更新
                current_breakdown["query_coverage"] = coverage_score
                
                coverage_obj = ConfidenceScore(
                    score=coverage_score,
                    factors=ConfidenceFactors(query_coverage=coverage_score),
                    breakdown=current_breakdown.copy() # 全要素を含むbreakdown
                )
                step_scores.append(coverage_obj)
                logger.info(f"Query coverage evaluation: {coverage_score:.2f}")
                
                # UIへの反映のために、最後の信頼度更新として通知する
                if self.on_confidence_update:
                    decision = ActionDecision(
                        level=InterventionLevel.SILENT,
                        confidence_score=coverage_score,
                        reason="Final coverage evaluation completed"
                    )
                    self.on_confidence_update(coverage_obj, decision)
                    
            except Exception as e:
                logger.warning(f"Query coverage evaluation failed: {e}")

        # ConfidenceAggregatorで統合
        if step_scores:
            aggregated_score = self.confidence_aggregator.aggregate(
                scores=step_scores,
                method="weighted"
            )
            logger.info(f"Aggregated confidence: {aggregated_score:.2f}")
            return aggregated_score

        # フォールバック: 単純平均
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

        # 最終回答を取得（最後のreasoningまたはlegacy_agentステップの出力）
        final_answer = None
        for step in reversed(state.plan.steps):
            if (step.action in ["reasoning", "run_legacy_agent"]) and step.step_id in state.step_results:
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

    # =========================================================================
    # Intervention Handler コールバック（Phase 3）
    # =========================================================================

    def _handle_intervention_notify(self, message: str) -> None:
        """通知レベルの介入処理（ログ出力のみ）"""
        logger.info(f"[NOTIFY] {message}")
        # UIへの通知（オプション）
        if self.on_intervention_required:
            self.on_intervention_required("notify", {"message": message})

    def _handle_intervention_confirm(
        self,
        request: InterventionRequest
    ) -> InterventionResponse:
        """確認レベルの介入処理"""
        logger.info(f"[CONFIRM] {request.message}")

        # on_intervention_requiredコールバックでUIに確認を要求
        if self.on_intervention_required:
            user_response = self.on_intervention_required("confirm", {
                "message": request.message,
                "reason": request.reason,
                "options": request.options,
                "confidence": request.confidence_score,
            })

            if user_response:
                # ユーザー応答を解析
                if user_response in ["はい、続行", "proceed", "yes"]:
                    return InterventionResponse(action=InterventionAction.PROCEED)
                elif user_response in ["計画を修正", "modify"]:
                    return InterventionResponse(action=InterventionAction.MODIFY)
                elif user_response in ["キャンセル", "cancel", "no"]:
                    return InterventionResponse(action=InterventionAction.CANCEL)
                else:
                    # ユーザー入力として扱う
                    return InterventionResponse(
                        action=InterventionAction.INPUT,
                        user_input=str(user_response)
                    )

        # コールバックがない場合はデフォルトで続行
        return InterventionResponse(action=InterventionAction.PROCEED)

    def _handle_intervention_escalate(
        self,
        request: InterventionRequest
    ) -> InterventionResponse:
        """エスカレーションレベルの介入処理"""
        logger.info(f"[ESCALATE] {request.message}")

        # on_intervention_requiredコールバックでUIにユーザー入力を要求
        if self.on_intervention_required:
            user_response = self.on_intervention_required("escalate", {
                "message": request.message,
                "question": request.question,
                "reason": request.reason,
                "confidence": request.confidence_score,
            })

            if user_response:
                return InterventionResponse(
                    action=InterventionAction.INPUT,
                    user_input=str(user_response)
                )

        # コールバックがない場合はタイムアウト扱い
        return InterventionResponse(
            action=InterventionAction.PROCEED,
            timeout_reached=True
        )

    def _handle_intervention_if_needed(
        self,
        action_decision: ActionDecision,
        step: PlanStep,
        state: ExecutionState
    ) -> Optional[InterventionResponse]:
        """
        必要に応じて介入を処理

        Args:
            action_decision: 信頼度に基づくアクション決定
            step: 現在のステップ
            state: 実行状態

        Returns:
            InterventionResponse: 介入レスポンス（介入が発生した場合）、またはNone
        """
        # SILENT/NOTIFYは自動続行
        if action_decision.level in [InterventionLevel.SILENT, InterventionLevel.NOTIFY]:
            if action_decision.level == InterventionLevel.NOTIFY:
                self.intervention_handler.handle(action_decision, step, state.plan)
            return None

        # CONFIRM/ESCALATEはユーザー介入が必要
        response = self.intervention_handler.handle(action_decision, step, state.plan)

        # キャンセルの場合は実行を中止
        if response.action == InterventionAction.CANCEL:
            state.is_cancelled = True

        return response


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