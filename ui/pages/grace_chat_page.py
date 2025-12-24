import streamlit as st
import time
import sys
import os
import logging
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.append(os.getcwd())

from grace.config import get_config
from grace.planner import create_planner
from grace.executor import create_executor, ExecutionResult, ExecutionState
from grace.schemas import ExecutionPlan, PlanStep
from grace.confidence import ConfidenceScore, ActionDecision, InterventionLevel
from grace.intervention import InterventionRequest, InterventionAction
from ui.components.grace_components import display_confidence_metric, display_execution_plan, display_intervention_request
from ui.components.rag_components import select_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================

def init_session_state():
    """Session Stateの初期化"""
    defaults = {
        "messages": [],
        "current_logs": [],
        "confidence_history": [],
        "execution_state": None,  # ExecutionStateオブジェクト
        "latest_confidence": None,
        "latest_decision": None,
        "event_history": [], # 現在のターンのイベント履歴
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

class StreamlitCallbackHandler:
    """コールバックハンドラー（イベント発行強化版）"""
    
    def _add_event(self, event: Dict[str, Any]):
        st.session_state.event_history.append(event)
        # 簡易ログにも追加（サイドバー用）
        msg = f"[{event['type']}] {str(event.get('content', ''))[:50]}..."
        st.session_state.current_logs.append(msg)

    def on_step_start(self, step: PlanStep):
        self._add_event({
            "type": "step_start",
            "content": f"Step {step.step_id}: {step.action}",
            "name": step.action
        })
        st.toast(f"🏃 Step {step.step_id}: {step.action}")

    def on_step_complete(self, result: Any):
        self._add_event({
            "type": "step_complete",
            "content": f"Step {result.step_id} Completed",
            "confidence": result.confidence
        })
        
        st.session_state.confidence_history.append({
            "step": result.step_id,
            "score": result.confidence
        })

    def on_confidence_update(self, score: ConfidenceScore, decision: ActionDecision):
        st.session_state.latest_confidence = score
        st.session_state.latest_decision = decision
        
        self._add_event({
            "type": "confidence_update",
            "score": score.score,
            "level": score.level,
            "breakdown": score.breakdown
        })
        
    def on_intervention_required(self, type: str, data: Dict[str, Any]):
        self._add_event({
            "type": "intervention_required",
            "content": data.get("message", "Intervention required"),
            "reason": data.get("reason", "")
        })

    def on_replan(self, reason: str, plan: Any):
        self._add_event({
            "type": "replan",
            "content": f"Steps: {len(plan.steps)}",
            "reason": reason
        })

def get_executor():
    """Executorのインスタンスを作成（設定は毎回読み込み）"""
    config = get_config()
    handler = StreamlitCallbackHandler()
    return create_executor(
        config=config,
        on_step_start=handler.on_step_start,
        on_step_complete=handler.on_step_complete,
        on_confidence_update=handler.on_confidence_update,
        on_intervention_required=handler.on_intervention_required,
        on_replan=handler.on_replan # Replanコールバックを追加
    )

def handle_intervention_response(response_type: str, value: Optional[str] = None):
    """介入レスポンスの処理"""
    state = st.session_state.execution_state
    if not state or not state.is_paused:
        return

    # 状態を更新
    state.is_paused = False
    
    # ユーザー入力を前のステップの結果に追加（簡易的なコンテキスト注入）
    if value and state.current_step_id in state.step_results:
        prev_output = state.step_results[state.current_step_id].output
        state.step_results[state.current_step_id].output = f"{prev_output}\n\n【ユーザーからの追加情報】\n{value}"
        st.session_state.current_logs.append(f"📝 User Input: {value}")

    # リクエストをクリア
    state.intervention_request = None
    
    # 再実行
    st.rerun()

def render_event(event: Dict[str, Any]):
    """イベントの描画ヘルパー（プロセス可視化強化版）"""
    event_type = event.get("type")
    content = event.get("content")
    name = event.get("name", "")
    
    # --- 1. Plan-and-Execute ---
    if event_type == "plan_created":
        st.success(f"📋 計画生成完了: {content}", icon="📋")
    
    elif event_type == "step_start":
        st.info(f"🏃 {content}", icon="🏃")
        
    elif event_type == "step_complete":
        confidence = event.get("confidence", 0.0)
        st.success(f"✅ {content} (信頼度: {confidence:.2f})", icon="✅")

    # --- 2. ReAct (Thought & Action) ---
    elif event_type == "log":
        # Thoughtなどのログ
        if "Thought:" in content or "考え:" in content:
            st.info(content, icon="🧠")
        elif "【ツール実行結果" in content:
            # ツール実行結果（検索結果など）をコードブロックで表示
            parts = content.split("\n", 1)
            header = parts[0]
            body = parts[1] if len(parts) > 1 else ""
            st.markdown(f"**{header}**")
            if body:
                st.code(body, language="json")
        else:
            st.text(content) # 一般ログ

    elif event_type == "tool_call":
        # ツール呼び出し
        with st.expander(f"🛠️ ツール実行: {name}", expanded=False):
            st.json(event.get("args"))
            
    elif event_type == "tool_result":
        # ツール結果
        with st.expander(f"📝 ツール結果: {name}", expanded=False):
            st.markdown(content)

    # --- 3. Confidence-aware ---
    elif event_type == "confidence_update":
        score = event.get("score", 0.0)
        level = event.get("level", "unknown")
        breakdown = event.get("breakdown", {})
        
        with st.expander(f"📊 信頼度評価: {score:.2f} ({level})", expanded=False):
            st.json(breakdown)

    # --- 4. HITL (Human-In-The-Loop) ---
    elif event_type == "intervention_required":
        reason = event.get("reason", "")
        st.warning(f"🛑 介入要求: {content}\n理由: {reason}", icon="🛑")
        
    elif event_type == "user_response":
        st.info(f"👤 ユーザー応答: {content}", icon="👤")

    # --- 5. Adaptive Replanning ---
    elif event_type == "replan":
        reason = event.get("reason", "")
        st.warning(f"🔄 再計画 (Replanning): {reason}", icon="🔄")
        with st.expander("新しい計画", expanded=True):
            st.text(content)

    # --- 6. Reflection ---
    elif event_type == "reflection":
        st.info(f"🪞 自己省察 (Reflection): {content}", icon="🪞")

    # Default fallback
    else:
        st.text(f"[{event_type}] {content}")

def process_execution():
    """エージェント実行ループ（再開・継続）"""
    state = st.session_state.execution_state
    if not state:
        return

    executor = get_executor()
    plan_placeholder = st.empty()
    
    # 計画の初期表示
    with plan_placeholder.container():
        display_execution_plan(state.plan, current_step_id=state.current_step_id)

    try:
        # ジェネレータ作成（既存の状態から再開）
        generator = executor.execute_plan_generator(state.plan, state=state)
        
        # ログ表示用のコンテナ（Expander）
        # ストリーミング中は開いておく
        log_expander = st.expander("📝 思考プロセス (Thought Process)", expanded=True)
        
        # 過去のイベントがあれば再描画（再開時など）
        with log_expander:
            for event in st.session_state.event_history:
                render_event(event)

        result = None
        while True:
            try:
                yielded_item = next(generator)
                
                if isinstance(yielded_item, dict):
                    # イベントログの保存と表示
                    st.session_state.event_history.append(yielded_item)
                    with log_expander:
                        render_event(yielded_item)

                elif isinstance(yielded_item, ExecutionState):
                    # 実行状態の更新
                    new_state = yielded_item
                    st.session_state.execution_state = new_state
                    
                    # 計画表示更新
                    with plan_placeholder.container():
                        display_execution_plan(new_state.plan, current_step_id=new_state.current_step_id)
                    
                    # 一時停止チェック
                    if new_state.is_paused:
                        st.rerun() # UI更新のためにリラン
                        return

            except StopIteration as e:
                result = e.value
                break
        
        # 完了処理
        if result:
            final_answer = result.final_answer
            confidence = result.overall_confidence
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "confidence": confidence,
                "plan": state.plan, # 計画も保存
                "logs": list(st.session_state.event_history) # ログをメッセージに紐付けて保存
            })
            
            # 実行完了として状態をクリア
            st.session_state.execution_state = None 
            # st.session_state.event_history = [] # ★履歴をクリアしない
            st.rerun()

    except Exception as e:
        st.error(f"Execution Error: {e}")
        logger.error(f"Execution failed: {e}", exc_info=True)
        st.session_state.execution_state = None

# =============================================================================
# UI Components
# =============================================================================

def render_confidence_sidebar():
    """サイドバー描画"""
    st.sidebar.title("🤖 GRACE Status")
    
    # モデル選択（共通コンポーネント）
    select_model()
    
    if "latest_confidence" in st.session_state and st.session_state.latest_confidence:
        score_obj = st.session_state.latest_confidence
        display_confidence_metric(score_obj.score, score_obj.level, score_obj.breakdown)

    if "latest_decision" in st.session_state and st.session_state.latest_decision:
        decision = st.session_state.latest_decision
        st.sidebar.info(f"Action: **{decision.level.value}**\n\n{decision.reason}")

    st.sidebar.subheader("📜 Execution Log")
    if "current_logs" in st.session_state:
        for log in st.session_state.current_logs:
            st.sidebar.text(log)

def render_chat_area():
    """チャットエリア描画"""
    st.title("🤖 GRACE Agent Chat")
    
    # 履歴表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # 計画があれば表示
                if "plan" in message:
                    with st.expander("実行計画を確認", expanded=False):
                        display_execution_plan(message["plan"])
                
                st.markdown(message["content"])
                
                # ログがあればExpanderで表示（デフォルトは閉じる）
                if "logs" in message and message["logs"]:
                    with st.expander("思考プロセスを表示", expanded=False):
                        for event in message["logs"]:
                            render_event(event)

                if "confidence" in message:
                    st.caption(f"Confidence: {message['confidence']:.2f}")
            else:
                st.markdown(message["content"])

    # 現在実行中の状態があれば表示
    state = st.session_state.get("execution_state")
    if state:
        with st.chat_message("assistant"):
            # 進行中のみ "Processing" を表示
            if not state.is_paused:
                st.info("🔄 Processing...")
            
            # 計画表示
            display_execution_plan(state.plan, current_step_id=state.current_step_id)
            
            # 途中経過のログを表示（リラン後も見えるように）
            if "event_history" in st.session_state and st.session_state.event_history:
                # 介入中はログを確認したい場合が多いのでデフォルトで開く設定にするなどの調整も可能
                # ここではexpanded=Falseにしておき、ユーザーが必要に応じて開けるようにする
                with st.expander("📝 思考プロセス (Thought Process)", expanded=False):
                    for event in st.session_state.event_history:
                        render_event(event)
            
            # 介入リクエストがあれば表示
            if state.is_paused and state.intervention_request:
                req = state.intervention_request
                # InterventionRequestオブジェクトから辞書へ変換（コンポーネント用）
                req_dict = {
                    "type": "confirm" if req.level == "confirm" else "escalate",
                    "data": {
                        "message": req.message
                    }
                }
                
                # コールバック関数
                def on_response(val):
                    handle_intervention_response(req_dict["type"], val)

                display_intervention_request(req_dict, on_response)

def handle_user_input():
    """新規ユーザー入力処理"""
    # 実行中は入力無効
    if st.session_state.get("execution_state"):
        return

    if prompt := st.chat_input("質問を入力してください..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 初期化
        st.session_state.current_logs = []
        st.session_state.confidence_history = []
        st.session_state.latest_confidence = None
        st.session_state.latest_decision = None
        st.session_state.event_history = [] # イベント履歴初期化
        
        # 計画作成 & 状態初期化
        try:
            config = get_config()
            planner = create_planner(config=config)
            plan = planner.create_plan(prompt)
            
            # 状態を作成して保存
            st.session_state.execution_state = ExecutionState(plan=plan)
            st.rerun() # process_executionを実行するためにリラン
            
        except Exception as e:
            st.error(f"Planning Error: {e}")

def show_grace_chat_page():
    init_session_state()
    render_confidence_sidebar()
    render_chat_area()
    
    # 実行状態があれば処理を進める
    state = st.session_state.get("execution_state")
    if state and not state.is_paused:
        process_execution()
        
    handle_user_input()

if __name__ == "__main__":
    st.set_page_config(page_title="GRACE Agent", page_icon="🤖", layout="wide")
    show_grace_chat_page()
