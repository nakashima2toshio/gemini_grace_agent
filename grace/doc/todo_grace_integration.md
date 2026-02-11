# TODO: GRACE Agent 改修計画

## 現状の発見事項

### `grace/` パッケージ（Phase 1-4 実装済み）

```
grace/
├── __init__.py      # 全コンポーネントの統合export
├── schemas.py       # ExecutionPlan, PlanStep, StepResult, ExecutionResult
├── config.py        # GraceConfig (Pydantic), ConfigLoader (YAML + env)
├── planner.py       # Planner.create_plan(query) → ExecutionPlan
├── executor.py      # Executor.execute_plan_generator(plan) → Generator[ExecutionState]
├── tools.py         # RAGSearchTool, ReasoningTool, AskUserTool, ToolRegistry
├── confidence.py    # Phase 2: ConfidenceCalculator, LLMSelfEvaluator
├── intervention.py  # Phase 3: InterventionHandler, ConfirmationFlow
└── replan.py        # Phase 4: ReplanManager, ReplanOrchestrator
```

### `grace_chat_page.py` の現状

**問題**: `ReActAgent` を直接使用しており、`grace/` パッケージを一切使っていない。

```python
# 現在 (grace_chat_page.py L18)
from services.agent_service import ReActAgent, get_available_collections_from_qdrant_helper

# 現在 (L231)
st.session_state.grace_agent = ReActAgent(
    selected_collections,
    selected_model,
    session_id=...,
    use_hybrid_search=...
)

# 現在 (L253) イベントループ
for event in st.session_state.grace_agent.execute_turn(prompt):
    # type: "log" / "tool_call" / "tool_result" / "final_answer"
```

---

## 改修(1): デフォルトモデルを `gemini-3-flash-preview` に変更

### 変更箇所（4ファイル・5箇所）

| # | ファイル | 行 | 変更内容 |
|---|---------|-----|---------|
| 1 | `config.py` | L417-424 | `GeminiConfig.AVAILABLE_MODELS` に `"gemini-3-flash-preview"` 追加 |
| 2 | `config.py` | L427 | `DEFAULT_MODEL` → `"gemini-3-flash-preview"` |
| 3 | `config.py` | L442-458 | `MODEL_PRICING` と `MODEL_LIMITS` に `gemini-3-flash-preview` エントリ追加 |
| 4 | `services/agent_service.py` | L148 | フォールバック `"gemini-2.0-flash"` → `"gemini-3-flash-preview"` |
| 5 | `grace/config.py` | L65 | `LLMConfig.model` デフォルト `"gemini-2.5-flash"` → `"gemini-3-flash-preview"` |

### 具体的な変更

**config.py (ルート)**
```python
# L417: AVAILABLE_MODELS に追加
AVAILABLE_MODELS: List[str] = [
    "gemini-3-flash-preview",       # ← 新デフォルト（先頭に配置）
    "gemini-2.5-flash",
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-preview",
    "gemini-2.5-pro-preview",
    "gemini-2.0-flash",
]

# L427: DEFAULT_MODEL 変更
DEFAULT_MODEL: str = "gemini-3-flash-preview"

# L442: MODEL_PRICING に追加
"gemini-3-flash-preview": {"input": 0.0001, "output": 0.0004},  # ← 要確認: 正式料金

# L452: MODEL_LIMITS に追加
"gemini-3-flash-preview": {"max_input_tokens": 1000000, "max_output_tokens": 64000},  # ← 要確認
```

**services/agent_service.py**
```python
# L148: フォールバックデフォルト変更
self.model_name = model_name or get_config("models.default", "gemini-3-flash-preview")
```

**grace/config.py**
```python
# L65: LLMConfig デフォルト変更
class LLMConfig(BaseModel):
    model: str = "gemini-3-flash-preview"
```

### リスク: ⚪低（文字列変更のみ）

### ⚠️ 確認事項
- `gemini-3-flash-preview` の正式料金・トークン制限
- モデル名が正しいか（Google AI Studio で確認）

---

## 改修(2): grace/ パッケージ接続（grace_chat_page.py 差し替え）

### 概要

`grace_chat_page.py` の import と初期化・イベントループを `Planner + Executor` に差し替え。
**grace/ は Phase 1-4 全て実装済み（4,591行）なので新規コードは不要。**

### アーキテクチャ変更

```
【現在】
grace_chat_page.py → ReActAgent.execute_turn() → イベントストリーム

【変更後】
grace_chat_page.py → Planner.create_plan()    → ExecutionPlan
                    → Executor.execute_plan_generator() → ExecutionState/イベントストリーム
```

### STEP 2-1: import 変更

```python
# 削除
from services.agent_service import ReActAgent, get_available_collections_from_qdrant_helper

# 追加
from grace import (
    Planner, create_planner,
    Executor, create_executor,
    ExecutionPlan, ExecutionState, StepResult,
    GraceConfig, get_config as get_grace_config,
)
from services.agent_service import get_available_collections_from_qdrant_helper  # これだけ残す
```

### STEP 2-2: 初期化ロジック変更

**現在 (L231付近)**
```python
st.session_state.grace_agent = ReActAgent(
    selected_collections,
    selected_model,
    session_id=st.session_state.grace_session_id,
    use_hybrid_search=use_hybrid_search
)
```

**変更後**
```python
# GraceConfig をオーバーライド（UI選択を反映）
grace_config = get_grace_config()
grace_config.llm.model = selected_model

# Planner + Executor を初期化
st.session_state.grace_planner = create_planner(
    config=grace_config,
    model_name=selected_model
)
st.session_state.grace_executor = create_executor(
    config=grace_config
)
```

**注意**: `selected_collections` と `use_hybrid_search` は `grace/tools.py` の `RAGSearchTool` が
`grace/config.py` の `QdrantConfig` 経由で取得する。UI選択を反映するには `grace_config.qdrant.search_priority` を更新するか、
ToolRegistry 生成時にパラメータを渡す必要がある。

→ **要確認**: `grace/tools.py` の `RAGSearchTool` がコレクション選択をどう扱っているか

### STEP 2-3: イベントループ変更

**現在 (L253付近)**
```python
for event in st.session_state.grace_agent.execute_turn(prompt):
    if event["type"] == "log": ...
    elif event["type"] == "tool_call": ...
    elif event["type"] == "tool_result": ...
    elif event["type"] == "final_answer": ...
```

**変更後**
```python
# === Phase 1: Plan ===
with st.expander("📋 計画策定 (Plan)", expanded=True):
    with st.spinner("計画を生成中..."):
        plan = st.session_state.grace_planner.create_plan(prompt)

    # 計画の表示
    st.markdown(f"**目標**: {plan.original_query}")
    st.markdown(f"**複雑度**: {plan.complexity:.1f} | **ステップ数**: {plan.estimated_steps}")
    for step in plan.steps:
        st.markdown(f"  {step.step_id}. [{step.action}] {step.description}")

# === Phase 2-4: Execute (with Confidence/Intervention/Replan) ===
with st.expander("⚡ 実行 (Execute)", expanded=True):
    executor = st.session_state.grace_executor
    gen = executor.execute_plan_generator(plan)

    execution_result = None
    try:
        while True:
            yielded = next(gen)

            if isinstance(yielded, ExecutionState):
                # ステップ進捗の表示
                state = yielded
                current = state.current_step_id
                step_status = state.step_statuses.get(current)
                st.markdown(f"Step {current}: {step_status}")

                # 介入リクエストがある場合
                if state.is_paused and state.intervention_request:
                    st.warning(f"⚠️ 確認が必要: {state.intervention_request.message}")
                    # TODO: Phase 3 HITL対応

            elif isinstance(yielded, dict):
                # ツール実行結果などのイベント
                event_type = yielded.get("type")
                if event_type == "log":
                    st.markdown(yielded["content"])
                elif event_type == "tool_call":
                    st.markdown(f"🛠️ **Tool Call:** `{yielded['name']}`")
                elif event_type == "tool_result":
                    st.markdown(f"📝 **Tool Result:** {yielded['content'][:200]}...")

    except StopIteration as e:
        execution_result = e.value  # ExecutionResult

# === 最終回答の表示 ===
if execution_result and execution_result.final_answer:
    final_response_content = execution_result.final_answer
    st.markdown(final_response_content)
    st.session_state.grace_chat_history.append(
        {"role": "assistant", "content": final_response_content}
    )
else:
    st.warning("エージェントからの応答がありませんでした。")
```

### STEP 2-4: session_state キー変更

| 旧キー | 新キー | 説明 |
|--------|--------|------|
| `grace_agent` | `grace_planner` | Planner インスタンス |
| （なし） | `grace_executor` | Executor インスタンス |

**クリアボタン**も `grace_planner` / `grace_executor` に変更。

### イベントマッピング

`Executor.execute_plan_generator()` は2種類のオブジェクトを yield する:

| yield 内容 | 型 | 用途 |
|-----------|-----|------|
| `ExecutionState` | dataclass | ステップ完了/一時停止の通知 |
| `dict` (type: "log") | dict | `_execute_step()` 内のツール結果表示 |

`_execute_legacy_agent_step()` が呼ばれた場合は、旧来の `{"type": "log"/"tool_call"/"tool_result"/"final_answer"}` も流れる。

### UI表示設計

```
┌─ 📋 計画策定 (Plan) ────────────────────────┐
│  目標: ユーザーの質問                         │
│  複雑度: 0.4 | ステップ数: 2                  │
│  1. [rag_search] RAGで関連情報を検索          │
│  2. [reasoning] 検索結果を統合して回答生成     │
└─────────────────────────────────────────────┘
┌─ ⚡ 実行 (Execute) ─────────────────────────┐
│  Step 1: rag_search → ✅ success (0.82)      │
│  📝 ツール実行結果: [検索結果の概要]          │
│  Step 2: reasoning → ✅ success (0.90)        │
│  ─── [RePlan v2 発生時] ───                   │
│  ❌ 計画修正: 検索結果不足のため再検索         │
│  Step 3: [新ステップ] → ✅ success            │
└─────────────────────────────────────────────┘
┌─ 最終回答 ──────────────────────────────────┐
│  〜の構成者は○○です。社内ナレッジによると...   │
│  信頼度: 0.85 | リプラン回数: 1               │
└─────────────────────────────────────────────┘
```

### リスク評価

| 項目 | リスク | 理由 |
|------|--------|------|
| import変更 | ⚪低 | 既存パッケージのimport先変更のみ |
| 初期化変更 | 🟡中 | GraceConfig ↔ UI設定の同期 |
| イベントループ | 🟡中 | Generator protocol (StopIteration.value) の扱い |
| コレクション選択連携 | 🟡中 | grace/tools.py のRAGSearchToolとの接続確認が必要 |

---

## 実施順序

```
改修(1) モデル変更 ← 独立、即実施可能、リスク低
  ↓
改修(2) STEP 2-1: import 変更
  ↓
改修(2) STEP 2-2: 初期化ロジック変更
  ↓
改修(2) STEP 2-3: イベントループ変更
  ↓
改修(2) STEP 2-4: session_state / クリアボタン修正
  ↓
動作テスト
```

## 確認事項（実装前に要回答）

1. **`gemini-3-flash-preview` の正式料金・トークン制限** — Google AI Studioで確認
2. **`grace/tools.py` の `RAGSearchTool`** — UI選択のコレクション・ハイブリッド検索をどう渡すか
3. **`agent_chat_page.py` は変更しない** — ReAct版維持で2画面の差異を明確化（確認）
