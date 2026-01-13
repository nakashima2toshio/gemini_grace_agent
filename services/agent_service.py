# agent_services.py
import os
import uuid
import google.generativeai as genai
from google.generativeai import ChatSession, GenerativeModel
from typing import Dict, List, Any, Optional, Union, Tuple, Generator
import logging
from qdrant_client import QdrantClient  # Added QdrantClient import

# Configuration and Tools
from config import AgentConfig, GeminiConfig
from agent_tools import (
    search_rag_knowledge_base,
    list_rag_collections,
    RAGToolError,
    search_rag_knowledge_base_cached  # 新戦略をインポート
)
from services.qdrant_service import get_all_collections
from services.log_service import log_unanswered_question
from regex_mecab import KeywordExtractor

# キャッシュと並列検索をインポート
from agent_cache import collection_cache
from agent_parallel_search import parallel_search_engine

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants & Configuration (Moved from agent_chat_page.py)
# -----------------------------------------------------------------------------

SYSTEM_INSTRUCTION_TEMPLATE = """
あなたは、社内ドキュメント検索システムと連携した「ハイブリッド・ナレッジ・エージェント」です。
あなたの役割は、ユーザーの質問に対して、一般的な知識と、提供されたツール（社内ナレッジ検索）を適切に使い分けて回答することです。

## ReAct プロセスと出力フォーマット (厳守)

あなたは **Thought (思考)**、**Action (ツール実行)**、**Observation (結果観察)** のサイクルを回して回答に到達する必要があります。

### 1. ツールを使用する場合（検索が必要な場合）
必ず以下の形式で思考を出力してから、ツールを呼び出してください。
**Thought: [なぜ検索が必要か、どんなクエリで検索するか]**
(この後にツール呼び出しが行われます)
**重要: 
- 検索クエリを作成する際は、提供された「重要キーワード」を必ず含めてください。
- `collection_name` パラメータは絶対に指定しないでください。システムが自動的に全コレクションから最適なものを選択します。**

### 2. 最終回答を行う場合（検索が完了した、または検索不要な場合）
必ず以下の形式で出力してください。
**Thought: [得られた情報に基づいてどう回答するか、または検索結果がなかった場合の判断]**
**Answer: [ユーザーへの最終的な回答]**

**重要:**
- 検索クエリは、質問文から「いつ」「誰」「何」などの具体的な要素を抽出して作成してください。抽象的な質問（例：「教えて」）をそのまま検索クエリにせず、具体的なキーワードに変換してください。
- 検索結果のスコアが低くても（例: 0.5程度）、内容が質問に関連していれば、その情報を積極的に使用して回答を作成してください。「情報が見つかりませんでした」と即断せず、得られた断片的な情報からでも回答を試みてください。
- 回答は必ず `Answer:` (または `**Answer:**`) で始めてください。

---

## 行動指針 (Router Guidelines)

1.  **専門知識の検索**:
    *   以下のいずれかに該当する場合は、**必ず `search_rag_knowledge_base` ツールを使用してください。**
        *   プロジェクト固有の仕様、設定、エラー、社内規定、Wikipediaの知識に関する質問。
        *   特定の情報源（例: "Wikipediaによると"、"ライブドアニュースで"）が指定されている質問。
        *   **内容が不明瞭であっても、社内ナレッジに関連する可能性があると判断される質問（例：特定のコード名、システム名、ランダムに見える文字列など）。**
        *   **ただし、一般的なプログラミング言語の文法や使い方に関する質問にはツールを使用しないでください。**
    *   **現在利用可能なコレクションは以下の通りです:**
        {available_collections}

2.  **スマート検索システム（自動コレクション選択）**:
    *   **重要: `search_rag_knowledge_base` ツールを呼び出す際、`collection_name` パラメータは絶対に指定しないでください。**
    *   システムが自動的に以下の戦略で最適なコレクションを選択します：
        *   **キャッシュ優先**: 前回成功したコレクションを優先的に検索
        *   **並列検索**: キャッシュミス時は全コレクションを同時並列検索
        *   **スコアベース選択**: 最もスコアが高い結果を自動的に返す
    *   あなたは `query` パラメータのみを指定してください。例: `search_rag_knowledge_base(query="カリン・フォン・アロルディンゲン")`
    *   
    *   **参考: 利用可能なコレクション（自動選択されます）**
        *   `cc_news`: 英語のニュース記事
        *   `wikipedia_ja`: 日本語の百科事典
        *   `livedoor`: 日本語のニュース・ブログ
        *   `japanese_text`: 日本語のWebテキスト
        *   `qa_pairs_custom_upload`, `custom_upload`: ユーザーアップロードの専門Q&A

3.  **一般的な会話**:
    *   挨拶、雑談、単純な計算など、専門知識が不要な場合は、ツールを使わずに `Answer:` で直接回答してください。

4.  **正直さと不足情報の処理 (Critical)**:
    *   ツール検索の結果、情報が得られなかった場合は、**絶対に**あなたの事前学習知識で捏造してはいけません。
    *   「提供された社内ナレッジには関連情報がありませんでした」と正直に伝えてください。

5.  **回答のスタイル**:
    *   丁寧な日本語（です・ます調）で回答してください。
    *   検索結果に基づく回答の場合、「社内ナレッジによると...」や「ソース [ファイル名] によると...」と出典を明示してください。
"""

REFLECTION_INSTRUCTION = """
## Reflection (自己評価と修正)

あなたは上記で作成した「回答案」を、以下の基準で客観的に評価し、必要であれば修正してください。

**チェックリスト:**
1.  **正確性:** 検索結果(もしあれば)に基づいているか？ 提供された情報源に含まれない情報を捏造していないか？
2.  **回答の適切性:** ユーザーの質問に直接的かつ明確に答えているか？
3.  **スタイル:** 親しみやすく、丁寧な日本語（です・ます調）か？ 箇条書きなどを活用して読みやすいか？

**指示:**
*   修正が不要な場合でも、必ず **Final Answer** を出力してください。
*   修正が必要な場合は、修正後の回答を **Final Answer** として出力してください。
*   思考プロセスは `Thought:` で始めてください。

**出力フォーマット:**
Thought: [評価と修正の思考プロセス]
Final Answer: [最終的な回答]
"""

TOOLS_MAP: Dict[str, Any] = {
    'search_rag_knowledge_base': search_rag_knowledge_base,
    'list_rag_collections'     : list_rag_collections
}


# -----------------------------------------------------------------------------
# ReActAgent Class
# -----------------------------------------------------------------------------

class ReActAgent:
    def __init__(self, selected_collections: List[str], model_name: str, session_id: Optional[str] = None):
        self.selected_collections = selected_collections
        self.model_name = model_name
        self.session_id = session_id or str(uuid.uuid4())  # セッションIDを生成または使用
        self.chat_session = self._setup_session()
        self.thought_log: List[str] = []  # Initialize thought_log here
        # キーワード抽出器の初期化
        try:
            self.keyword_extractor = KeywordExtractor(prefer_mecab=True)
            logger.info(f"KeywordExtractor initialized successfully. Session: {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to initialize KeywordExtractor: {e}")
            self.keyword_extractor = None

        logger.info(f"ReActAgent initialized with session_id: {self.session_id}")

    def _setup_session(self) -> ChatSession:
        """Geminiエージェントのセットアップ"""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API Key missing: GEMINI_API_KEY or GOOGLE_API_KEY not set.")

        genai.configure(api_key=api_key)

        tools_list = [search_rag_knowledge_base, list_rag_collections]

        collections_str = ", ".join(self.selected_collections) if self.selected_collections else "(コレクションが見つかりません)"
        system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(available_collections=collections_str)

        model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=tools_list,
            system_instruction=system_instruction
        )

        chat = model.start_chat(enable_automatic_function_calling=False)
        return chat

    def execute_turn(self, user_input: str) -> Generator[Dict[str, Any], None, None]:
        """
        ReAct → Reflection の順にエージェントのターンを実行し、
        進捗状況をイベントとしてyieldするジェネレータ。
        """
        self.thought_log = []  # Clear log for new turn

        # --- Phase 1: ReAct Loop ---
        yield {"type": "log", "content": "🤖 **ReAct Phase Start**"}
        draft_answer: Optional[str] = None
        for event in self._execute_react_loop(user_input):
            yield event
            if event["type"] == "final_text":  # This event carries the draft answer from ReAct
                draft_answer = event["content"]

        # --- Phase 2: Reflection ---
        if draft_answer:
            yield {"type": "log", "content": "🔄 **Reflection Phase (推敲)**"}
            final_answer_after_reflection = yield from self._execute_reflection_phase(draft_answer)
            draft_answer = final_answer_after_reflection  # Update draft with reflected answer

        yield {"type": "final_answer", "content": self._format_final_answer(draft_answer)}

    def _execute_react_loop(self, user_input: str) -> Generator[Dict[str, Any], None, None]:
        """
        ReActループを実行し、各ステップのイベントをyieldする。
        最終的なドラフト回答を 'final_text' イベントとしてyieldする。
        """
        # 重要単語抽出とプロンプト拡張
        augmented_input = user_input
        if self.keyword_extractor:
            try:
                keywords = self.keyword_extractor.extract(user_input, top_n=5)
                if keywords:
                    keywords_str = ", ".join(keywords)
                    augmented_input = f"""{user_input}\n\n【重要: 検索クエリ作成の指示】\n以下の抽出された重要キーワードを、必ず検索クエリに含めてください。\n重要キーワード: {keywords_str}"""
                    logger.info(f"Augmented input with keywords: {keywords_str}")
                    yield {"type": "log", "content": f"🔑 **Extracted Keywords:** {keywords_str}"}
            except Exception as e:
                logger.warning(f"Keyword extraction failed during turn: {e}")

        current_response_obj = self.chat_session.send_message(augmented_input)
        max_turns = 10
        turn_count = 0
        final_text_from_react = ""

        while turn_count < max_turns:
            turn_count += 1
            function_call_found = False
            current_turn_text_from_model = ""

            for part in current_response_obj.parts:
                if part.text:
                    text = part.text.strip()
                    if "Thought:" in text or "考え:" in text:
                        self.thought_log.append(f"🧠 **Thought:**\n{text}")
                        yield {"type": "log", "content": f"🧠 **Thought:**\n{text}"}
                        current_turn_text_from_model = text
                    else:
                        current_turn_text_from_model = text

                if part.function_call:
                    function_call_found = True
                    fn = part.function_call
                    tool_name = fn.name
                    tool_args = dict(fn.args)

                    logger.info(f"Agent Tool Call: {tool_name}({tool_args})")
                    self.thought_log.append(f"🛠️ **Tool Call:** `{tool_name}`\nArgs: `{tool_args}`")
                    yield {"type": "tool_call", "name": tool_name, "args": tool_args}

                    tool_result = ""
                    try:
                        if tool_name in TOOLS_MAP:
                            # search_rag_knowledge_base の場合はキャッシュ版を使用
                            if tool_name == 'search_rag_knowledge_base':
                                tool_result = search_rag_knowledge_base_cached(
                                    query=tool_args.get('query', ''),
                                    session_id=self.session_id,
                                    collection_name=tool_args.get('collection_name')
                                )
                            else:
                                tool_result = TOOLS_MAP[tool_name](**tool_args)
                        else:
                            tool_result = f"Error: Tool '{tool_name}' not found."
                    except RAGToolError as e:
                        tool_result = f"エラーが発生しました: {str(e)}"
                        logger.error(f"RAG Tool Error during '{tool_name}': {e}")
                    except Exception as e:
                        tool_result = f"予期せぬエラー: {str(e)}"
                        logger.error(f"Unexpected error during tool '{tool_name}': {e}", exc_info=True)

                    log_tool_result = str(tool_result)[:500] + "..." if len(str(tool_result)) > 500 else str(
                        tool_result)
                    self.thought_log.append(f"📝 **Tool Result:**\n{log_tool_result}")
                    yield {"type": "tool_result", "content": log_tool_result}
                    logger.info(f"Tool Result: {log_tool_result}")

                    if isinstance(tool_result, str) and tool_result.startswith("[[NO_RAG_RESULT"):
                        reason = "NO_RESULT"
                        if "LOW_SCORE" in tool_result:
                            reason = "LOW_SCORE"
                        collection_arg = tool_args.get('collection_name', 'unknown')
                        log_unanswered_question(
                            query=user_input,
                            collections=[collection_arg],
                            reason=reason,
                            agent_response="(Search Failed)"
                        )

                    current_response_obj = self.chat_session.send_message(
                        [genai.protos.Part(
                            function_response={
                                "name"    : tool_name,
                                "response": {'result': tool_result}
                            }
                        )]
                    )
                    break

            if not function_call_found:
                # ツール呼び出しがない場合、現在のテキストを回答案とする
                # 明示的な "Answer:" タグがなくても、最後のテキスト出力を採用する (Fallback)
                final_text_from_react = current_turn_text_from_model
                break

        yield {"type": "final_text", "content": final_text_from_react}  # Yield the draft answer from ReAct

    def _execute_reflection_phase(self, draft_answer: str) -> Generator[Dict[str, Any], None, str]:
        """
        Reflectionフェーズを実行し、修正後の回答を返す。
        進捗状況をイベントとしてyieldするジェネレータ。
        """
        final_response_text = draft_answer
        try:
            reflection_msg = f"{REFLECTION_INSTRUCTION}\n\n**あなたの回答案:**\n{draft_answer}"

            # Reflectionフェーズではツールを使用させないため、tools設定を空にするか
            # テキスト生成のみを強制する設定が必要だが、ChatSessionの途中での設定変更は複雑。
            # そのため、応答からFunctionCallが含まれていないかチェックし、
            # 含まれていた場合はそのFunctionCallを無視してテキスト部分を探すか、ドラフトを返す。

            reflection_response = self.chat_session.send_message(reflection_msg)

            reflection_text = ""

            # partsを確認してテキストを抽出
            if reflection_response.parts:
                for part in reflection_response.parts:
                    if part.text:
                        reflection_text += part.text
                    elif part.function_call:
                        logger.warning("Reflection phase generated a function call, ignoring.")

            if not reflection_text:
                # テキストが生成されなかった場合（FunctionCallのみなど）
                logger.warning("Reflection phase did not generate text.")
                return draft_answer

            reflection_thought = ""
            reflection_answer = ""

            if "Final Answer:" in reflection_text:
                parts = reflection_text.split("Final Answer:", 1)
                reflection_thought = parts[0].strip()
                reflection_answer = parts[1].strip()
            else:
                reflection_thought = "Format mismatch in reflection."
                reflection_answer = reflection_text

            if reflection_thought:
                clean_thought = reflection_thought.replace("Thought:", "").strip()
                self.thought_log.append(f"🤔 **Reflection Thought:**\n{clean_thought}")
                logger.info(f"Reflection Thought: {clean_thought}")
                yield {"type"   : "log",
                       "content": f"🤔 **Reflection Thought:**\n{clean_thought}"}  # Yield reflection thought

            if reflection_answer:
                final_response_text = reflection_answer
                logger.info(f"Reflection Answer: {reflection_answer}")

        except Exception as e:
            logger.error(f"Error during reflection phase: {e}")
            self.thought_log.append(f"⚠️ **Reflection Error:** {str(e)}")
            yield {"type": "log", "content": f"⚠️ **Reflection Error:** {str(e)}"}  # Yield reflection error
            final_response_text = draft_answer

        return final_response_text

    def _format_final_answer(self, raw_answer: str) -> str:
        """
        最終回答の整形を行うヘルパーメソッド。
        """
        if "Answer:" in raw_answer:
            parts = raw_answer.split("Answer:", 1)
            return parts[1].strip()
        elif raw_answer.startswith("Thought:"):
            return raw_answer.replace("Thought:", "").strip()
        elif raw_answer.startswith("考え:"):
            return raw_answer.replace("考え:", "").strip()
        return raw_answer


# Helper function (Moved from agent_chat_page.py)
def get_available_collections_from_qdrant_helper() -> List[str]:
    """Qdrantから利用可能なコレクション名を取得 (helper for now, will integrate into Agent if needed)"""
    try:
        client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        collections = client.get_collections()
        return [c.name for c in collections.collections]
    except Exception as e:
        logger.error(f"Failed to fetch collections: {e}")
        return []
