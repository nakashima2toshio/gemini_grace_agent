# python agent_main.py

import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content # Assuming this is needed for content.Part
from google.generativeai import ChatSession, GenerativeModel # Added for type hinting
from dotenv import load_dotenv
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple # Added Union, Tuple
from config import AgentConfig, PathConfig
from agent_tools import search_rag_knowledge_base, list_rag_collections, RAGToolError

# Define SYSTEM_INSTRUCTION here or move to config.py for better type hinting if it contains f-strings
SYSTEM_INSTRUCTION: str = f"""
あなたは、社内ドキュメント検索システムと連携した「ハイブリッド・ナレッジ・エージェント」です。
あなたの役割は、ユーザーの質問に対して、一般的な知識と、提供されたツール（社内ナレッジ検索）を適切に使い分けて回答することです。

## 思考プロセス (Chain of Thought) の可視化

回答やツール使用の前に、必ずあなたの思考プロセスを出力してください。
**特に、なぜその行動（検索する、あるいは検索しない）を選んだのか、その理由を簡潔に述べてください。**
形式: `Thought: ここに思考を記述...`

## 行動指針 (Router Guidelines)

1.  **専門知識の検索**:
    *   以下のいずれかに該当する場合は、**必ず `search_rag_knowledge_base` ツールを使用してください。**
        *   プロジェクト固有の仕様、設定、エラー、社内規定、Wikipediaの知識に関する質問。
        *   特定の情報源（例: "Wikipediaによると"、"ライブドアニュースで"）が指定されている質問。
        *   **内容が不明瞭であっても、社内ナレッジに関連する可能性があると判断される質問（例：特定のコード名、システム名、ランダムに見える文字列など）。**
        *   **ただし、一般的なプログラミング言語の文法や使い方に関する質問にはツールを使用しないでください。**
    *   **ツールの利用時には、必要に応じて `collection_name` 引数に、検索対象のQdrantコレクション名を指定してください。**
    *   **現在利用可能なコレクションは以下の通りです:**
        {", ".join(AgentConfig.RAG_AVAILABLE_COLLECTIONS)}
    *   あなたの事前学習知識だけで回答せず、必ずツールからの情報を優先してください。

2.  **コレクション選択のヒント**:
    *   「Wikipedia」に関する質問であれば、`qa_a02_qa_pairs_wikipedia_ja` コレクションを使用してください。
    *   「ライブドアニュース」に関する質問であれば、`qa_a02_qa_pairs_livedoor` コレクションを使用してください。
    *   その他の一般的な質問や、コレクションが特定できない場合は、デフォルトのコレクションを使用してください。

3.  **一般的な会話**:
    *   挨拶、雑談、単純な計算など、**上記「専門知識の検索」に該当しない場合は、ツールを使わずに直接回答してください。**
    *   **一般的なプログラミングの文法や使い方に関する質問は、あなたの事前学習知識で回答してください（ツール使用禁止）。**

4.  **正直さと不足情報の処理 (Critical)**:
    *   ツールを使用し、その結果（Observation）が「検索結果が見つかりませんでした」または関連情報を含まない場合、**どのような状況であっても、絶対に**あなたの事前学習知識で捏造してはいけません。
    *   **たとえ、あなたが一般常識で答えられる内容であっても、ツール検索で結果がなかった場合は、このルールを最優先してください。**
    *   「申し訳ありませんが、提供された情報源の中には、その質問に対する回答が見つかりませんでした。」と正直に伝えてください。
    *   その上で、「もしよろしければ、もう少し詳しいキーワードや別の表現で質問していただけますか？」とユーザーを誘導してください。

5.  **回答のスタイル**:
    *   丁寧な日本語（です・ます調）で回答してください。
    *   検索結果に基づく回答の場合、「社内ナレッジによると...」や「社内ナレッジによると...」と出典を明示すると信頼性が高まります。
"""

logger = logging.getLogger(__name__)

tools_map: Dict[str, Any] = { # tools_map is a dictionary of functions
    'search_rag_knowledge_base': search_rag_knowledge_base,
    'list_rag_collections': list_rag_collections
}

def setup_logging() -> logging.Logger:
    log_file_path: Path = PathConfig.LOG_DIR / AgentConfig.CHAT_LOG_FILE_NAME
    PathConfig.ensure_dirs()
    logging.basicConfig(
        level=getattr(logging, AgentConfig.CHAT_LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file_path, encoding='utf-8')]
    )
    return logging.getLogger(__name__)

def setup_agent() -> ChatSession: # Return type ChatSession
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    tools_list: List[Any] = [search_rag_knowledge_base, list_rag_collections] # List of functions
    model: GenerativeModel = genai.GenerativeModel(
        model_name=AgentConfig.MODEL_NAME,
        tools=tools_list,
        system_instruction=SYSTEM_INSTRUCTION
    )
    chat: ChatSession = model.start_chat(enable_automatic_function_calling=False)
    return chat

def print_colored(text: str, color: str = "white") -> None:
    colors: Dict[str, str] = {
        "cyan": "\033[96m", "green": "\033[92m", "yellow": "\033[93m",
        "red": "\033[91m", "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def run_agent_turn(chat_session: ChatSession, user_input: str, return_tool_info: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """
    Executes a single turn of the agent (User Input -> [Tools] -> Agent Response).
    This function handles the ReAct loop internally and returns the final response.
    
    Args:
        chat_session: The Gemini chat session object.
        user_input (str): The user's query.
        return_tool_info (bool): If True, returns (final_response_text, tool_info_dict).
                                 Otherwise, returns final_response_text.
                                 
    Returns:
        Union[str, Tuple[str, Dict[str, Any]]]: Agent's final response and optionally tool usage info.
    """
    logger.info(f"User Input: {user_input}")
    
    tool_info: Dict[str, Any] = {"tool_used": False, "tool_name": None, "collection_name": None}
    final_response_text: str = ""
    
    response = chat_session.send_message(user_input)
    
    while True:
        function_call_found: bool = False
        
        for part in response.parts:
            if part.text:
                log_message: str = part.text.strip()
                if "Thought:" in log_message or "考え:" in log_message:
                    logger.info(f"Agent Thought: {log_message}")
                else:
                    final_response_text = log_message
                    logger.info(f"Agent Response: {log_message}")

            if part.function_call:
                function_call_found = True
                fn = part.function_call
                tool_name: str = fn.name
                tool_args: Dict[str, Any] = dict(fn.args) # type: ignore
                
                logger.info(f"Agent Tool Call: {tool_name}({tool_args})")
                
                tool_info["tool_used"] = True
                tool_info["tool_name"] = tool_name
                if "collection_name" in tool_args:
                    tool_info["collection_name"] = tool_args["collection_name"]
                
                tool_result: str = ""
                try:
                    if tool_name in tools_map:
                        # mypy will complain about dynamic **tool_args, but it's valid at runtime
                        tool_result = tools_map[tool_name](**tool_args) 
                    else:
                        tool_result = f"Error: Tool '{tool_name}' not found."
                        logger.warning(f"Attempted to call unknown tool: {tool_name}")
                except RAGToolError as e: # Catch custom RAG tool errors
                    tool_result = f"エラーが発生しました: {str(e)}"
                    logger.error(f"RAG Tool Error during '{tool_name}': {e}")
                except Exception as e:
                    tool_result = f"予期せぬエラー: {str(e)}"
                    logger.error(f"Unexpected error during tool '{tool_name}': {e}", exc_info=True)

                log_tool_result: str = str(tool_result)[:500] + "..." if len(str(tool_result)) > 500 else str(tool_result)
                logger.info(f"Tool Result: {log_tool_result}")
                
                response = chat_session.send_message(
                    [genai.protos.Part(
                        function_response={
                            "name": tool_name,
                            "response": {'result': tool_result}
                        }
                    )]
                )
                break 

        if function_call_found:
            continue
        else:
            break
            
    if return_tool_info:
        return final_response_text, tool_info
    else:
        return final_response_text

def main() -> None:
    logger = setup_logging()
    print("🤖 Hybrid Knowledge Agent (ReAct + CoT) Started!")
    print("------------------------------------------------")
    print("一般的な質問と専門知識（RAG）を自律的に使い分け、思考プロセスを表示します。")
    print("終了するには 'exit' または 'quit' と入力してください。\n")
    
    logger.info(f"Agent session started at {datetime.datetime.now()}")

    try:
        chat_session: ChatSession = setup_agent()
    except Exception as e:
        print_colored(f"Error setting up agent: {e}", "red")
        logger.error(f"Error setting up agent: {e}")
        return

    while True:
        try:
            user_input: str = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested exit. Agent session ended.")
                print("Agent: Goodbye!")
                break
            
            print_colored(f"You: {user_input}", "reset")
            
            response_tuple: Union[str, Tuple[str, Dict[str, Any]]] = run_agent_turn(chat_session, user_input, return_tool_info=True)
            
            # Since return_tool_info is True, we expect a tuple
            if isinstance(response_tuple, tuple):
                response_text, _ = response_tuple
            else: # Fallback in case run_agent_turn's behavior changes
                response_text = response_tuple

            print(f"\nAgent: {response_text}")

        except KeyboardInterrupt:
            logger.info("User interrupted with Ctrl+C. Agent session ended.")
            print("\nAgent: Goodbye!")
            break
        except Exception as e:
            print_colored(f"\nError during chat: {e}", "red")
            logger.error(f"Error during chat session: {e}", exc_info=True)
            continue

if __name__ == "__main__":
    main()