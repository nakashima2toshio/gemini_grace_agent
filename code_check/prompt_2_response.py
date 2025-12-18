# Simulate Prompt to Response and Thought Extraction
#
# このスクリプトは、ユーザーのプロンプトがGeminiモデルに送られ、
# その応答から「Thought (思考)」部分がどのように抽出されるかをシミュレートします。
# 実際のRAGエージェントの簡略化された動作を再現します。
#具体的には、
# 1. ファイル名: ui/pages/agent_chat_page.py
# 2. 関数名: run_agent_turn 内で、LLM が生成する Thought (思考) の部分
# 3. 処理: SYSTEM_INSTRUCTION_TEMPLATE に基づき、LLM
#    自身が「なぜ検索が必要か、どのコレクションを、どんなクエリで検索するか」を思考し、
#    必要に応じて元のユーザーの質問から検索に適したクエリを生成（最適化）します
# 実行には環境変数 `GEMINI_API_KEY` が必要です。

import os
import re
import google.generativeai as genai

# ## 1. 定義されている System Instruction
# エージェントは以下の指示に従って、回答の前に必ず `Thought:` を出力するように設計されています。
SYSTEM_INSTRUCTION_TEMPLATE = """
あなたは、社内ドキュメント検索システムと連携した「ハイブリッド・ナレッジ・エージェント」です。
あなたの役割は、ユーザーの質問に対して、一般的な知識と、提供されたツール（社内ナレッジ検索）を適切に使い分けて回答することです。

## ReAct プロセスと出力フォーマット (厳守)

あなたは **Thought (思考)**、**Action (ツール実行)**、**Observation (結果観察)** のサイクルを回して回答に到達する必要があります。

### 1. ツールを使用する場合（検索が必要な場合）
必ず以下の形式で思考を出力してから、ツールを呼び出してください。
**Thought: [なぜ検索が必要か、どのコレクションを、どんなクエリで検索するか]**
(この後にツール呼び出しが行われます)

### 2. 最終回答を行う場合（検索が完了した、または検索不要な場合）
必ず以下の形式で出力してください。
**Thought: [得られた情報に基づいてどう回答するか、または検索結果がなかった場合の判断]**
**Answer: [ユーザーへの最終的な回答]**
"""

# ## 2. Thought 抽出ロジックの実装
# `code_check/thought_check.py` から `extract_thought_only` 関数をコピーします。
def extract_thought_only(model_response_text: str) -> str:
    """
    モデルの応答テキストから 'Thought:' (または '考え:') で始まる行/ブロックを抽出する。
    """
    lines = model_response_text.split('\n')
    thought_lines = []
    recording = False
    
    for line in lines:
        clean_line = line.strip()
        
        # 開始判定
        if clean_line.startswith("Thought:") or clean_line.startswith("考え:") or clean_line.startswith("**Thought:**"):
            if not recording:
                thought_lines.append(line) 
            recording = True
        elif recording and (clean_line.startswith("Answer:") or clean_line.startswith("**Answer:**") or clean_line.startswith("Action:")):
            # 終了判定
            recording = False
            break

        elif recording:
            thought_lines.append(line)
            
    cleaned_extracted_thoughts = []
    for t_line in thought_lines:
        if t_line.strip().startswith("Thought:"):
            cleaned_extracted_thoughts.append(t_line.strip()[len("Thought:"):].strip())
        elif t_line.strip().startswith("考え:"):
            cleaned_extracted_thoughts.append(t_line.strip()[len("考え:"):].strip())
        elif t_line.strip().startswith("**Thought:**"):
            cleaned_extracted_thoughts.append(t_line.strip()[len("**Thought:**"):].strip())
        else:
            cleaned_extracted_thoughts.append(t_line.strip())

    return "\n".join(cleaned_extracted_thoughts).strip()


def main():
    print("--- Prompt to Response Simulation ---")

    # Gemini API キーの設定を確認
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\nエラー: 環境変数 'GOOGLE_API_KEY' が設定されていません。")
        print("APIキーを設定してからスクリプトを実行してください。")
        print("例: export GEMINI_API_KEY='YOUR_API_KEY'")
        return

    genai.configure(api_key=api_key)

    # Gemini モデルの初期化
    # 実際のRAGエージェントではツールが渡されますが、ここではThoughtの生成をシミュレートするため、
    # ツールなしのモデルを使用します。モデルはそれでもThoughtを生成するよう促されます。
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash", # または "gemini-1.5-pro"
            system_instruction=SYSTEM_INSTRUCTION_TEMPLATE
        )
        chat = model.start_chat()
        print(f"Gemini model 'gemini-2.5-flash' initialized with system instruction.")
    except Exception as e:
        print(f"\nエラー: Geminiモデルの初期化に失敗しました: {e}")
        print("APIキーが有効であるか、モデル名が正しいか確認してください。")
        return

    print("\n" + "="*40 + "\n")

    # シミュレーション用のユーザープロンプト
    user_prompt = "Qdrantとは何ですか？また、どのように使われていますか？"
    print(f"User Prompt: {user_prompt}")

    # モデルからの応答を取得
    try:
        response = chat.send_message(user_prompt)
        
        # モデルのテキスト応答を結合
        model_full_text_response = ""
        for part in response.parts:
            if part.text:
                model_full_text_response += part.text

        print(f"\n--- Raw Model Response ---\n{model_full_text_response.strip()}")

        # Thought 部分の抽出
        extracted_thought = extract_thought_only(model_full_text_response)

        print(f"\n--- Extracted Thought Only ---\n{extracted_thought if extracted_thought else '(No Thought extracted)'}")

    except Exception as e:
        print(f"\nエラー: モデルからの応答取得中に問題が発生しました: {e}")
        print("ネットワーク接続やAPIレート制限を確認してください。")

    print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
