# Thought Extraction Logic Check
#
# このスクリプトでは、`agent_rag.py` (実体は `ui/pages/agent_chat_page.py`) で使用されている
# **ReAct Agent** の「思考 (Thought)」部分の抽出ロジックを確認・検証します。
#
# ユーザーの入力 (`prompt`) に対して、エージェントがどのように「思考」を出力し、
# システムがそれをどう抜き出しているかをシミュレーションします。

# ## 1. 定義されている System Instruction
#
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

# print("System Instruction Loaded.") # main関数に移動

# ## 2. Thought 抽出ロジックの実装
#
# `ui/pages/agent_chat_page.py` の `run_agent_turn` 関数内で使用されているロジックを再現した関数です。
# Gemini APIのレスポンスオブジェクトを模倣したテキスト入力から、思考部分を抽出します。

import re

def extract_thought_only(model_response_text: str) -> str:
    """
    モデルの応答テキストから 'Thought:' (または '考え:') で始まる行/ブロックを抽出する。
    実際のシステムでは response.parts をループして 'Thought:' を含む part を抽出しているが、
    ここではテキスト全体からの抽出をシミュレートする。
    """
    lines = model_response_text.split('\n')
    thought_lines = []
    recording = False
    
    for line in lines:
        clean_line = line.strip()
        
        # 開始判定
        if clean_line.startswith("Thought:") or clean_line.startswith("考え:") or clean_line.startswith("**Thought:**"):
            # Thought行自体も抽出対象に含める
            if not recording: # 複数Thoughtが連続する場合に重複しないように
                thought_lines.append(line) 
            recording = True
        elif recording and (clean_line.startswith("Answer:") or clean_line.startswith("**Answer:**") or clean_line.startswith("Action:")):
            # 終了判定 (Answer: や Action: が来たら思考終了とみなす)
            recording = False
            break # 終了マーカーを見つけたらループを抜ける

        elif recording:
            thought_lines.append(line)
            
    # Thought: や 考え: のプレフィックスを取り除く
    # 例: "Thought: This is a thought." -> "This is a thought."
    # これは元の agent_chat_page.py では Thought: を含む行全体をログに記録しているため厳密には異なるが、
    # 「Thoughtの部分だけを抜き出して」という意図にはこちらが近い。
    
    # ここでThought: や 考え: のプレフィックスを削除する
    # 各行から Thought: や 考え: を除去し、先頭と末尾の空白も除去
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
    print("System Instruction Loaded.")

    # ## 3. シミュレーション: ユーザーの入力に対するレスポンス解析
    #
    # いくつかのパターンで、ユーザーの入力 (`prompt`) に対するモデルの応答を想定し、
    # そこから `Thought` がどう見えるかを確認します。

    # パターンA: 検索が必要なケース
    user_prompt_a = "社内の規定について教えて"
    model_response_a = """
Thought: ユーザーは社内規定について尋ねている。これは社内ナレッジ（livedoorコレクションなど）を検索する必要がある。
クエリは「社内規定」で検索を実行する。
"""

    # パターンB: 最終回答のケース
    user_prompt_b = "ありがとう"
    model_response_b = """
Thought: ユーザーは感謝を述べている。これは一般的な会話なので、検索は不要である。
Answer: どういたしまして！他に何かお手伝いできることはありますか？
"""

    # パターンC: フォーマットが少し崩れたケース (Bold Thought)
    user_prompt_c = "Geminiについて"
    model_response_c = """
**Thought:** 
Geminiに関する質問だ。最新情報は wikipedia_ja コレクションにあるかもしれない。
まずは wikipedia_ja で「Gemini」を検索してみよう。
"""

    # パターンD: Thoughtがなく、直接Answerが来るケース (本来のReActフローではありえないが、テスト用)
    user_prompt_d = "こんにちは"
    model_response_d = """
Answer: こんにちは！何かお手伝いできることはありますか？
"""

    # 実行と確認
    test_cases = [
        (user_prompt_a, model_response_a),
        (user_prompt_b, model_response_b),
        (user_prompt_c, model_response_c),
        (user_prompt_d, model_response_d), # Thoughtがないケース
    ]

    for i, (prompt, resp) in enumerate(test_cases, 1):
        print(f"--- Case {i} ---")
        print(f"User Prompt: {prompt}")
        print(f"Raw Response: \n{resp.strip()}\n")
        
        # 抽出実行
        extracted = extract_thought_only(resp)
        print(f"> Extracted Thought Only:\n{extracted if extracted else '(No Thought extracted)'}")
        print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()