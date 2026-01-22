# check_step2.py
"""
Step2: 意味的分割（Semantic Chunking）の同期版・簡易確認プログラム

【目的】
段落を意味的な類似度に基づいて再構成する。
話題の転換点で分割し、形式的な改行ではなく意味のまとまりで分割する。

【処理の流れ】
1. Step1の出力（段落リスト）を入力として受け取る
2. 各段落をGemini APIに送信し、意味的なチャンクに分割
3. 分割されたチャンクのリストを返す
"""

import os
from google import genai
from google.genai import types

# chunking モジュールからインポート
from chunking.models import StructuralResult
from chunking.prompts import SEMANTIC_CHUNKING_PROMPT


def step2_semantic_chunking(paragraphs: list[str], api_key: str) -> list[str]:
    """
    段落を意味的なチャンクに分割する（Step2のコア機能）

    Args:
        paragraphs: 段落のリスト（Step1の出力）
        api_key: Gemini API キー

    Returns:
        意味的に分割されたチャンクのリスト
    """
    client = genai.Client(api_key=api_key)

    print(f"入力: {len(paragraphs)}段落")

    chunks = []

    for i, para in enumerate(paragraphs):
        print(f"段落 {i + 1}/{len(paragraphs)} 処理中...")

        # プロンプト作成
        prompt = f"{SEMANTIC_CHUNKING_PROMPT}\n\n【入力テキスト】\n{para}"

        # Gemini API 呼び出し（同期）
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuralResult
            )
        )

        # レスポンスをパース
        result = StructuralResult.model_validate_json(response.text)

        # チャンクを抽出
        for chunk_para in result.paragraphs:
            chunks.append(chunk_para.full_text)

        print(f"  → {len(result.paragraphs)}個のチャンクに分割")

    return chunks


def main():
    """メイン処理"""

    # APIキー取得
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY 環境変数を設定してください")
        print("  export GOOGLE_API_KEY='your-api-key'")
        return

    # テスト用段落（Step1の出力を想定）
    # 話題が混在している段落を含む
    test_paragraphs = [
        """第1章 人工知能の基礎
人工知能（AI）は、コンピュータに人間のような知能を持たせる技術です。
機械学習やディープラーニングがその中核をなしています。
AIの研究は1950年代から始まり、現在では様々な分野で応用されています。""",

        """第2章 機械学習の手法
教師あり学習では、ラベル付きデータから学習します。
代表的な手法には、ランダムフォレストやサポートベクターマシンがあります。
一方、教師なし学習では、ラベルのないデータからパターンを発見します。
クラスタリングや次元削減などが代表的な手法です。""",

        """強化学習は、エージェントが環境と相互作用しながら学習する手法です。
報酬を最大化するように行動を学習していきます。
ゲームAIやロボット制御などに応用されています。
ところで、昨日食べたラーメンが美味しかったです。
次回も同じ店に行きたいと思います。
話を戻すと、深層強化学習はDeep Learningと強化学習を組み合わせた手法です。"""
    ]

    print("=" * 50)
    print("【入力段落（Step1の出力）】")
    print("=" * 50)
    for i, para in enumerate(test_paragraphs, 1):
        print(f"\n--- 段落{i} ---")
        print(para)
    print()

    # Step2 実行
    print("=" * 50)
    print("【Step2 実行】")
    print("=" * 50)
    chunks = step2_semantic_chunking(test_paragraphs, api_key)

    # 結果表示
    print()
    print("=" * 50)
    print(f"【結果】{len(test_paragraphs)}段落 → {len(chunks)}チャンク")
    print("=" * 50)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- チャンク{i} ({len(chunk)}文字) ---")
        print(chunk)

    # 検証ポイント
    print()
    print("=" * 50)
    print("【検証ポイント】")
    print("=" * 50)
    print("✓ 意味的に類似した内容が同じチャンクにまとまっているか")
    print("✓ 話題転換（ラーメンの話）が別チャンクに分離されるか")
    print("✓ テキストが省略されず保持されているか")


if __name__ == "__main__":
    main()
