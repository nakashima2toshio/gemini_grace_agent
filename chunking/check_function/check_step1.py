# check_step1.py
"""
Step1: 階層構造化（段落分割）の同期版・簡易確認プログラム

【目的】
テキストを段落単位に分割する。
見出し（第X章など）と本文は分離せず、1つの段落としてまとめる。

【処理の流れ】
1. 入力テキストをブロック（2000文字単位）に分割
2. 各ブロックをGemini APIに送信し、段落構造を抽出
3. 抽出された段落のリストを返す
"""

import os
from google import genai
from google.genai import types

# chunking モジュールからインポート
from chunking.models import StructuralResult
from chunking.prompts import PARAGRAPH_SEPARATION_PROMPT


def step1_hierarchical_split(text: str, api_key: str, block_size: int = 2000) -> list[str]:
    """
    テキストを段落単位に分割する（Step1のコア機能）

    Args:
        text: 入力テキスト
        api_key: Gemini API キー
        block_size: ブロックサイズ（文字数）

    Returns:
        段落のリスト
    """
    client = genai.Client(api_key=api_key)

    # テキストをブロックに分割
    blocks = [text[i:i + block_size] for i in range(0, len(text), block_size)]
    print(f"入力: {len(text)}文字 → {len(blocks)}ブロック")

    paragraphs = []

    for i, block in enumerate(blocks):
        print(f"ブロック {i + 1}/{len(blocks)} 処理中...")

        # プロンプト作成
        prompt = f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{block}"

        # Gemini API 呼び出し（同期）
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuralResult
            )
        )

        # レスポンスをパース
        result = StructuralResult.model_validate_json(response.text)

        # 段落を抽出
        for para in result.paragraphs:
            paragraphs.append(para.full_text)

        print(f"  → {len(result.paragraphs)}個の段落を抽出")

    return paragraphs


def main():
    """メイン処理"""

    # APIキー取得
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY 環境変数を設定してください")
        print("  export GOOGLE_API_KEY='your-api-key'")
        return

    # テスト用テキスト
    test_text = """第1章 人工知能の基礎
人工知能（AI）は、コンピュータに人間のような知能を持たせる技術です。
機械学習やディープラーニングがその中核をなしています。

第2章 機械学習の手法
教師あり学習では、ラベル付きデータから学習します。
代表的な手法には、ランダムフォレストやサポートベクターマシンがあります。

ところで、昨日食べたラーメンが美味しかったです。
次回も同じ店に行きたいと思います。"""

    print("=" * 50)
    print("【入力テキスト】")
    print("=" * 50)
    print(test_text)
    print()

    # Step1 実行
    print("=" * 50)
    print("【Step1 実行】")
    print("=" * 50)
    paragraphs = step1_hierarchical_split(test_text, api_key)

    # 結果表示
    print()
    print("=" * 50)
    print(f"【結果】{len(paragraphs)}個の段落")
    print("=" * 50)
    for i, para in enumerate(paragraphs, 1):
        print(f"\n--- 段落{i} ({len(para)}文字) ---")
        print(para)

    # 検証ポイント
    print()
    print("=" * 50)
    print("【検証ポイント】")
    print("=" * 50)
    print("✓ 見出し（第X章）と本文が1つの段落にまとまっているか")
    print("✓ 空行や章の変わり目で分割されているか")
    print("✓ テキストが省略されず保持されているか")


if __name__ == "__main__":
    main()
