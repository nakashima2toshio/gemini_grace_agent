# check_async.py
"""
非同期処理（async/await）の学習用プログラム

Step1 → Step2 → Step3 を順番に実行し、
テキストを意味的なチャンクに分割する全体フローを確認する。

【非同期処理のポイント】
1. async def: 非同期関数を定義
2. await: 非同期関数の完了を待つ
3. asyncio.run(): 非同期処理を開始するエントリーポイント

【処理フロー】
Step1: テキスト → 段落リスト
Step2: 段落リスト → チャンクリスト（意味的分割）
Step3: チャンクリスト → 最終チャンクリスト（連続性チェック・結合）
"""

import asyncio
import os
from google import genai
from google.genai import types

# chunking モジュールからインポート
from chunking.models import StructuralResult, ContinuityResult
from chunking.prompts import (
    PARAGRAPH_SEPARATION_PROMPT,
    SEMANTIC_CHUNKING_PROMPT,
    CONTINUITY_CHECK_PROMPT
)


# ================================================================
# Step1: 階層構造化（段落分割）
# ================================================================
async def step1_hierarchical_split(text: str, client: genai.Client) -> list[str]:
    """
    テキストを段落単位に分割する

    Args:
        text: 入力テキスト
        client: Gemini API クライアント

    Returns:
        段落のリスト
    """
    print("\n" + "=" * 50)
    print("【Step1】階層構造化（段落分割）")
    print("=" * 50)

    # プロンプト作成
    prompt = f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{text}"

    # API呼び出し（同期だが、awaitで他の処理に制御を渡せる）
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StructuralResult
        )
    )

    # パース
    result = StructuralResult.model_validate_json(response.text)
    paragraphs = [para.full_text for para in result.paragraphs]

    print(f"入力: {len(text)}文字 → 出力: {len(paragraphs)}段落")
    for i, para in enumerate(paragraphs, 1):
        preview = para.replace('\n', ' ')[:60]
        print(f"  段落{i}: {preview}...")

    return paragraphs


# ================================================================
# Step2: 意味的分割
# ================================================================
async def step2_semantic_chunking(paragraphs: list[str], client: genai.Client) -> list[str]:
    """
    段落を意味的なチャンクに分割する

    Args:
        paragraphs: 段落のリスト
        client: Gemini API クライアント

    Returns:
        チャンクのリスト
    """
    print("\n" + "=" * 50)
    print("【Step2】意味的分割")
    print("=" * 50)

    chunks = []

    for i, para in enumerate(paragraphs):
        print(f"段落{i + 1}/{len(paragraphs)} 処理中...")

        # プロンプト作成
        prompt = f"{SEMANTIC_CHUNKING_PROMPT}\n\n【入力テキスト】\n{para}"

        # API呼び出し
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuralResult
            )
        )

        # パース
        result = StructuralResult.model_validate_json(response.text)

        for chunk_para in result.paragraphs:
            chunks.append(chunk_para.full_text)

        print(f"  → {len(result.paragraphs)}チャンクに分割")

        # 非同期のポイント: 他のタスクに制御を渡す
        await asyncio.sleep(0)  # イベントループに制御を戻す

    print(f"合計: {len(paragraphs)}段落 → {len(chunks)}チャンク")
    return chunks


# ================================================================
# Step3: 文脈連続性チェック
# ================================================================
async def step3_continuity_check(chunks: list[str], client: genai.Client) -> list[str]:
    """
    隣接チャンク間の連続性をチェックし結合/分離する

    Args:
        chunks: チャンクのリスト
        client: Gemini API クライアント

    Returns:
        最終チャンクリスト
    """
    print("\n" + "=" * 50)
    print("【Step3】文脈連続性チェック")
    print("=" * 50)

    if len(chunks) <= 1:
        print("チャンク数が1以下のため、スキップ")
        return chunks

    # 隣接ペアの連続性を判定
    continuity_flags = []

    for i in range(len(chunks) - 1):
        print(f"ペア{i + 1}/{len(chunks) - 1} 判定中...")

        # プロンプト作成
        prompt = f"{CONTINUITY_CHECK_PROMPT}\n\n【前のテキスト】\n{chunks[i]}\n\n【次のテキスト】\n{chunks[i + 1]}"

        # API呼び出し
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ContinuityResult
            )
        )

        # パース
        result = ContinuityResult.model_validate_json(response.text)
        continuity_flags.append(result.is_connected)

        status = "連続→結合" if result.is_connected else "非連続→分離"
        print(f"  → {status}")

        # 非同期のポイント: 他のタスクに制御を渡す
        await asyncio.sleep(0)

    # マージ処理
    print("\nマージ処理...")
    final_chunks = [chunks[0]]

    for i, is_connected in enumerate(continuity_flags):
        if is_connected:
            final_chunks[-1] += "\n\n" + chunks[i + 1]
        else:
            final_chunks.append(chunks[i + 1])

    print(f"合計: {len(chunks)}チャンク → {len(final_chunks)}チャンク")
    return final_chunks


# ================================================================
# メイン処理（全Stepを順番に実行）
# ================================================================
async def process_text(text: str, api_key: str) -> list[str]:
    """
    テキストを処理する全体フロー

    【非同期処理の流れ】
    await step1 → 完了を待つ → await step2 → 完了を待つ → await step3

    Args:
        text: 入力テキスト
        api_key: Gemini API キー

    Returns:
        最終チャンクリスト
    """
    # クライアント初期化
    client = genai.Client(api_key=api_key)

    # Step1: テキスト → 段落
    paragraphs = await step1_hierarchical_split(text, client)

    # Step2: 段落 → チャンク（意味的分割）
    chunks = await step2_semantic_chunking(paragraphs, client)

    # Step3: チャンク → 最終チャンク（連続性チェック）
    final_chunks = await step3_continuity_check(chunks, client)

    return final_chunks


async def main():
    """
    エントリーポイント

    【非同期処理の基本構造】
    async def main():      # 非同期関数として定義
        result = await xxx()  # 非同期関数を呼び出し、完了を待つ

    asyncio.run(main())    # イベントループを起動して実行
    """

    # APIキー取得
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY 環境変数を設定してください")
        return

    # テスト用テキスト
    test_text = """第1章 人工知能の基礎
人工知能（AI）は、コンピュータに人間のような知能を持たせる技術です。
機械学習やディープラーニングがその中核をなしています。

第2章 機械学習の手法
教師あり学習では、ラベル付きデータから学習します。
代表的な手法には、ランダムフォレストやサポートベクターマシンがあります。
ところで、昨日食べたラーメンが美味しかったです。
次回も同じ店に行きたいと思います。

第3章 深層学習
深層学習は、多層のニューラルネットワークを用いる手法です。
画像認識や自然言語処理で高い性能を発揮します。"""

    print("=" * 50)
    print("【入力テキスト】")
    print("=" * 50)
    print(test_text)

    # 全Stepを実行（awaitで順番に処理）
    final_chunks = await process_text(test_text, api_key)

    # 最終結果表示
    print("\n" + "=" * 50)
    print("【最終結果】")
    print("=" * 50)
    for i, chunk in enumerate(final_chunks, 1):
        print(f"\n--- 最終チャンク{i} ({len(chunk)}文字) ---")
        print(chunk)

    print("\n" + "=" * 50)
    print("【処理完了】")
    print("=" * 50)


# ================================================================
# プログラム実行
# ================================================================
if __name__ == "__main__":
    # asyncio.run() で非同期処理を開始
    # これがイベントループを作成し、main()を実行する
    asyncio.run(main())
