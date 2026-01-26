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
Step1: テキスト → 段落リスト（階層構造化）
Step2: 段落リスト → チャンクリスト（意味的分割）
Step3: チャンクリスト → 最終チャンクリスト（連続性チェック・結合）

【検証パターン】
- 前方依存: 「この」「それ」等の指示語で前を参照
- 後方依存: 専門用語が未定義のまま使用される
- 独立判定: 話題は同じでも単独で理解可能
- 章構造: 章が変わった場合の独立性
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
async def step1_hierarchical_split(text: str, client: genai.Client, block_size: int = 2000) -> list[str]:
    """
    テキストを段落単位に分割する

    【目的】
    テキストを段落単位に分割する。
    見出し（第X章など）と本文は分離せず、1つの段落としてまとめる。

    【分割ルール】
    - 空行（\\n\\n）が存在する箇所のみで分割
    - 見出しと直後の本文は空行がなければ同じ段落に
    - 章が変わっても空行がなければ分割しない
    - 改行（\\n）だけでは分割しない

    【処理の流れ】
    1. 入力テキストをブロック（2000文字単位）に分割
    2. 各ブロックをGemini APIに送信し、段落構造を抽出
    3. 抽出された段落のリストを返す

    Args:
        text: 入力テキスト
        client: Gemini API クライアント
        block_size: ブロックサイズ（文字数）

    Returns:
        段落のリスト
    """
    print("\n" + "=" * 50)
    print("【Step1】階層構造化（段落分割）")
    print("=" * 50)

    # テキストをブロックに分割
    blocks = [text[i:i + block_size] for i in range(0, len(text), block_size)]
    print(f"入力: {len(text)}文字 → {len(blocks)}ブロック")

    paragraphs = []

    for i, block in enumerate(blocks):
        print(f"ブロック {i + 1}/{len(blocks)} 処理中...")

        # プロンプト作成
        prompt = f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{block}"

        # Gemini API 呼び出し（同期だが、awaitで他の処理に制御を渡せる）
        # gemini-2.5-flash: 最新の安定版、高いレート制限とパフォーマンス
        # URL: https://ai.google.dev/gemini-api/docs/text-generation?lang=python
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuralResult
            )
        )

        # パース
        result = StructuralResult.model_validate_json(response.text)

        # 段落を抽出
        for para in result.paragraphs:
            paragraphs.append(para.full_text)

        print(f"  → {len(result.paragraphs)}個の段落を抽出")

        # 非同期のポイント: 他のタスクに制御を渡す
        await asyncio.sleep(0)

    print(f"合計: {len(blocks)}ブロック → {len(paragraphs)}段落")

    return paragraphs


# ================================================================
# Step2: 意味的分割
# ================================================================
async def step2_semantic_chunking(paragraphs: list[str], client: genai.Client) -> list[str]:
    """
    段落を意味的なチャンクに分割する

    【目的】
    段落を意味的な類似度に基づいて再構成する。
    話題の転換点で分割し、形式的な改行ではなく意味のまとまりで分割する。

    【Step1との違い】
    - Step1: 物理的構造（空行のみ）で分割
    - Step2: 意味的な類似度（話題の転換）で分割
    - 章の変わり目（第1章→第2章）はStep2で分割

    【処理の流れ】
    1. Step1の出力（段落リスト）を入力として受け取る
    2. 各段落をGemini APIに送信し、意味的なチャンクに分割
    3. 分割されたチャンクのリストを返す

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

    # Step1との違い: Step1はブロック（2000文字）単位、Step2は段落単位で処理
    for i, para in enumerate(paragraphs):
        print(f"段落 {i + 1}/{len(paragraphs)} 処理中...")

        # プロンプト作成
        prompt = f"{SEMANTIC_CHUNKING_PROMPT}\n\n【入力テキスト】\n{para}"

        # Gemini API 呼び出し（同期）
        # gemini-2.5-flash: 最新の安定版、高いレート制限
        # URL: https://ai.google.dev/gemini-api/docs/text-generation?lang=python
        response = client.models.generate_content(
            model="gemini-2.5-flash",
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

        print(f"  → {len(result.paragraphs)}個のチャンクに分割")

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

    【目的】
    隣接するチャンク間の文脈連続性を判定し、
    連続している場合は結合、非連続の場合は分離する。

    【Step2との違い】
    - Step2: 分割（1段落→複数チャンク、チャンク数が増加）
    - Step3: 結合（複数チャンク→少数チャンク、チャンク数が減少）
    - Step3はStep2の「過分割」を修正する役割

    【処理の流れ】
    1. Step2の出力（チャンクリスト）を入力として受け取る
    2. 隣接ペアごとにGemini APIで連続性を判定
    3. 判定結果に基づいてチャンクを結合/分離
    4. 最終的なチャンクリストを返す

    【検証パターン】
    - 前方依存: 「この」「それ」等の指示語で前を参照 → 結合（True）
    - 後方依存: 専門用語が未定義のまま使用 → 結合（True）
    - 話題転換: 完全に別のトピック → 分離（False）
    - 独立判定: 話題は同じでも単独で理解可能 → 分離（False）
    - 章構造: 章が変わった場合 → 分離（False）

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

    print(f"入力: {len(chunks)}チャンク")

    # 隣接ペアの連続性を判定
    continuity_flags = []

    for i in range(len(chunks) - 1):
        print(f"ペア {i + 1}/{len(chunks) - 1} 判定中...")

        # プロンプト作成
        prompt = f"{CONTINUITY_CHECK_PROMPT}\n\n【前のテキスト】\n{chunks[i]}\n\n【次のテキスト】\n{chunks[i + 1]}"

        # Gemini API 呼び出し（同期）
        # gemini-2.5-flash: 最新の安定版、高いレート制限
        # URL: https://ai.google.dev/gemini-api/docs/text-generation?lang=python
        # Step1, Step2との違い: Step3はContinuityResult（ブール値のみ）を使用
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ContinuityResult  # ブール値のみを返す
            )
        )

        # パース
        result = ContinuityResult.model_validate_json(response.text)
        continuity_flags.append(result.is_connected)

        status = "連続 → 結合" if result.is_connected else "非連続 → 分離"
        print(f"  → {status}")

        # 非同期のポイント: 他のタスクに制御を渡す
        await asyncio.sleep(0)

    # マージ処理
    print()
    print("マージ処理...")
    final_chunks = [chunks[0]]

    for i, is_connected in enumerate(continuity_flags):
        if is_connected:
            # 結合: 空行（\n\n）で連結し、段落構造を保持
            final_chunks[-1] += "\n\n" + chunks[i + 1]
            print(f"  チャンク{i + 1} + チャンク{i + 2} → 結合")
        else:
            # 分離: 新しいチャンクとして追加
            final_chunks.append(chunks[i + 1])
            print(f"  チャンク{i + 2} → 新規追加")

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

    # ============================================================
    # テスト用テキスト
    # Step2・Step3で前方依存・後方依存・完全独立を検証するための入力
    # 空行（\n\n）で5つの段落に分割されることを期待
    # ============================================================
    test_text = """RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。
外部知識ベースから関連情報を取得し、それをLLMのコンテキストとして渡します。
2020年にFacebookが発表し、現在では多くのシステムで採用されています。
この手法の最大の利点は、最新情報を反映できることです。
それにより、LLM単体では対応できない時事的な質問にも回答可能になります。
また、ハルシネーションを軽減する効果も報告されています。

セマンティックチャンキングは、テキストを意味単位で分割する技術です。
「チャンク」とは、分割されたテキストの各ブロックを指します。
「埋め込み」（Embedding）は、テキストを数値ベクトルに変換したものです。
チャンクサイズは検索精度に大きく影響します。
小さすぎると文脈が失われ、埋め込みの品質が低下します。
大きすぎると検索ノイズが増加し、関連性の低い情報が混入します。

京都の紅葉は11月中旬から下旬が見頃です。
清水寺や嵐山が特に人気のスポットとして知られています。
混雑を避けるなら平日の早朝がおすすめです。
沖縄の海は透明度が高く、シュノーケリングに最適です。
那覇から車で約1時間の恩納村には美しいビーチが点在しています。
夏季は台風に注意が必要ですが、それ以外の季節も温暖で過ごしやすいです。

ベクトルデータベースは、高次元ベクトルを効率的に格納・検索するシステムです。
代表的な製品にPinecone、Weaviate、Chromaなどがあります。
ANN（Approximate Nearest Neighbor）アルゴリズムにより高速な類似検索を実現します。
ANNの精度とスピードはトレードオフの関係にあります。
HNSWやIVFなどのインデックス手法を選択することで、このバランスを調整できます。
ベクトルDBの選定では、スケーラビリティとコストも重要な判断基準となります。

第1章 機械学習入門
機械学習は、データからパターンを学習するアルゴリズムの総称です。
教師あり学習、教師なし学習、強化学習の3つに大別されます。
本章では、これらの基本概念を解説しました。
第2章 深層学習の基礎
深層学習は、多層のニューラルネットワークを用いる機械学習の一手法です。
画像認識や自然言語処理で革命的な成果を上げています。
本章では、CNNとRNNの基本アーキテクチャを説明します。"""

    # ============================================================
    # テスト用テキスト2（空行なし版）
    # Step1で1段落として認識され、Step2で意味的に分割されることを検証
    # ============================================================
    test_text2 = """RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。
外部知識ベースから関連情報を取得し、それをLLMのコンテキストとして渡します。
2020年にFacebookが発表し、現在では多くのシステムで採用されています。
この手法の最大の利点は、最新情報を反映できることです。
それにより、LLM単体では対応できない時事的な質問にも回答可能になります。
また、ハルシネーションを軽減する効果も報告されています。
セマンティックチャンキングは、テキストを意味単位で分割する技術です。
「チャンク」とは、分割されたテキストの各ブロックを指します。
「埋め込み」（Embedding）は、テキストを数値ベクトルに変換したものです。
チャンクサイズは検索精度に大きく影響します。
小さすぎると文脈が失われ、埋め込みの品質が低下します。
大きすぎると検索ノイズが増加し、関連性の低い情報が混入します。
京都の紅葉は11月中旬から下旬が見頃です。
清水寺や嵐山が特に人気のスポットとして知られています。
混雑を避けるなら平日の早朝がおすすめです。
沖縄の海は透明度が高く、シュノーケリングに最適です。
那覇から車で約1時間の恩納村には美しいビーチが点在しています。
夏季は台風に注意が必要ですが、それ以外の季節も温暖で過ごしやすいです。
ベクトルデータベースは、高次元ベクトルを効率的に格納・検索するシステムです。
代表的な製品にPinecone、Weaviate、Chromaなどがあります。
ANN（Approximate Nearest Neighbor）アルゴリズムにより高速な類似検索を実現します。
ANNの精度とスピードはトレードオフの関係にあります。
HNSWやIVFなどのインデックス手法を選択することで、このバランスを調整できます。
ベクトルDBの選定では、スケーラビリティとコストも重要な判断基準となります。
第1章 機械学習入門
機械学習は、データからパターンを学習するアルゴリズムの総称です。
教師あり学習、教師なし学習、強化学習の3つに大別されます。
本章では、これらの基本概念を解説しました。
第2章 深層学習の基礎
深層学習は、多層のニューラルネットワークを用いる機械学習の一手法です。
画像認識や自然言語処理で革命的な成果を上げています。
本章では、CNNとRNNの基本アーキテクチャを説明します。"""

    print("=" * 50)
    print("【入力テキスト】")
    print("=" * 50)
    print(test_text)
    # print(test_text2)  # 空行なし版をテストする場合はこちらを使用

    print()
    print("【期待される処理結果】")
    print("=" * 50)
    print("Step1実行後: 1テキスト → 5段落")
    print("  段落1: RAGの説明（定義 + 利点）")
    print("  段落2: セマンティックチャンキングの説明（用語定義 + 用語使用）")
    print("  段落3: 観光情報（京都 + 沖縄）")
    print("  段落4: ベクトルDBの説明（定義 + 活用）")
    print("  段落5: 章構造（第1章 + 第2章）")
    print()
    print("Step2実行後: 5段落 → 10チャンク")
    print("  チャンク1-2: RAG（定義 / 利点）")
    print("  チャンク3-4: チャンキング（用語定義 / 用語使用）")
    print("  チャンク5-6: 観光（京都 / 沖縄）")
    print("  チャンク7-8: ベクトルDB（定義 / 活用）")
    print("  チャンク9-10: 章構造（第1章 / 第2章）")
    print()
    print("Step3実行後: 10チャンク → 7チャンク")
    print("  最終チャンク1: チャンク1+2（前方依存で結合）")
    print("  最終チャンク2: チャンク3+4（後方依存で結合）")
    print("  最終チャンク3: チャンク5（独立）")
    print("  最終チャンク4: チャンク6（独立）")
    print("  最終チャンク5: チャンク7+8（後方依存で結合）")
    print("  最終チャンク6: チャンク9（独立）")
    print("  最終チャンク7: チャンク10（独立）")
    print("=" * 50)

    # 全Stepを実行（awaitで順番に処理）
    final_chunks = await process_text(test_text, api_key)
    # final_chunks = await process_text(test_text2, api_key)  # 空行なし版

    # 最終結果表示
    print("\n" + "=" * 50)
    print("【最終結果】")
    print("=" * 50)
    for i, chunk in enumerate(final_chunks, 1):
        print(f"\n--- 最終チャンク{i} ({len(chunk)}文字) ---")
        print(chunk)

    print("\n" + "=" * 50)
    print("【結果検証】")
    print("=" * 50)
    expected_chunks = 7
    if len(final_chunks) == expected_chunks:
        print(f"✅ 期待通り {expected_chunks} チャンクに結合されました")
    else:
        print(f"⚠️  期待: {expected_chunks} チャンク, 実際: {len(final_chunks)} チャンク")
        print("   結合/分離が期待と異なる場合、プロンプトの調整が必要な可能性があります")

    print("\n" + "=" * 50)
    print("【検証ポイント】")
    print("=" * 50)
    print("✓ 前方依存: 「この」「それ」等の指示語で前を参照 → 結合されるか")
    print("✓ 後方依存: 専門用語が未定義のまま使用 → 結合されるか")
    print("✓ 話題転換: 完全に別のトピック → 分離されるか")
    print("✓ 独立判定: 話題は同じでも単独で理解可能 → 分離されるか")
    print("✓ 章構造: 章が変わった場合 → 分離されるか")
    print("✓ 結合後のテキストが正しく保持されているか")

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
