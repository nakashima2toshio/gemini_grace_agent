# check_step2.py
"""
_step2_semantic_chunking 関数の動作確認プログラム

このプログラムは以下を確認します:
1. 段落が意味的な類似度に基づいて再構成されているか
2. 話題の転換点で適切に分割されているか
3. 形式的な改行ではなく、意味のまとまりで分割されているか

実行方法:
  プロジェクトルートから: python -m chunking.check_function.check_step2
"""

import asyncio
import os
from typing import List, Any, Coroutine

# 絶対インポート（推奨）
from chunking.models import StructuralResult, ParagraphUnit
from chunking.prompts import SEMANTIC_CHUNKING_PROMPT

# Gemini APIクライアント用（実装は簡易版を使用）
from google import genai
from google.genai import types


# ================================================================
# 簡易版 AsyncAPIClient（テスト用）
# ================================================================

class SimpleAsyncAPIClient:
    """テスト用の簡易APIクライアント"""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    async def generate_content(
            self,
            model: str,
            contents: str,
            response_schema: type,
            task_id: str = ""
    ) -> str | None:
        """
        Gemini APIを呼び出してJSON構造化レスポンスを取得

        Args:
            model: モデル名
            contents: プロンプト
            response_schema: Pydanticモデル（レスポンススキーマ）
            task_id: タスクID（ログ用）

        Returns:
            JSON文字列
        """
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema
                )
            )

            return response.text

        except Exception as e:
            print(f"API呼び出しエラー [{task_id}]: {e}")
            return None


# ================================================================
# Step2 関数（元の関数から非同期処理部分を簡略化）
# ================================================================

async def step2_semantic_chunking_test(
        paragraphs: List[str],
        client: SimpleAsyncAPIClient,
        model: str = "gemini-2.0-flash-exp"
) -> List[str]:
    """
    Step 2: 意味的分割のテスト版

    Args:
        paragraphs: 段落のリスト（Step1の出力）
        client: APIクライアント
        model: 使用するモデル

    Returns:
        意味的に分割されたチャンクのリスト
    """
    print("=" * 60)
    print("[Step 2/3] 意味的分割（Semantic Chunking）")
    print("=" * 60)
    print(f"入力: {len(paragraphs)} 段落")
    print()

    # 各段落を処理
    chunks = []
    for i, para in enumerate(paragraphs):
        print(f"--- 段落 {i + 1}/{len(paragraphs)} を処理中... ---")
        print(f"入力段落（最初の100文字）: {para[:100]}...")
        print()

        # プロンプト作成
        prompt = f"{SEMANTIC_CHUNKING_PROMPT}\n\n【入力テキスト】\n{para}"

        # API呼び出し
        result_json = await client.generate_content(
            model=model,
            contents=prompt,
            response_schema=StructuralResult,
            task_id=f"step2_para_{i}"
        )

        # レスポンスをパース
        if result_json:
            try:
                result = StructuralResult.model_validate_json(result_json)

                # 各チャンクを抽出
                para_chunks = []
                for chunk_para in result.paragraphs:
                    chunk_text = chunk_para.full_text
                    chunks.append(chunk_text)
                    para_chunks.append(chunk_text)

                print(f"  ✓ {len(result.paragraphs)} 個のチャンクに分割")

                # 分割の詳細を表示
                if len(result.paragraphs) > 1:
                    print(f"  📊 分割詳細:")
                    for j, chunk in enumerate(para_chunks):
                        preview = chunk.replace('\n', ' ')[:80]
                        print(f"     チャンク{j + 1}: {preview}...")

            except Exception as e:
                print(f"  ✗ パース失敗: {e}")
        else:
            print("  ✗ API呼び出し失敗")

        print()

    print(f"合計 {len(chunks)} 個のチャンクを生成しました")
    print("=" * 60)

    return chunks


# ================================================================
# 結果表示関数
# ================================================================

def display_results(
        input_paragraphs: List[str],
        output_chunks: List[str]
):
    """
    処理前後の比較を見やすく表示

    Args:
        input_paragraphs: 入力段落のリスト
        output_chunks: 出力チャンクのリスト
    """
    print("\n")
    print("=" * 60)
    print("処理結果の詳細")
    print("=" * 60)

    # 入力の表示
    print("\n【入力（Step1の出力）】")
    print(f"段落数: {len(input_paragraphs)}")
    print("-" * 60)
    for i, para in enumerate(input_paragraphs):
        print(f"\n段落 {i + 1}:")
        print(para)
        print(f"（文字数: {len(para)}）")

    # 出力の表示
    print("\n" + "=" * 60)
    print("\n【出力（Step2の出力）】")
    print(f"チャンク数: {len(output_chunks)}")
    print("-" * 60)
    for i, chunk in enumerate(output_chunks):
        print(f"\nチャンク {i + 1}:")
        print(chunk)
        print(f"（文字数: {len(chunk)}）")

    # 統計情報
    print("\n" + "=" * 60)
    print("統計情報")
    print("=" * 60)
    print(f"入力段落数: {len(input_paragraphs)}")
    print(f"出力チャンク数: {len(output_chunks)}")
    print(f"変化: {len(output_chunks) - len(input_paragraphs):+d}")

    if input_paragraphs:
        avg_para_len = sum(len(p) for p in input_paragraphs) / len(input_paragraphs)
        print(f"平均段落長: {avg_para_len:.1f} 文字")

    if output_chunks:
        avg_chunk_len = sum(len(c) for c in output_chunks) / len(output_chunks)
        print(f"平均チャンク長: {avg_chunk_len:.1f} 文字")

    # 検証ポイント
    print("\n" + "=" * 60)
    print("検証ポイント")
    print("=" * 60)
    print("✓ 意味的に類似した内容が同じチャンクにまとまっているか?")
    print("✓ 話題の転換点で適切に分割されているか?")
    print("✓ 形式的な改行ではなく、意味のまとまりで分割されているか?")
    print("✓ 元のテキストが省略・要約されず保持されているか?")
    print("=" * 60)


# ================================================================
# テスト実行
# ================================================================

async def main():
    """メイン処理"""

    # APIキーの設定（環境変数から取得）
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: 環境変数 GOOGLE_API_KEY が設定されていません")
        print("以下のコマンドで設定してください:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        return

    # APIクライアント初期化
    client = SimpleAsyncAPIClient(api_key=api_key)

    # テスト用入力段落（Step1の出力を想定）
    # このデータは、話題が混在している段落を含んでいます
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

    print("テスト入力段落（Step1の出力を想定）:")
    print("=" * 60)
    for i, para in enumerate(test_paragraphs):
        print(f"\n【段落 {i + 1}】")
        print(para)
        print()
    print("=" * 60)
    print()

    print("このテストでは、以下を確認します:")
    print("1. 機械学習関連の内容が適切にまとまっているか")
    print("2. 話題転換（ラーメンの話）が別チャンクに分離されるか")
    print("3. 話題が戻った部分の処理が適切か")
    print()

    # Step2を実行
    output_chunks = await step2_semantic_chunking_test(
        paragraphs=test_paragraphs,
        client=client,
        model="gemini-2.0-flash-exp"
    )

    # 結果を表示
    display_results(test_paragraphs, output_chunks)


if __name__ == "__main__":
    asyncio.run(main())
