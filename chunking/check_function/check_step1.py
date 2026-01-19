# check_step1.py
"""
_step1_hierarchical_split 関数の動作確認プログラム

このプログラムは以下を確認します:
1. 入力テキストが正しく階層構造（段落 > 文）に分割されるか
2. 見出しと本文が分離されず1つのParagraphにまとまっているか
3. 空行や章の変わり目で適切に分割されているか
"""

import asyncio
import os
from typing import List, Any, Coroutine

# アップロードされたモジュールをインポート
from chunking.models import StructuralResult, ParagraphUnit
from chunking.prompts import PARAGRAPH_SEPARATION_PROMPT

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
# Step1 関数（元の関数から非同期処理部分を簡略化）
# ================================================================

async def step1_hierarchical_split_test(
        text: str,
        client: SimpleAsyncAPIClient,
        model: str = "gemini-2.0-flash-exp",
        block_size: int = 2000
) -> List[str]:
    """
    Step 1: 階層構造化のテスト版

    Args:
        text: 入力テキスト
        client: APIクライアント
        model: 使用するモデル
        block_size: ブロックサイズ（文字数）

    Returns:
        段落のリスト
    """
    print("=" * 60)
    print("[Step 1/3] 階層構造化（段落 > 文）")
    print("=" * 60)

    # テキストをブロックに分割
    blocks = [text[i:i + block_size] for i in range(0, len(text), block_size)]
    print(f"入力テキスト長: {len(text)} 文字")
    print(f"ブロック数: {len(blocks)}")
    print()

    # 各ブロックを処理
    paragraphs = []
    for i, block in enumerate(blocks):
        print(f"--- ブロック {i + 1}/{len(blocks)} を処理中... ---")

        # プロンプト作成
        prompt = f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{block}"

        # API呼び出し
        result_json = await client.generate_content(
            model=model,
            contents=prompt,
            response_schema=StructuralResult,
            task_id=f"step1_block_{i}"
        )

        # レスポンスをパース
        if result_json:
            try:
                result = StructuralResult.model_validate_json(result_json)

                # 各段落を抽出
                for para in result.paragraphs:
                    paragraphs.append(para.full_text)

                print(f"  ✓ {len(result.paragraphs)} 個の段落を抽出")

            except Exception as e:
                print(f"  ✗ パース失敗: {e}")
        else:
            print("  ✗ API呼び出し失敗")

        print()

    print(f"合計 {len(paragraphs)} 個の段落を抽出しました")
    print("=" * 60)

    return paragraphs


# ================================================================
# 結果表示関数
# ================================================================

def display_results(paragraphs: List[str]):
    """
    抽出された段落を見やすく表示

    Args:
        paragraphs: 段落のリスト
    """
    print("\n")
    print("=" * 60)
    print("抽出結果の詳細")
    print("=" * 60)

    for i, para in enumerate(paragraphs):
        print(f"\n【段落 {i + 1}】")
        print("-" * 60)
        print(para)
        print("-" * 60)
        print(f"文字数: {len(para)}")
        print(f"行数: {para.count(chr(10)) + 1}")  # 改行の数 + 1

    print("\n")
    print("=" * 60)
    print("検証ポイント")
    print("=" * 60)
    print("✓ 見出し（第X章）と本文が分離されず1つの段落になっているか?")
    print("✓ 空行や章の変わり目で段落が分割されているか?")
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

    # テスト用入力テキスト
    test_text = """第1章 人工知能の基礎
人工知能（AI）は、コンピュータに人間のような知能を持たせる技術です。
機械学習やディープラーニングがその中核をなしています。

第2章 機械学習の手法
教師あり学習では、ラベル付きデータから学習します。
代表的な手法には、ランダムフォレストやサポートベクターマシンがあります。

ところで、昨日食べたラーメンが美味しかったです。
次回も同じ店に行きたいと思います。"""

    print("テスト入力テキスト:")
    print("=" * 60)
    print(test_text)
    print("=" * 60)
    print()

    # Step1を実行
    paragraphs = await step1_hierarchical_split_test(
        text=test_text,
        client=client,
        model="gemini-2.0-flash",
        block_size=2000
    )

    # 結果を表示
    display_results(paragraphs)


if __name__ == "__main__":
    asyncio.run(main())
