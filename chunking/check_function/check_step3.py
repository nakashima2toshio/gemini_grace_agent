# check_step3.py
"""
_step3_continuity_check 関数の動作確認プログラム

このプログラムは以下を確認します:
1. 隣接するチャンク間の文脈連続性を正しく判定できるか
2. 連続している場合に適切に結合されるか
3. 話題が転換している場合に適切に分離されるか
4. マージ処理が正しく動作するか

実行方法:
  プロジェクトルートから: python -m chunking.check_function.check_step3
"""

import asyncio
import os
from typing import List, Tuple

# 絶対インポート（推奨）
from chunking.models import ContinuityResult
from chunking.prompts import CONTINUITY_CHECK_PROMPT

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
    ) -> str:
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
# Step3 関数（元の関数から非同期処理部分を簡略化）
# ================================================================

async def step3_continuity_check_test(
        chunks: List[str],
        client: SimpleAsyncAPIClient,
        model: str = "gemini-2.0-flash-exp"
) -> Tuple[List[str], List[dict]]:
    """
    Step 3: 文脈連続性チェックのテスト版

    Args:
        chunks: チャンクのリスト（Step2の出力）
        client: APIクライアント
        model: 使用するモデル

    Returns:
        (最終チャンクリスト, 判定詳細リスト)
    """
    print("=" * 60)
    print("[Step 3/3] 文脈連続性チェック")
    print("=" * 60)
    print(f"入力: {len(chunks)} チャンク")
    print()

    if len(chunks) <= 1:
        print("チャンクが1つ以下のため、チェックをスキップします")
        return chunks, []

    # 隣接ペアの判定結果を保存
    continuity_results = []

    # 各隣接ペアを処理
    for i in range(len(chunks) - 1):
        print(f"--- ペア {i + 1}/{len(chunks) - 1} を処理中... ---")
        print(f"前のチャンク（最初の80文字）: {chunks[i][:80]}...")
        print(f"次のチャンク（最初の80文字）: {chunks[i + 1][:80]}...")
        print()

        # プロンプト作成
        prompt = f"{CONTINUITY_CHECK_PROMPT}\n\n【前のテキスト】\n{chunks[i]}\n\n【次のテキスト】\n{chunks[i + 1]}"

        # API呼び出し
        result_json = await client.generate_content(
            model=model,
            contents=prompt,
            response_schema=ContinuityResult,
            task_id=f"step3_pair_{i}"
        )

        # レスポンスをパース
        if result_json:
            try:
                result = ContinuityResult.model_validate_json(result_json)
                is_connected = result.is_connected

                continuity_results.append({
                    'pair_index'    : i,
                    'chunk1_preview': chunks[i][:100],
                    'chunk2_preview': chunks[i + 1][:100],
                    'is_connected'  : is_connected
                })

                if is_connected:
                    print(f"  ✓ 判定: 連続している (True) → 結合されます")
                else:
                    print(f"  ✗ 判定: 連続していない (False) → 分離されます")

            except Exception as e:
                print(f"  ✗ パース失敗: {e}")
                continuity_results.append({
                    'pair_index'    : i,
                    'chunk1_preview': chunks[i][:100],
                    'chunk2_preview': chunks[i + 1][:100],
                    'is_connected'  : False,
                    'error'         : str(e)
                })
        else:
            print("  ✗ API呼び出し失敗")
            continuity_results.append({
                'pair_index'    : i,
                'chunk1_preview': chunks[i][:100],
                'chunk2_preview': chunks[i + 1][:100],
                'is_connected'  : False,
                'error'         : 'API call failed'
            })

        print()

    # マージ処理
    print("=" * 60)
    print("マージ処理を実行中...")
    print("=" * 60)

    final_chunks = [chunks[0]]
    merge_log = []

    for i, result_info in enumerate(continuity_results):
        is_connected = result_info.get('is_connected', False)

        if is_connected:
            # 結合
            before_merge = final_chunks[-1][:50]
            final_chunks[-1] += "\n\n" + chunks[i + 1]
            after_merge = final_chunks[-1][:50]

            merge_log.append({
                'action'    : 'merge',
                'pair_index': i,
                'result'    : f"チャンク{i}とチャンク{i + 1}を結合"
            })
            print(f"  [結合] チャンク{i} + チャンク{i + 1} → 1つのチャンクに")
        else:
            # 分離
            final_chunks.append(chunks[i + 1])
            merge_log.append({
                'action'    : 'separate',
                'pair_index': i,
                'result'    : f"チャンク{i + 1}を新しいチャンクとして追加"
            })
            print(f"  [分離] チャンク{i + 1} → 新しいチャンクとして追加")

    print()
    print(f"合計 {len(final_chunks)} 個のチャンクになりました（{len(chunks)} → {len(final_chunks)}）")
    print("=" * 60)

    return final_chunks, continuity_results


# ================================================================
# 結果表示関数
# ================================================================

def display_results(
        input_chunks: List[str],
        output_chunks: List[str],
        continuity_results: List[dict]
):
    """
    処理前後の比較と判定詳細を見やすく表示

    Args:
        input_chunks: 入力チャンクのリスト
        output_chunks: 出力チャンクのリスト
        continuity_results: 連続性判定の詳細リスト
    """
    print("\n")
    print("=" * 60)
    print("処理結果の詳細")
    print("=" * 60)

    # 入力の表示
    print("\n【入力（Step2の出力）】")
    print(f"チャンク数: {len(input_chunks)}")
    print("-" * 60)
    for i, chunk in enumerate(input_chunks):
        preview = chunk.replace('\n', ' ')[:100]
        print(f"\nチャンク {i + 1}: {preview}...")
        print(f"（文字数: {len(chunk)}）")

    # 連続性判定の詳細
    print("\n" + "=" * 60)
    print("\n【連続性判定の詳細】")
    print("-" * 60)
    for result in continuity_results:
        pair_idx = result['pair_index']
        is_connected = result.get('is_connected', False)
        status = "✓ 連続" if is_connected else "✗ 非連続"
        action = "→ 結合" if is_connected else "→ 分離"

        print(f"\nペア {pair_idx + 1}: チャンク{pair_idx} と チャンク{pair_idx + 1}")
        print(f"  判定: {status} {action}")

        if 'error' in result:
            print(f"  エラー: {result['error']}")

    # 出力の表示
    print("\n" + "=" * 60)
    print("\n【出力（Step3の出力）】")
    print(f"チャンク数: {len(output_chunks)}")
    print("-" * 60)
    for i, chunk in enumerate(output_chunks):
        preview = chunk.replace('\n', ' ')[:100]
        print(f"\nチャンク {i + 1}: {preview}...")
        print(f"（文字数: {len(chunk)}）")

    # 統計情報
    print("\n" + "=" * 60)
    print("統計情報")
    print("=" * 60)
    print(f"入力チャンク数: {len(input_chunks)}")
    print(f"出力チャンク数: {len(output_chunks)}")
    print(f"変化: {len(output_chunks) - len(input_chunks):+d}")
    print(f"結合された回数: {len(input_chunks) - len(output_chunks)}")

    # 連続性判定の統計
    if continuity_results:
        connected_count = sum(1 for r in continuity_results if r.get('is_connected', False))
        separated_count = len(continuity_results) - connected_count
        print(f"連続と判定: {connected_count}/{len(continuity_results)}")
        print(f"非連続と判定: {separated_count}/{len(continuity_results)}")

    if input_chunks:
        avg_input_len = sum(len(c) for c in input_chunks) / len(input_chunks)
        print(f"平均入力チャンク長: {avg_input_len:.1f} 文字")

    if output_chunks:
        avg_output_len = sum(len(c) for c in output_chunks) / len(output_chunks)
        print(f"平均出力チャンク長: {avg_output_len:.1f} 文字")

    # 検証ポイント
    print("\n" + "=" * 60)
    print("検証ポイント")
    print("=" * 60)
    print("✓ 同じトピックのチャンクが適切に結合されているか?")
    print("✓ 話題が転換している箇所で適切に分離されているか?")
    print("✓ 章の変わり目で適切に分離されているか?")
    print("✓ マージされたチャンクの内容が正しく保持されているか?")
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

    # テスト用入力チャンク（Step2の出力を想定）
    # 連続性判定をテストするために、様々なパターンを含む
    test_chunks = [
        # チャンク1: 機械学習の基礎
        """機械学習は、データからパターンを学習する技術です。
教師あり学習、教師なし学習、強化学習の3つに大別されます。
これらの手法は様々な分野で応用されています。""",

        # チャンク2: 機械学習の応用（連続している）
        """機械学習の応用例としては、画像認識や自然言語処理があります。
特に深層学習の登場により、精度が飛躍的に向上しました。
医療診断や自動運転などの分野でも活用されています。""",

        # チャンク3: ラーメンの話（話題転換）
        """ところで、昨日食べたラーメンが非常に美味しかったです。
醤油ベースのスープに、細麺が絶妙にマッチしていました。
チャーシューも柔らかく、また行きたいと思います。""",

        # チャンク4: ラーメン店の情報（連続している）
        """そのラーメン店は駅から徒歩5分の場所にあります。
営業時間は11時から22時までで、定休日は水曜日です。
次回は友人を誘って行こうと考えています。""",

        # チャンク5: 深層学習（話題転換）
        """深層学習は、多層のニューラルネットワークを用いる手法です。
畳み込みニューラルネットワーク（CNN）やリカレントニューラルネットワーク（RNN）が代表的です。
大量のデータと計算資源が必要ですが、高い性能を発揮します。"""
    ]

    print("テスト入力チャンク（Step2の出力を想定）:")
    print("=" * 60)
    for i, chunk in enumerate(test_chunks):
        print(f"\n【チャンク {i + 1}】")
        print(chunk)
        print()
    print("=" * 60)
    print()

    print("このテストでは、以下の判定を確認します:")
    print("  ペア1 (機械学習基礎 vs 機械学習応用): 連続 → 結合")
    print("  ペア2 (機械学習応用 vs ラーメン): 非連続 → 分離")
    print("  ペア3 (ラーメン vs ラーメン店): 連続 → 結合")
    print("  ペア4 (ラーメン店 vs 深層学習): 非連続 → 分離")
    print()

    # Step3を実行
    output_chunks, continuity_results = await step3_continuity_check_test(
        chunks=test_chunks,
        client=client,
        model="gemini-2.0-flash-exp"
    )

    # 結果を表示
    display_results(test_chunks, output_chunks, continuity_results)


if __name__ == "__main__":
    asyncio.run(main())
