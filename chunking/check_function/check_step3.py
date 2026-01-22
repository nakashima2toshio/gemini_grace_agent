# check_step3.py
"""
Step3: 文脈連続性チェックの同期版・簡易確認プログラム

【目的】
隣接するチャンク間の文脈連続性を判定し、
連続している場合は結合、非連続の場合は分離する。

【処理の流れ】
1. Step2の出力（チャンクリスト）を入力として受け取る
2. 隣接ペアごとにGemini APIで連続性を判定
3. 判定結果に基づいてチャンクを結合/分離
4. 最終的なチャンクリストを返す
"""

import os
from google import genai
from google.genai import types

# chunking モジュールからインポート
from chunking.models import ContinuityResult
from chunking.prompts import CONTINUITY_CHECK_PROMPT


def step3_continuity_check(chunks: list[str], api_key: str) -> list[str]:
    """
    隣接チャンク間の連続性をチェックし結合/分離する（Step3のコア機能）

    Args:
        chunks: チャンクのリスト（Step2の出力）
        api_key: Gemini API キー

    Returns:
        連続性に基づいて結合/分離された最終チャンクリスト
    """
    client = genai.Client(api_key=api_key)

    print(f"入力: {len(chunks)}チャンク")

    if len(chunks) <= 1:
        print("チャンクが1つ以下のため、そのまま返します")
        return chunks

    # 隣接ペアの連続性を判定
    continuity_flags = []

    for i in range(len(chunks) - 1):
        print(f"ペア {i + 1}/{len(chunks) - 1} 判定中...")

        # プロンプト作成
        prompt = f"{CONTINUITY_CHECK_PROMPT}\n\n【前のテキスト】\n{chunks[i]}\n\n【次のテキスト】\n{chunks[i + 1]}"

        # Gemini API 呼び出し（同期）
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ContinuityResult
            )
        )

        # レスポンスをパース
        result = ContinuityResult.model_validate_json(response.text)
        continuity_flags.append(result.is_connected)

        status = "連続 → 結合" if result.is_connected else "非連続 → 分離"
        print(f"  → {status}")

    # マージ処理
    print()
    print("マージ処理...")
    final_chunks = [chunks[0]]

    for i, is_connected in enumerate(continuity_flags):
        if is_connected:
            # 結合
            final_chunks[-1] += "\n\n" + chunks[i + 1]
            print(f"  チャンク{i} + チャンク{i + 1} → 結合")
        else:
            # 分離
            final_chunks.append(chunks[i + 1])
            print(f"  チャンク{i + 1} → 新規追加")

    return final_chunks


def main():
    """メイン処理"""

    # APIキー取得
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY 環境変数を設定してください")
        print("  export GOOGLE_API_KEY='your-api-key'")
        return

    # テスト用チャンク（Step2の出力を想定）
    # 連続性判定をテストするため、様々なパターンを含む
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

    print("=" * 50)
    print("【入力チャンク（Step2の出力）】")
    print("=" * 50)
    for i, chunk in enumerate(test_chunks, 1):
        print(f"\n--- チャンク{i} ---")
        print(chunk)
    print()

    print("【期待される判定】")
    print("  ペア1 (機械学習基礎 vs 応用): 連続 → 結合")
    print("  ペア2 (機械学習 vs ラーメン): 非連続 → 分離")
    print("  ペア3 (ラーメン vs ラーメン店): 連続 → 結合")
    print("  ペア4 (ラーメン店 vs 深層学習): 非連続 → 分離")
    print()

    # Step3 実行
    print("=" * 50)
    print("【Step3 実行】")
    print("=" * 50)
    final_chunks = step3_continuity_check(test_chunks, api_key)

    # 結果表示
    print()
    print("=" * 50)
    print(f"【結果】{len(test_chunks)}チャンク → {len(final_chunks)}チャンク")
    print("=" * 50)
    for i, chunk in enumerate(final_chunks, 1):
        print(f"\n--- 最終チャンク{i} ({len(chunk)}文字) ---")
        print(chunk)

    # 検証ポイント
    print()
    print("=" * 50)
    print("【検証ポイント】")
    print("=" * 50)
    print("✓ 同じトピックのチャンクが結合されているか")
    print("✓ 話題転換箇所で分離されているか")
    print("✓ 結合後のテキストが正しく保持されているか")


if __name__ == "__main__":
    main()
