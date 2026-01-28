# step1.py
"""
Step1: 階層構造化（段落分割）の同期版・簡易確認プログラム

【目的】
テキストを段落単位に分割する。
見出し（第X章など）と本文は分離せず、1つの段落としてまとめる。

【処理の流れ】
1. 入力テキストをブロック（2000文字単位）に分割
2. 各ブロックをGemini APIに送信し、段落構造を抽出
3. 抽出された段落のリストを返す

【Step2・Step3との連携】
このStep1の出力は、Step2の入力として使用される。
以下のパターンを検証できるよう設計：
- 前方依存: 「この」「それ」等の指示語で前を参照
- 後方依存: 専門用語が未定義のまま使用される
- 独立判定: 話題は同じでも単独で理解可能
- 章構造: 章が変わった場合の独立性
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
    # print(test_text)
    print(test_text2)
    print()

    print("【期待される分割結果】")
    print("  入力テキストを空行（\\n\\n）で5段落に分割")
    print()
    print("  段落1: RAGの説明")
    print("    - RAGの定義（検索と生成の組み合わせ、2020年発表）")
    print("    - RAGの利点（「この手法」「それ」で前を参照）")
    print("         ※ Step2で2チャンクに分割 → Step3で前方依存判定")
    print()
    print("  段落2: セマンティックチャンキングの説明")
    print("    - 用語定義（チャンク、埋め込みの説明）")
    print("    - 用語使用（チャンクサイズ、埋め込みの品質）")
    print("         ※ Step2で2チャンクに分割 → Step3で後方依存判定")
    print()
    print("  段落3: 観光情報")
    print("    - 京都観光（紅葉、清水寺、嵐山）")
    print("    - 沖縄観光（海、シュノーケリング、恩納村）")
    print("         ※ Step2で2チャンクに分割 → Step3で独立判定")
    print()
    print("  段落4: ベクトルDBの説明")
    print("    - ベクトルDBの定義（Pinecone等、ANNアルゴリズム）")
    print("    - ベクトルDBの活用（ANNのトレードオフ、HNSW/IVF）")
    print("         ※ Step2で2チャンクに分割 → Step3で後方依存判定")
    print()
    print("  段落5: 章構造")
    print("    - 第1章 機械学習入門")
    print("    - 第2章 深層学習の基礎")
    print("         ※ Step2で2チャンクに分割 → Step3で章構造による独立判定")
    print()
    print("【期待される最終結果】1テキスト → 5段落")
    print()

    # Step1 実行
    print("=" * 50)
    print("【Step1 実行】")
    print("=" * 50)
    # paragraphs = step1_hierarchical_split(test_text, api_key)
    paragraphs = step1_hierarchical_split(test_text2, api_key)

    # 結果表示
    print()
    print("=" * 50)
    print(f"【結果】{len(paragraphs)}個の段落")
    print("=" * 50)
    for i, para in enumerate(paragraphs, 1):
        print(f"\n--- 段落{i} ({len(para)}文字) ---")
        print(para)

    # 結果検証
    print()
    print("=" * 50)
    print("【結果検証】")
    print("=" * 50)
    expected_paragraphs = 5
    if len(paragraphs) == expected_paragraphs:
        print(f"✅ 期待通り {expected_paragraphs} 段落に分割されました")
    else:
        print(f"⚠️  期待: {expected_paragraphs} 段落, 実際: {len(paragraphs)} 段落")
        print("   分割が期待と異なる場合、プロンプトの調整が必要な可能性があります")

    # 検証ポイント
    print()
    print("=" * 50)
    print("【検証ポイント】")
    print("=" * 50)
    print("✓ 空行（\\n\\n）で段落が分割されているか")
    print("✓ 見出し（第X章）と本文が1つの段落にまとまっているか")
    print("✓ テキストが省略されず保持されているか")

    # Step2・Step3との連携情報
    print()
    print("=" * 50)
    print("【Step2・Step3との連携】")
    print("=" * 50)
    print("Step1の出力がStep2の入力となり、以下の分割を検証:")
    print()
    print("  段落1 → チャンク1（RAG定義）+ チャンク2（RAG利点）")
    print("  段落2 → チャンク3（用語定義）+ チャンク4（用語使用）")
    print("  段落3 → チャンク5（京都観光）+ チャンク6（沖縄観光）")
    print("  段落4 → チャンク7（ベクトルDB定義）+ チャンク8（活用）")
    print("  段落5 → チャンク9（第1章）+ チャンク10（第2章）")
    print()
    print("Step2実行後の期待結果: 5段落 → 10チャンク")
    print()
    print("Step3実行後の期待結果: 10チャンク → 7チャンク")
    print("  最終チャンク1: チャンク1+2（前方依存で結合）")
    print("  最終チャンク2: チャンク3+4（後方依存で結合）")
    print("  最終チャンク3: チャンク5（独立）")
    print("  最終チャンク4: チャンク6（独立）")
    print("  最終チャンク5: チャンク7+8（後方依存で結合）")
    print("  最終チャンク6: チャンク9（独立）")
    print("  最終チャンク7: チャンク10（独立）")


if __name__ == "__main__":
    main()
