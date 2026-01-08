# csv_to_chunks_text.py
import os
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# --- 1. Pydanticモデル定義 ---

class SentenceUnit(BaseModel):
    text: str = Field(description="1つの文、または意味の最小単位")


class ParagraphUnit(BaseModel):
    id: int = Field(description="Paragraph ID")
    sentences: List[SentenceUnit] = Field(description="この段落に含まれる文のリスト")

    @property
    def full_text(self) -> str:
        return "".join([s.text for s in self.sentences])


class StructuralResult(BaseModel):
    paragraphs: List[ParagraphUnit]


# 文脈連続性判定用のモデル
class ContinuityResult(BaseModel):
    is_connected: bool = Field(
        description="前のテキストと次のテキストが、意味的に連続している（同じトピックである）場合はTrue、話題が転換している場合はFalse")


# --- 2. グローバル設定 (クライアントとプロンプト) ---

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY not set.")
    client = None
else:
    client = genai.Client(api_key=api_key)

# プロンプト 1: 階層分割
paragraph_separation_prompt = """
あなたはテキスト構造化エンジンです。入力されたテキストを以下の【分割ルール】に従って解析し、階層構造（段落 > 文）に変換してください。

【分割ルール】
入力されたテキストを、以下のルールに従って構造化してください。
目的は、テキストを「大きな意味のブロック（Paragraph）」に分け、その中を「文（Sentence）」に分解することです。

【Rule 1: Paragraphの分割（最優先）】
- **見出しと本文を分離しないこと**。
- 「第〇章」や「見出し」がある場合、それ単体でParagraphを作らず、**直後の本文も含めて1つのParagraph**としてまとめてください。
- Paragraphを分ける基準は、原則として「空行（\\n\\n）」や「章の変わり目」のみです。

【Rule 2: Sentenceの分割】
- Paragraphの中身を、句点「。」や改行ごとに区切って sentences リストに格納してください。
- 見出し部分も1つの sentence として扱ってください。

【出力要件】
- JSONスキーマに従い、paragraphs リストの中に sentences リストを持つ構造で出力すること。
- 元のテキストの内容を省略したり要約したりせず、**そのままの文字列**を保持すること。
"""

# プロンプト 2: 意味的分割
semantic_chunking_prompt = """
あなたは「セマンティック・チャンキング（意味的分割）エンジン」です。
入力されたテキストを、形式的な段落や改行ではなく、「意味のまとまり（トピック）」に基づいて再構成してください。

【処理ロジック: 仮想的なベクトル類似度判定】
1. テキストを文脈に沿って読み進め、隣り合う文同士の「意味的な距離」を分析してください。
2. 文の内容が連続している、または高い関連性を持つ場合は、同じブロック（Paragraph）に結合してください。
3. **「話題の転換点」**（意味の類似度がしきい値を下回るような、話題の切り替わり）を見つけたら、そこでブロックを分割してください。

【分割の基準】
- **文字数や物理的な改行（\\n）は無視すること**。
- たとえ改行がなくても、話題が大きく変われば分割する。
- たとえ改行があっても、文脈や意味が続いているなら分割しない。

【出力要件】
- 意味的に凝集したブロックを1つの Paragraph と定義し、その中の文を sentences リストに格納して出力すること。
- 元のテキストを一言一句変更せず保持すること。
"""

# 【追加】プロンプト 3: 文脈連続性チェック
continuity_check_prompt = """
あなたは「文脈判定エンジン」です。
提示された「前のテキスト(Prev)」と「次のテキスト(Next)」を読み、
これらが**「一つの連続した話題（トピック）」**としてつながっているかを判定してください。

【判定基準】
- **False (切断すべき)**:
    - 章が変わった（例：「第1章」から「第2章」へ）。
    - 全く別の話題、製品、カテゴリの話に切り替わった。
    - 前の文が「完結」しており、次の文から新しいセクションが始まっている。
- **True (接続すべき)**:
    - 文脈が連続しており、前の文の情報を知らないと次の文が理解しにくい。
    - 同じトピックの説明が続いている。

判定結果（is_connected）のみをJSONで返してください。
"""


# --- 3. 個別機能関数 ---

def recursive_character_text_splitter(
        text: str,
        model: str = "gemini-2.0-flash",
        prompt: str = paragraph_separation_prompt
) -> List[str]:
    """Geminiを使用してテキストを意味的なパラグラフに分割し、リストとして返す関数。"""
    if not client: return []
    paragraph_list = []
    try:
        response = client.models.generate_content(
            model=model,
            contents=f"{prompt}\n\n【対象テキスト】\n{text}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuralResult,
            ),
        )
        if response.text:
            result = StructuralResult.model_validate_json(response.text)
            if result.paragraphs:
                for p in result.paragraphs:
                    clean_text = p.full_text.replace('\n', ' ')
                    paragraph_list.append(clean_text)
    except Exception as e:
        print(f"Error in recursive_character_text_splitter: {e}")
        return []
    return paragraph_list


def semantic_chunking(
        text: str,
        model: str = "gemini-2.0-flash",
        prompt: str = semantic_chunking_prompt
) -> List[str]:
    """Geminiを使用して、意味のまとまり（トピック）ごとにテキストを分割する関数。"""
    if not client: return []
    chunk_list = []
    try:
        response = client.models.generate_content(
            model=model,
            contents=f"{prompt}\n\n【対象テキスト】\n{text}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuralResult,
            ),
        )
        if response.text:
            result = StructuralResult.model_validate_json(response.text)
            if result.paragraphs:
                for p in result.paragraphs:
                    clean_text = p.full_text.replace('\n', ' ')

                    # 【改修】末尾に改行がない場合は付与する
                    if not clean_text.endswith('\n'):
                        clean_text += '\n'

                    chunk_list.append(clean_text)
    except Exception as e:
        print(f"Error in semantic_chunking: {e}")
        return []
    return chunk_list


# 連続性判定関数
def check_continuity(prev_text: str, next_text: str, model: str) -> bool:
    """Geminiを使って2つのテキストの意味的連続性を判定する"""
    if not client: return False
    try:
        response = client.models.generate_content(
            model=model,
            contents=f"{continuity_check_prompt}\n\n【前のテキスト】\n{prev_text}\n\n【次のテキスト】\n{next_text}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ContinuityResult,
            ),
        )
        if response.text:
            result = ContinuityResult.model_validate_json(response.text)
            return result.is_connected
    except Exception as e:
        print(f"Error in continuity check: {e}")
        return False
    return False


def chunk_overlap(paragraphs: List[str], model: str = "gemini-2.0-flash") -> List[str]:
    """
    隣り合うパラグラフ間で、意味が連続している場合のみオーバーラップさせます。
    Args:
        paragraphs (List[str]): 分割済みのパラグラフリスト
        model (str): Geminiモデル
    """
    if not paragraphs: return []

    overlapped_result = []

    print("Checking continuity for overlaps...")
    for i, current_text in enumerate(paragraphs):
        if i == 0:
            overlapped_result.append(current_text)
            continue

        prev_text = paragraphs[i - 1]
        print(f"{i} ", end="")

        # --- 【改修】 AIによる意味的接続チェック ---
        is_connected = check_continuity(prev_text, current_text, model)

        if not is_connected:
            # 意味がつながっていない（章が変わった等）場合、オーバーラップさせない
            # 必要であればデバッグ用に以下を有効化
            # print(f"  [Split Detected] ID:{i+1} starts a new topic. No overlap.")
            overlapped_result.append(current_text)
            continue

        # --- 接続判定(True)の場合のみオーバーラップ処理を実行 ---

        # 正規表現: 句点(。)、ピリオド(.)、感嘆符(!)、疑問符(?) の後ろで分割
        sentences = re.split(r'(?<=[。．！!？?])', prev_text)
        sentences = [s for s in sentences if s.strip()]

        if sentences:
            overlap_part = sentences[-1]  # 最後の1文を取得
        else:
            overlap_part = prev_text

        combined_text = overlap_part + current_text
        overlapped_result.append(combined_text)

    return overlapped_result


def show_paragraphs(paragraphs: List[str], title: Optional[str] = None) -> None:
    """分割されたパラグラフのリストを整形して標準出力に表示します。"""
    if title:
        print(f"--- {title} ---")

    if paragraphs:
        for i, p_text in enumerate(paragraphs):
            # 表示が見やすいように改行を除去して短縮表示
            display_text = p_text.replace('\n', '')
            # 必要に応じて長すぎる場合はカットしても良い
            # if len(display_text) > 100: display_text = display_text[:100] + "..."
            print(f"Chunk [ID:{i + 1}]: {display_text}")
    else:
        print("Failed to split text or empty.")
    print("")


# --- 大規模ファイル処理用クラス ---
class LargeTextProcessor:
    def __init__(self, block_size: int = 2000):
        """
        Args:
            block_size: 1回のAPI処理の文字数。安全のため 2000 を推奨。
        """
        self.block_size = block_size

    def split_into_batches(self, text: str) -> List[str]:
        """
        テキストを block_size 以下になるように確実に分割する。
        改行がない長文も強制的に分割する。
        """
        # 1. 改行コードを統一（\r\n, \r -> \n）
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        batches = []
        current_batch = []
        current_length = 0

        # まず行単位で分ける
        raw_lines = text.split('\n')

        for raw_line in raw_lines:
            # 行自体が block_size を超えている場合（ここが以前のバグの原因）
            # 強制的に文字数でぶった切る
            if len(raw_line) > self.block_size:
                # 現在のバッファがあれば先に吐き出す
                if current_batch:
                    batches.append("\n".join(current_batch))
                    current_batch = []
                    current_length = 0

                # 長い行を block_size ずつスライスして追加
                for i in range(0, len(raw_line), self.block_size):
                    chunk = raw_line[i: i + self.block_size]
                    batches.append(chunk)
                continue

            # --- 通常の積み上げ処理 ---
            # 改行分(+1)を考慮
            line_len = len(raw_line) + 1

            if current_length + line_len > self.block_size:
                if current_batch:
                    batches.append("\n".join(current_batch))
                current_batch = [raw_line]
                current_length = line_len
            else:
                current_batch.append(raw_line)
                current_length += line_len

        # 残りを追加
        if current_batch:
            batches.append("\n".join(current_batch))

        return batches

    def process(self, text: str, model: str = "gemini-2.0-flash") -> List[str]:
        # 分割実行
        batches = self.split_into_batches(text)
        print(f"Total Batches: {len(batches)}")

        all_semantic_chunks = []

        for i, batch_text in enumerate(batches):
            if not batch_text.strip():
                continue

            # ログで文字数を厳重確認
            batch_len = len(batch_text)
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] >> Processing Batch {i + 1}/{len(batches)} ({batch_len} chars)...")

            # 安全装置: まかり間違って巨大なままならスキップ（エラー回避）
            if batch_len > 10000:
                print(f"  [WARNING] Batch size {batch_len} is too large! Skipping to prevent API error.")
                continue

            # Step 1: 階層分割
            step1_paragraphs = recursive_character_text_splitter(batch_text, model)

            # Step 2: 意味解析
            step2_chunks = []
            if step1_paragraphs:
                for p in step1_paragraphs:
                    sub_chunks = semantic_chunking(p, model)
                    step2_chunks.extend(sub_chunks)
            else:
                step2_chunks = [batch_text]

            all_semantic_chunks.extend(step2_chunks)

        print(f"\nStep 1 & 2 Completed. Total Semantic Chunks: {len(all_semantic_chunks)}")

        # Step 3
        print("Starting Step 3: Global Chunk Overlap check...")
        final_chunks = chunk_overlap(all_semantic_chunks, model=model)

        return final_chunks


""" --- 4. 一気通貫処理関数 (Wrapper) --- """
def chunks_all(text: str, model: str = "gemini-2.0-flash") -> List[str]:
    """
    テキスト処理パイプラインを一気通貫で実行します。
    1. 階層分割: recursive_character_text_splitter
    2. 意味解析: semantic_chunking
    3. 重複付与: chunk_overlap (with AI Check)
    """
    processor = LargeTextProcessor(block_size=2000)
    return processor.process(text, model)

# --- 5. メイン処理 ---

def main():
    # テスト用テキスト（拡張版）
    sample_text = """
    第1章：導入手順
    まずは電源を入れてください。
    電源ボタンは本体側面にあり、3秒以上長押しすると青いランプが点灯します。
    ランプが点灯しない場合は、充電ケーブルが正しく接続されているか確認してください。
    次に設定ボタンを押します。
    設定画面が表示されたら、言語設定で「日本語」を選択してください。
    これで初期セットアップは完了です。
    今月だけの限定サービスのお知らせです。
    本商品の保証期間のお得な延長ができます。
    購入時に申し込めば、保証2年延長が20%割引されます。
    このキャンペーンは今月末までの期間限定となります。
    第2章：トラブルシューティング
    電源ボタンの不具合対応について解説します。
    もしボタンを押しても反応がない場合、まずは強制再起動を試してください。
    それでも改善しない場合は、サポートセンターへお問い合わせください。
    さらに今月の最新情報です。
    Appleは2027年に新型のiPhoneを発売しました。
    同社はさらに独自のAIチップも発表しています。
    """

    # 日本語サンプル
    sample_text_jp = """
    人工知能（AI）は、機械学習を基盤として急速に発展しています。
    特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが画期的な成果を上げました。
    BERTやGPTのような大規模言語モデルは、文脈理解能力を大幅に向上させています。
    AIの応用は医療診断から自動運転まで幅広く、社会に大きな影響を与えています。
    """

    # 英語サンプル
    sample_text_en = """
    Artificial intelligence (AI) is rapidly advancing based on machine learning.
    In the field of natural language processing (NLP), transformer models have achieved results.
    Large language models like BERT and GPT have enhanced contextual understanding.
    AI applications span widely from medical diagnosis to autonomous driving.
    """

    model = "gemini-2.0-flash"
    # Largeファイル名を指定
    file_path = "./OUTPUT/wikipedia_ja_20251130_041304.txt"

    print('個別ステップの確認（デバッグ用）==================')
    # （1） 段落分割 Check
    paragraphs1 = recursive_character_text_splitter(sample_text, model)
    show_paragraphs(paragraphs1, title='（1） 段落分割 Check')

    #（2） 意味分割 Check
    semantic_results_list = []
    for p_text in paragraphs1:
        sub_chunks = semantic_chunking(p_text, model)
        semantic_results_list.extend(sub_chunks)
    show_paragraphs(semantic_results_list, title='（2） 意味分割 Check')

    # （3） 重複付与 (Smart Overlap) Check
    # ここでも model を渡す
    final_chunks = chunk_overlap(semantic_results_list, model=model)
    show_paragraphs(final_chunks, title='（3）重複付与 (Smart Overlap) Check')

    # 一気通貫関数の実行
    print("\n========== 一気通貫処理テスト ==========")

    # 日本語テキスト
    result_jp = chunks_all(sample_text, model)
    show_paragraphs(result_jp)

    # 長文テキストのテスト： chunks_all する
    print(f"長文テスト:{file_path} chunks_all --> class LargeTextProcessor")
    # ※前回の処理で作成されたファイル名を指定
    file_path = "./OUTPUT/wikipedia_ja_5per.txt"

    print(f"Reading file: {file_path} ...")

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                long_text = f.read()
            print(f"File loaded successfully. Length: {len(long_text)} chars.")

            # プロセッサの初期化 (2000文字で分割)
            processor = LargeTextProcessor(block_size=2000)

            # 実行
            final_result = processor.process(long_text, model=model)

            # 結果表示
            print("\n========== FINAL RESULT (from File) ==========")
            print(f"Total Chunks: {len(final_result)}")

            # 先頭のいくつかを表示
            show_paragraphs(final_result[:10], title="First 5 Chunks")

        except Exception as e:
            print(f"Error processing file: {e}")
    else:
        print(f"Error: File '{file_path}' not found. Please verify the file generation step.")

        # ファイルがない場合のテスト用（短いテキスト）
        print("Running test with sample text instead...")
        sample_text = """
            第1章：テスト
            これはテストです。ファイルが見つかりませんでした。
            """
        res = chunks_all(sample_text, model)
        show_paragraphs(res)


if __name__ == '__main__':
    main()

