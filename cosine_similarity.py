# cosine_similarity.py
"""
（1）重要単語の切り出し機能：

（2）文章に分割する機能：

（3）コサイン近似度を求める機能：
"""
# !pip install google-generativeai
import os
import google.generativeai as genai
from pydantic import BaseModel

from regex_mecab import KeywordExtractor
from qa_generation.semantic import SemanticCoverage

class ChunksResult(BaseModel):
    summary: str
    sentiment: str

def main():
    """メイン実行関数"""

    # 1. 日本語サンプル
    sample_text_jp = """
    人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
    特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
    BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。
    AIの応用は医療診断から自動運転まで幅広く、社会に大きな影響を与えています。
    """

    # 2. 英語サンプル
    sample_text_en = """
    Artificial intelligence (AI) is rapidly advancing based on machine learning and deep learning.
    In the field of natural language processing (NLP) in particular, transformer models have achieved revolutionary results.
    Large language models like BERT and GPT have significantly enhanced contextual understanding capabilities.
    AI applications span widely from medical diagnosis to autonomous driving, profoundly impacting society.
    """

    #
    sample_text_recursive_character_text_splitter = """
    第1章：導入手順
        まずは電源を入れてください
        次に設定ボタンを押します。
        説明終わり
        ここで、本商品の保証期間の延長契約のお知らせです。
        購入時に申し込めば、保証2年延長が20%割引されます。

    第2章：トラブルシューティング
        電源ボタンの不具合対応について解説します。
    """

    # print('（1）重要単語の切り出し機能：')
    # extractor = KeywordExtractor()
    # keywords = extractor.extract(sample_text_jp, top_n=10)
    # keywords = extractor.extract(sample_text_en, top_n=10)


    # （2）文章に分割する機能: recursive_character_text_splitter
    # qa_generation/semantic.py の _chunk_by_paragraphs および _split_into_sentencesで分割する。
    # APIキー設定
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    chunks_model = genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config=genai.GenarationConfig(
            response_mime_type="application/json",
            response_schema=ChunksResult
        )
    )
    semantic_prompt = """
    
    """
    input_text = sample_text_recursive_character_text_splitter
    output_text = ''
    semantic = SemanticCoverage(embedding_model="gemini-embedding-001")


    # （3）チャンク分割実行：Recursive Character Text Splitter（再帰的文字分割）
    print('\nchunks分割：Recursive Character Text Splitter ---')
    chunks = semantic.create_semantic_chunks(
        document=sample_text_recursive_character_text_splitter,
        max_tokens=400,  # 最大トークン数
        min_tokens=10,  # 最小トークン数
        prefer_paragraphs=True,  # 段落優先
        overlap_tokens=0  # オーバーラップなし
    )
    import pprint
    pprint.pprint(chunks)

    """
    （4）Semantic Chunking（セマンティックチャンク）
        意味解析 (2): 構造が同じでも「意味」が離れていればそこで切り、
    """
    # semantic2 = SemanticCoverage(embedding_model="gemini-embedding-001")
    # print('\nchunks分割：Semantic Chunking ---')
    # chunks2 = semantic2.create_semantic_chunks(
    #     document=sample_text_recursive_character_text_splitter,
    #     max_tokens=400,  # 最大トークン数
    #     min_tokens=10,  # 最小トークン数
    #     prefer_paragraphs=True,  # 段落優先
    #     overlap_tokens=0  # オーバーラップなし
    # )
    # pprint.pprint(chunks2)

    # # （5）Chunk Overlap(チャンクオーバーラップ）
    # print('\n chunks分割：')
    # chunks = semantic.create_semantic_chunks(
    #     document=sample_text_recursive_character_text_splitter,
    #     max_tokens=400,  # 最大トークン数
    #     min_tokens=10,  # 最小トークン数
    #     prefer_paragraphs=True,  # 段落優先
    #     overlap_tokens=0  # オーバーラップなし
    # )
    # import pprint
    # pprint.pprint(chunks)

    # 埋め込み生成
    # embeddings = semantic.generate_embeddings(chunks)


if __name__ == '__main__':
    main()
