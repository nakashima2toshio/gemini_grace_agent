# code_check/check_mecab.py
import sys
import os

# プロジェクトルートをパスに追加して regex_mecab をインポートできるようにする
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from regex_mecab import KeywordExtractor

def main():
    """
    MeCab (または正規表現フォールバック) を使用して
    入力文字列から重要単語を抽出する確認プログラム
    """
    
    # 1. 入力文字列
    input_string = '浦沢直樹が受賞した作品名は？　また、何の賞ですか？'
    print(f"入力文字列: {input_string}")
    print("-" * 40)

    # 2. 処理: 重要単語を取り出す
    # KeywordExtractorのインスタンス化 (デフォルトでMeCab優先)
    extractor = KeywordExtractor(prefer_mecab=True)
    
    # extractメソッドでキーワード抽出 (上位5件、スコアリングあり)
    # top_nは必要に応じて調整可能ですが、ここではデフォルトに近い形にします
    keywords = extractor.extract(input_string, top_n=5, use_scoring=True)

    # 3. 出力: 重要単語を表示する
    print("抽出された重要単語:")
    for i, word in enumerate(keywords, 1):
        print(f"{i}. {word}")

if __name__ == "__main__":
    main()
