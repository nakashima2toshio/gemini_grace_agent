"""
python a20_output_qa_csv.py

各Q&A生成プログラムの最新出力CSVファイルからquestionとanswerの列のみを抽出し、
統一フォーマットのCSVファイルを作成する。

入力ファイルパターン（最新のファイルを自動選択）:
- qa_output/a02/qa_pairs_cc_news_*.csv
- qa_output/a02/qa_pairs_livedoor_*.csv
- qa_output/a03/qa_pairs_cc_news_*.csv
- qa_output/a03/qa_pairs_livedoor_*.csv
- qa_output/a10/batch_qa_pairs_cc_news_gpt_5_mini_b*_*.csv
- qa_output/a10/batch_qa_pairs_livedoor_gpt_5_mini_b*_*.csv

出力ファイル:
- qa_output/a02_qa_pairs_cc_news.csv
- qa_output/a02_qa_pairs_livedoor.csv
- qa_output/a03_qa_pairs_cc_news.csv
- qa_output/a03_qa_pairs_livedoor.csv
- qa_output/a10_qa_pairs_cc_news.csv
- qa_output/a10_qa_pairs_livedoor.csv
"""

import csv
import glob
import os


def extract_qa_pairs(input_file: str, output_file: str) -> None:
    """
    CSVファイルからquestionとanswerの列を抽出して新しいCSVファイルを作成する。

    Args:
        input_file: 入力CSVファイルのパス
        output_file: 出力CSVファイルのパス
    """
    # 入力ファイルの存在確認
    if not os.path.exists(input_file):
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        return

    # 出力ディレクトリの作成
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)

            # ヘッダーの確認
            if 'question' not in reader.fieldnames or 'answer' not in reader.fieldnames:
                print(f"エラー: {input_file} にquestionまたはanswerカラムが見つかりません")
                print(f"利用可能なカラム: {reader.fieldnames}")
                return

            # question と answer のみを抽出
            with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=['question', 'answer'])
                writer.writeheader()

                row_count = 0
                for row in reader:
                    writer.writerow({
                        'question': row['question'],
                        'answer': row['answer']
                    })
                    row_count += 1

        print(f"✓ {input_file} から {row_count} 件のQ&Aペアを抽出 → {output_file}")

    except Exception as e:
        print(f"エラー: {input_file} の処理中にエラーが発生しました: {e}")


def get_latest_file(pattern: str) -> str | None:
    """
    指定されたパターンにマッチする最新のファイルを取得する。

    Args:
        pattern: ファイルパターン (例: 'qa_output/a02/qa_pairs_cc_news_*.csv')

    Returns:
        最新のファイルパス。見つからない場合はNone
    """
    files = glob.glob(pattern)
    if not files:
        return None
    # 更新時刻でソートして最新を返す
    return max(files, key=os.path.getmtime)


def main():
    """
    各プログラムの最新出力ファイルからquestion/answerを抽出する。
    """
    # 処理対象ファイルのパターン
    file_patterns = [
        {
            'pattern': 'qa_output/a02/qa_pairs_cc_news_*.csv',
            'output': 'qa_output/a02_qa_pairs_cc_news.csv'
        },
        {
            'pattern': 'qa_output/a02/qa_pairs_livedoor_*.csv',
            'output': 'qa_output/a02_qa_pairs_livedoor.csv'
        },
        {
            'pattern': 'qa_output/a03/qa_pairs_cc_news_*.csv',
            'output': 'qa_output/a03_qa_pairs_cc_news.csv'
        },
        {
            'pattern': 'qa_output/a03/qa_pairs_livedoor_*.csv',
            'output': 'qa_output/a03_qa_pairs_livedoor.csv'
        },
        {
            'pattern': 'qa_output/a10/batch_qa_pairs_cc_news_gpt_5_mini_b*_*.csv',
            'output': 'qa_output/a10_qa_pairs_cc_news.csv'
        },
        {
            'pattern': 'qa_output/a10/batch_qa_pairs_livedoor_gpt_5_mini_b*_*.csv',
            'output': 'qa_output/a10_qa_pairs_livedoor.csv'
        }
    ]

    print("=" * 60)
    print("Q&Aペア抽出処理を開始します")
    print("=" * 60)

    for mapping in file_patterns:
        pattern = mapping['pattern']
        output_file = mapping['output']

        # 最新のファイルを取得
        latest_file = get_latest_file(pattern)

        if latest_file is None:
            print(f"警告: パターン '{pattern}' にマッチするファイルが見つかりません")
            continue

        print(f"最新ファイル: {latest_file}")
        extract_qa_pairs(latest_file, output_file)

    print("=" * 60)
    print("処理が完了しました")
    print("=" * 60)


if __name__ == "__main__":
    main()