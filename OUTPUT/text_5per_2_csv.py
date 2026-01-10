# 変換スクリプト: text_5per_2_csv.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
入力テキストファイルの5%を抽出してCSV形式で保存するスクリプト

使用例:
python text_5per_2_csv.py -i input_text.txt -o output.csv

python text_5per_2_csv.py -i japanese_text_20251130_041517.txt -o japanese_text_5per.csv
python text_5per_2_csv.py -i livedoor_20251130_041553.txt -o livedoor_5per.csv
python text_5per_2_csv.py -i cc_news_20251130_041450.txt -o cc_news_5per.csv
python text_5per_2_csv.py -i wikipedia_ja_20251130_041304.txt -o wikipedia_ja_5per.csv
python text_5per_2_csv.py -i qa_pairs_fineweb_edu_ja.csv -o fineweb_edu_ja_5per.csv

"""

import argparse
import pandas as pd
import os
import sys


def main():
    # コマンドライン引数のパーサーを設定
    parser = argparse.ArgumentParser(
        description='テキストファイルの5%を抽出してCSV形式で保存します。'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='入力テキストファイルのパス'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='出力CSVファイルのパス'
    )

    args = parser.parse_args()

    input_filename = args.input
    output_filename = args.output

    # 入力ファイルの存在確認
    if not os.path.exists(input_filename):
        print(f"エラー: ファイル '{input_filename}' が見つかりませんでした。", file=sys.stderr)
        sys.exit(1)

    print(f"入力ファイル: {input_filename}")
    print(f"出力ファイル: {output_filename}")
    print("-" * 50)

    # ファイルを読み込み
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"エラー: ファイルの読み込みに失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    # 全体の長さを取得
    total_length = len(content)

    # 5%の長さを計算
    five_percent_length = int(total_length * 0.05)

    # 5%分を切り出し
    sliced_content = content[:five_percent_length]

    print(f"オリジナルファイルの文字数: {total_length:,}")
    print(f"抽出した文字数 (5%): {five_percent_length:,}")

    # 行ごとに分割
    texts = sliced_content.split('\n')

    # 空行を除外してDataFrameに変換
    df = pd.DataFrame({
        'text'         : [line.strip() for line in texts if line.strip()],
        'Combined_Text': [line.strip() for line in texts if line.strip()]
    })

    # CSV形式で保存
    try:
        df.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"CSV行数: {len(df):,}")
        print(f"✓ ファイル '{output_filename}' を作成しました。")
    except Exception as e:
        print(f"エラー: CSVファイルの保存に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
