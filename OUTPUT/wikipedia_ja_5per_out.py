import os

input_filename = 'wikipedia_ja_20251130_041304.txt'
output_filename = 'wikipedia_ja_5per.txt'

# ファイルが存在するか確認
if os.path.exists(input_filename):
    with open(input_filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # 全体の長さを取得
    total_length = len(content)

    # 10%の長さを計算 (整数に切り捨て)
    ten_percent_length = int(total_length * 0.05)

    # 5%分を切り出し
    sliced_content = content[:ten_percent_length]

    # 新しいファイルに書き出し
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(sliced_content)

    print(f"オリジナルファイルの文字数: {total_length}")
    print(f"作成したファイルの文字数 (10%): {ten_percent_length}")
    print(f"ファイル '{output_filename}' を作成しました。")
else:
    print(f"エラー: ファイル '{input_filename}' が見つかりませんでした。")
