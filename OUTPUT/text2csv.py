# 変換スクリプト
import pandas as pd

# テキストファイルを読み込み
with open('wikipedia_ja_5per_chunks_cleaned.txt', 'r', encoding='utf-8') as f:
    texts = f.readlines()

# DataFrameに変換
df = pd.DataFrame({
    'text': [line.strip() for line in texts if line.strip()],
    'Combined_Text': [line.strip() for line in texts if line.strip()]
})

# CSV形式で保存
df.to_csv('wikipedia_ja_5per_chunks_cleaned.csv', index=False, encoding='utf-8')
print(f"変換完了: {len(df)} 行")
