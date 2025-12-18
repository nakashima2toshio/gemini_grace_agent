#
import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter

# データのルートディレクトリ
DATA_DIR = "./datasets/livedoor/text/"


def load_livedoor_data():
    docs = []
    # カテゴリごとのフォルダを走査
    categories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    for category in categories:
        print(f"Loading category: {category}...")
        files = glob.glob(os.path.join(DATA_DIR, category, "*.txt"))

        for file_path in files:
            if os.path.basename(file_path) == "LICENSE.txt": continue  # ライセンスファイルは除外

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) < 4: continue  # 中身がないファイルはスキップ

                url = lines[0].strip()
                date = lines[1].strip()
                title = lines[2].strip()
                body = "".join(lines[3:]).strip()

                # メタデータとしてカテゴリやタイトルを保持
                metadata = {
                    "source"  : "livedoor",
                    "category": category,
                    "title"   : title,
                    "date"    : date,
                    "url"     : url
                }

                # 本文とメタデータをセットにする
                docs.append({"text": body, "metadata": metadata})

    return docs

# --- この後、EmbeddingしてQdrantにUpsertする処理に続きます ---
