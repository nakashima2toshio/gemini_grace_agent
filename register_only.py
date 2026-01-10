# register_only.py
import pandas as pd
from services.qdrant_service import (
    create_or_recreate_collection_for_qdrant,
    embed_texts_for_qdrant,
    upsert_points_to_qdrant,
    build_points_for_qdrant
)
from qdrant_client_wrapper import create_qdrant_client

csv_path = "qa_output/qa_pairs_wikipedia_ja.csv"
collection_name = "qa_wikipedia_ja"

df = pd.read_csv(csv_path)
print(f"CSV件数: {len(df)}")

texts = (df['question'].astype(str) + "\n" + df['answer'].astype(str)).tolist()

client = create_qdrant_client()
create_or_recreate_collection_for_qdrant(client, collection_name, recreate=True)

batch_size = 100
total = 0

for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i:i + batch_size]
    batch_texts = texts[i:i + batch_size]

    vectors = embed_texts_for_qdrant(batch_texts)
    points = build_points_for_qdrant(
        batch_df, vectors,
        domain=collection_name,
        source_file="qa_pairs_wikipedia_ja_5per.csv",
        start_index=i
    )
    upsert_points_to_qdrant(client, collection_name, points)
    total += len(points)
    print(f"進捗: {total}/{len(df)}")

print(f"完了！")
