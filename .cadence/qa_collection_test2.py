# qa_collection_test2.py
import pandas as pd
from qdrant_client import QdrantClient
from services.qdrant_service import embed_query_for_search
from qdrant_client_wrapper import search_collection

# CSVから1件取得
df = pd.read_csv("qa_output/qa_pairs_wikipedia_ja_5per.csv")
test_question = df.iloc[0]["question"]
print(f"テスト質問: {test_question}")

# 検索
client = QdrantClient(url="http://localhost:6333")
qvec = embed_query_for_search(test_question, "gemini-embedding-001", 3072)
results = search_collection(client, "qa_wikipedia_ja_5per", qvec, limit=5)

print(f"\n検索結果:")
for r in results:
    print(f"  スコア: {r['score']:.4f} | 質問: {r['payload']['question'][:50]}...")
