
import sys
import os
import math
from google import genai

# Add project root to path
sys.path.append(os.getcwd())

from grace.config import get_config

def cosine_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    return dot / (norm1 * norm2)

def check_similarity():
    config = get_config()
    client = genai.Client()
    model = config.embedding.model
    
    # Target text (from Qdrant)
    target_q = "浦沢直樹が初めて受賞したのはいつ、何の賞ですか？"
    target_a = "1982年に第9回小学館新人コミック大賞一般部門に入選したのが初めての受賞です。"
    target_text = f"{target_q} {target_a}" # This is usually how it's stored
    
    # Candidate queries
    queries = [
        "浦沢直樹の受賞歴に関する情報をWikipediaで検索する", # The bad one
        "浦沢直樹 初受賞 賞", # The good keyword one
        "浦沢直樹が初めて受賞したのはいつ、何の賞ですか？", # Exact match
        "浦沢直樹 最初の受賞"
    ]
    
    print(f"Embedding Model: {model}")
    print(f"Target Text: {target_text[:50]}...")
    
    # Embed target
    try:
        resp_target = client.models.embed_content(model=model, contents=target_text)
        vec_target = resp_target.embeddings[0].values
        print(f"Vector Dimension: {len(vec_target)}")
    except Exception as e:
        print(f"Error embedding target: {e}")
        return

    for q in queries:
        try:
            resp_q = client.models.embed_content(model=model, contents=q)
            vec_q = resp_q.embeddings[0].values
            score = cosine_similarity(vec_q, vec_target)
            print(f"Query: '{q}' -> Score: {score:.4f}")
        except Exception as e:
            print(f"Error embedding '{q}': {e}")

if __name__ == "__main__":
    check_similarity()
