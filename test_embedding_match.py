
import os
import numpy as np
from helper_embedding import create_embedding_client

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

try:
    client = create_embedding_client("gemini")
    text_q = "高等学校における「地理歴史科」では、地理はどのように細分化されていますか？"
    text_a = "高等学校の「地理歴史科」では、地理は「地理A」と 「地理B」に細分されています。"
    text_qa = text_q + "\n" + text_a
    
    vec_doc_qa = client.embed_text(text_qa, task_type="retrieval_document")
    vec_doc_q = client.embed_text(text_q, task_type="retrieval_document")
    vec_query_q = client.embed_text(text_q, task_type="retrieval_query")
    
    print(f"Similarity (Doc(Q+A) vs Query(Q)): {cosine_similarity(vec_doc_qa, vec_query_q):.4f}")
    print(f"Similarity (Doc(Q) vs Query(Q)): {cosine_similarity(vec_doc_q, vec_query_q):.4f}")

    
except Exception as e:
    print(f"Error: {e}")
