
import logging
from typing import List, Dict, Any

# Configure logging to show info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Replicating the logic from agent_tools.py exactly ---
def filter_results_by_keywords_debug(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Debug version of filter_results_by_keywords to show what's happening.
    """
    import re
    
    # Logic from agent_tools.py
    tokens = query.split()
    required_keywords = []
    
    for t in tokens:
        if len(t) >= 2:
             required_keywords.append(t)

    required_keywords = list(set(required_keywords))
    print(f"DEBUG: Query split() tokens: {tokens}")
    print(f"DEBUG: Required keywords extracted: {required_keywords}")

    filtered_results = []
    for res in results:
        payload = res.get("payload", {})
        content = (str(payload.get("question", "")) + " " + 
                   str(payload.get("answer", "")) + " " + 
                   str(payload.get("content", "")))

        print(f"\nDEBUG: Checking Result (Score: {res.get('score', 0):.4f})")
        print(f"DEBUG: Content Preview: {content[:50]}...")

        is_relevant = True
        if required_keywords:
            hit_count = sum(1 for k in required_keywords if k in content)
            print(f"DEBUG: Hit count: {hit_count}/{len(required_keywords)}")
            
            # 1つもヒットしない場合は除外
            if hit_count == 0:
                is_relevant = False
                print(f"DEBUG: -> REJECTED (No keywords found)")
            else:
                print(f"DEBUG: -> ACCEPTED")
        else:
            print("DEBUG: -> ACCEPTED (No keywords to filter)")

        if is_relevant:
            filtered_results.append(res)
            
    return filtered_results

def run_investigation():
    # 1. The Query
    query = "古代ギリシア人の哲学に見られた二つの傾向とは何ですか？"
    
    # 2. The high-scoring result (Simulated based on previous Qdrant findings)
    # Score was 0.9042
    result_high_score = {
        "score": 0.9042,
        "payload": {
            "question": "古代ギリシア人の哲学に見られた二つの傾向とは何ですか？",
            "answer": "合理的で冷静な傾向と、迷信的で熱狂的な傾向の二つが見られました。",
            "content": ""
        }
    }
    
    results = [result_high_score]
    
    print("=== Start Investigation of Filtering Logic ===")
    print(f"Query: {query}")
    
    filtered = filter_results_by_keywords_debug(results, query)
    
    print("\n=== Summary ===")
    print(f"Input Results: {len(results)}")
    print(f"Output Results: {len(filtered)}")
    
    if len(results) > 0 and len(filtered) == 0:
        print("CONCLUSION: The high-scoring result was discarded by the filter.")
    else:
        print("CONCLUSION: The result survived the filter.")

if __name__ == "__main__":
    run_investigation()
