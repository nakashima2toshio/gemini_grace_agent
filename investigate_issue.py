
import sys
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Add project root to path
sys.path.append(os.getcwd())

# Import wrapper to use the same logic as the agent
from qdrant_client_wrapper import (
    embed_query_unified, 
    search_collection, 
    QDRANT_CONFIG,
    DEFAULT_EMBEDDING_PROVIDER
)
from grace.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def investigate():
    print("=== Investigation Start ===")
    
    # 1. Connect to Qdrant
    print(f"Connecting to Qdrant at {QDRANT_CONFIG['url']}...")
    client = QdrantClient(url=QDRANT_CONFIG['url'])
    
    # 2. Check Collections
    collections = client.get_collections().collections
    print(f"Found {len(collections)} collections:")
    target_collection = "wikipedia_ja"
    found = False
    for c in collections:
        print(f" - {c.name}")
        if c.name == target_collection:
            found = True
            
    if not found:
        print(f"WARNING: Collection '{target_collection}' not found. Searching for similar...")
        # Try to find one that looks like it
        for c in collections:
            if "wikipedia" in c.name:
                target_collection = c.name
                found = True
                print(f"Selected '{target_collection}' instead.")
                break
    
    if not found:
        print("ERROR: No suitable collection found.")
        return

    # 3. Check Collection Config (Vector Size)
    col_info = client.get_collection(target_collection)
    vectors_config = col_info.config.params.vectors
    print(f"\nCollection '{target_collection}' Config:")
    
    vec_size = 0
    if isinstance(vectors_config, dict):
        for name, cfg in vectors_config.items():
            print(f" - Vector '{name}': size={cfg.size}, distance={cfg.distance}")
            vec_size = cfg.size
    else:
        print(f" - Default Vector: size={vectors_config.size}, distance={vectors_config.distance}")
        vec_size = vectors_config.size

    # 4. Embed Query
    query_text = "古代ギリシア人の哲学に見られた二つの傾向とは何ですか？"
    print(f"\nQuery: {query_text}")
    print(f"Current Embedding Provider: {DEFAULT_EMBEDDING_PROVIDER}")
    
    try:
        query_vec = embed_query_unified(query_text)
        print(f"Generated Query Vector Dimension: {len(query_vec)}")
        
        if len(query_vec) != vec_size:
            print(f"CRITICAL MISMATCH: Query dim ({len(query_vec)}) != Collection dim ({vec_size})")
            print("This is likely the cause of poor search results.")
        else:
            print("Dimension match confirmed.")
            
    except Exception as e:
        print(f"Error embedding query: {e}")
        return

    # 5. Search
    print(f"\nSearching in '{target_collection}'...")
    try:
        results = search_collection(client, target_collection, query_vec, limit=5)
        
        print(f"Found {len(results)} results:")
        for i, res in enumerate(results):
            score = res['score']
            payload = res['payload']
            q = payload.get('question', 'N/A')
            a = payload.get('answer', 'N/A')
            print(f"\n[{i+1}] Score: {score:.4f}")
            print(f"    Q: {q}")
            print(f"    A: {a}")
            
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    investigate()
