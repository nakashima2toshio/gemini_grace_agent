
import sys
import os
from qdrant_client import QdrantClient

sys.path.append(os.getcwd())
from grace.config import get_config

def check_collection_info():
    config = get_config()
    client = QdrantClient(url=config.qdrant.url)
    
    collection_name = "wikipedia_ja" # Or the specific collection name being used
    
    try:
        # Check if collection exists
        if not client.collection_exists(collection_name):
            print(f"Collection {collection_name} not found.")
            # List all
            print("Collections:", [c.name for c in client.get_collections().collections])
            return

        info = client.get_collection(collection_name)
        print(f"Collection: {collection_name}")
        print(f"Vector Config: {info.config.params.vectors}")
        
        # Try to find alias or real name if different
        # In the logs user saw "qa_a02_qa_pairs_wikipedia_ja" maybe?
        # The user said "collection: wikipedia_ja".
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_collection_info()
