
import sys
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Connect to Qdrant
try:
    client = QdrantClient(url="http://localhost:6333")
    collections = client.get_collections()
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    sys.exit(1)

print(f"Found {len(collections.collections)} collections.")

for col in collections.collections:
    name = col.name
    print(f"\n--- Collection: {name} ---")
    try:
        info = client.get_collection(name)
        config = info.config.params.vectors
        
        dims = "Unknown"
        if hasattr(config, "size"):
            dims = config.size
        elif isinstance(config, dict):
             # Handle named vectors
             for k, v in config.items():
                 print(f"  Vector '{k}': Size={v.size}, Distance={v.distance}")
                 dims = v.size # Just take the last one for now
        
        print(f"  Dimensions: {dims}")
        print(f"  Points: {info.points_count}")
        
        # Search for specific question
        from helper_embedding import create_embedding_client
        import numpy as np
        
        print(f"  Listing first 10 points...")
        points, _ = client.scroll(
            collection_name=name,
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        for p in points:
             print(f"    - {p.payload.get('question', 'N/A')[:50]}...")
        
    except Exception as e:
        print(f"  Error getting info: {e}")
            
    except Exception as e:
        print(f"  Error getting info: {e}")
