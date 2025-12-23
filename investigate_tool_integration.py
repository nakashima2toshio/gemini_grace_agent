
import sys
import os
import logging
import json
from dataclasses import asdict

# Add project root to path
sys.path.append(os.getcwd())

from grace.tools import RAGSearchTool, ToolResult
from grace.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def investigate_tool_output():
    print("=== Tool Integration Investigation ===")
    
    # 1. Initialize Tool
    config = get_config()
    tool = RAGSearchTool(config=config)
    
    query = "古代ギリシア人の哲学に見られた二つの傾向とは何ですか？"
    collection = "wikipedia_ja" # Target collection
    
    print(f"Executing RAGSearchTool with query: {query}")
    print(f"Collection: {collection}")
    
    # 2. Execute
    try:
        result: ToolResult = tool.execute(query=query, collection=collection)
        
        print("\n=== Tool Result Summary ===")
        print(f"Success: {result.success}")
        print(f"Execution Time: {result.execution_time_ms}ms")
        
        print("\n=== Confidence Factors ===")
        print(json.dumps(result.confidence_factors, indent=2, ensure_ascii=False))
        
        print("\n=== Output (First Item) ===")
        if isinstance(result.output, list) and len(result.output) > 0:
            first = result.output[0]
            print(f"Score: {first.get('score')}")
            print(f"Payload Q: {first.get('payload', {}).get('question')}")
        else:
            print(f"Output type: {type(result.output)}")
            print(f"Output content: {result.output}")

    except Exception as e:
        print(f"Tool Execution Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_tool_output()
