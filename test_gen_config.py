import google.generativeai as genai
from pydantic import BaseModel
import os

# Dummy Pydantic model
class ChunksResult(BaseModel):
    summary: str
    sentiment: str

print("--- Testing GenerationConfig instantiation ---")
try:
    config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=ChunksResult
    )
    print("Success: GenerationConfig created with Pydantic model.")
    print(config)
except Exception as e:
    print(f"Error: {e}")

print("\n--- Testing GenerativeModel instantiation ---")
try:
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config=config
    )
    print("Success: GenerativeModel created.")
except Exception as e:
    print(f"Error: {e}")
