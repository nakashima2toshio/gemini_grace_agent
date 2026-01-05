import google.generativeai as genai
import inspect
from typing import get_type_hints

print("--- genai version ---")
try:
    print(genai.__version__)
except:
    print("No version attribute")

print("\n--- GenerationConfig inspect ---")
try:
    print(inspect.signature(genai.GenerationConfig))
except Exception as e:
    print(f"Error inspecting signature: {e}")

print("\n--- GenerationConfig dir ---")
try:
    print(dir(genai.GenerationConfig))
except:
    pass

print("\n--- types.GenerationConfigDict ---")
try:
    from google.generativeai.types import GenerationConfigDict
    print("GenerationConfigDict exists")
    print(GenerationConfigDict.__annotations__)
except ImportError:
    print("GenerationConfigDict not found in google.generativeai.types")
except Exception as e:
    print(f"Error: {e}")
