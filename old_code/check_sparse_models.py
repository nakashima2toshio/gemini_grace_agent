from fastembed import SparseTextEmbedding

try:
    models = SparseTextEmbedding.list_supported_models()
    print("Supported Sparse Models:")
    for model in models:
        print(f"- {model['model']} (dim: {model.get('dim', 'N/A')}, description: {model.get('description', 'N/A')})")
except Exception as e:
    print(f"Error listing models: {e}")
