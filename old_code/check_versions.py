import sys
try:
    import streamlit
    print(f"Streamlit: {streamlit.__version__}")
except ImportError:
    print("Streamlit: Not installed")

try:
    import streamlit_mermaid
    # streamlit_mermaid might not have a __version__ attribute, or it might be different.
    # checking if we can import it is the first step.
    try:
        version = streamlit_mermaid.__version__
    except AttributeError:
        version = "Installed (unknown version)"
    print(f"Streamlit Mermaid: {version}")
except ImportError:
    print("Streamlit Mermaid: Not installed")
