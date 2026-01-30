import os
from dotenv import load_dotenv

load_dotenv()


# Model Configuration
MISTRAL_EMBED_MODEL = "mistral-embed"
MISTRAL_CHAT_MODEL = "mistral-large-latest"
GROQ_CHAT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
HF_MODEL = "deepseek-ai/DeepSeek-V3.2"

OLLAMA_TOOL_MODEL = "phi4-mini" 
OLLAMA_CHAT_MODEL = "phi4-mini"


# Database Configuration
CONNECTING_STRING = os.getenv("CONNECTING_STRING")


# Vector Store Configuration
COLLECTION_NAME = "rag_pdf_collection" 
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100



# API Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max limit