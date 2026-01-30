from langchain_mistralai import MistralAIEmbeddings
from langchain_postgres import PGVector
from config import CONNECTING_STRING, MISTRAL_EMBED_MODEL, COLLECTION_NAME


embed_model = MistralAIEmbeddings(model=MISTRAL_EMBED_MODEL)

_retriever_instance = None 

def get_retriever():
    global _retriever_instance

    if _retriever_instance is None:
        _retriever_instance = PGVector(
            connection=CONNECTING_STRING,
            collection_name=COLLECTION_NAME,
            embeddings=embed_model
        )
    return _retriever_instance