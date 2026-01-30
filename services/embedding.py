from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from database.vector_store import embed_model
from config import CHUNK_SIZE, CHUNK_OVERLAP, CONNECTING_STRING, COLLECTION_NAME


def pdf_embed(file_path, user_id, thread_id):
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()

    # Adding user_id and thread_id in doc, as metadata
    def enrich_doc_metadata(docs, user_id, thread_id):
        for page in docs:
            page.metadata['user_id'] = user_id
            page.metadata['thread_id'] = thread_id
        print("doc id attachement DONE")

    enrich_doc_metadata(docs, user_id, thread_id)


    # splitting into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP, 
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents=docs)
    

    # storing the embedding chunks along with respective text into vector store
    def embed_store(chunks):
        PGVector.from_documents(
            connection=CONNECTING_STRING,
            embedding=embed_model,
            documents=chunks,
            collection_name=COLLECTION_NAME,
            use_jsonb=True
        )
        print('embedded storing in PG vector store is DONE')

    embed_store(chunks)