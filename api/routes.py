from langchain_core.messages import HumanMessage, AIMessage
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import StreamingResponse
import asyncio
import shutil
import os
import tempfile
import json
from langsmith import traceable

from services.embedding import pdf_embed
from services.rag import rag_graph, checkpointer
from services.hallucination import verification_report, formatted_hallucination_report
from database.vector_store import get_retriever, embed_model
from config import MAX_FILE_SIZE


router = APIRouter()

rag_workflow = rag_graph()


# for chat message
class MessageRequest(BaseModel):
    content: str = Field(..., description='User message input')
    hallucination_check: bool = Field(default=False, description='Enable hallucination check')

    @field_validator('content')
    @classmethod
    def not_blank(cls, v: str):
        if not v.strip():
            raise ValueError("Can't be empty space")
        return v.strip()


# for semantic search
class SearchRequest(BaseModel):
    query: str = Field(..., description='Search input')
    top_k: int = Field(default=1, gt=0)

    @field_validator('query')
    @classmethod
    def not_blank(cls, v: str):
        if not v.strip():
            raise ValueError("Can't be empty space")
        return v.strip()


# ------------------------------- Endpoints

# chat message
@router.post('/users/{user_id}/threads/{thread_id}/messages')
async def message_stream(user_id: str, thread_id: str, req: MessageRequest):
    return StreamingResponse(
        generate_stream(
            user_id=user_id,
            thread_id=thread_id,
            user_input=req.content,
            hallucination_check=req.hallucination_check
        ),
        media_type='text/event-stream'
    )


# pdf uploading 
@router.post('/users/{user_id}/threads/{thread_id}/documents')
async def upload_pdf(user_id: str, thread_id: str, file: UploadFile = File(...)):
   
    validate_upload_file(file=file)

    temp_dir = tempfile.gettempdir()
    file_name = file.filename.replace(' ', '_')
    temp_path = os.path.join(temp_dir, file_name)

    # save file temporary
    with open(file=temp_path, mode='wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        pdf_embed(file_path=temp_path, user_id=user_id, thread_id=thread_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error in embedding storing in DB: {e}')

    finally:  # cleaning temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "status": 'success',
        'filename': file.filename,
        'user_id': user_id,
        'thread_id': thread_id
    }


# semantic searching 
@router.post('/users/{user_id}/search')
def query_search(user_id: str, req: SearchRequest):
    query = req.query
    top_k = req.top_k
     
    try: 
        response = retrieve_semantic_content(query=query, user_id=user_id, top_k=top_k)
        if not response:
            return {'result': f'Awww! I found nothing for you "{user_id}"'}
        return {'result': response}
    except Exception as e:
        return {'result': f"Oops! we get error here '{str(e)}'"}         


# health 
@router.get('/health')
def health_check():
    try:
        get_retriever().similarity_search("test", k=1)
        db_status = "connected"
    except Exception as e:
        db_status = f"error in supabase or embedding : {str(e)}"    
    return {
        'status':'healthy',
        'database':db_status,
        'service': 'RAG API'
    }

# ------------------------------- Functions

@traceable
async def generate_stream(user_id: str, thread_id: str, user_input: str, hallucination_check: bool = None):
    config = {'configurable': {'thread_id': thread_id, 'user_id': user_id}}

    response = rag_workflow.stream(
        {'messages': HumanMessage(content=user_input)},
        config=config,
        stream_mode='messages'
    )
    
    try:
        for chunk, metadata in response:
            if chunk.content and isinstance(chunk, AIMessage):
                yield chunk.content
    except Exception as e:
        yield f"Sorry! I got problem ;) {str(e)}"

    # hallucination response
    if hallucination_check:
        yield "\n\n\n[======= **HALLUCINATION_REPORT** =======]\n\n"
        await asyncio.sleep(1)
        try:
            message_data = checkpointer.get(config)['channel_values']['messages']
            report = verification_report(message_data)
            final_report = formatted_hallucination_report(report)
            yield final_report

        except Exception as e:
            yield f"Sorry! we getting error here \n{str(e)}"


def retrieve_semantic_content(query: str, user_id: str = None, top_k=1) -> str:
    response = get_retriever().similarity_search(
        query=query, k=top_k,
        filter={'user_id': user_id} if user_id else None
    )
    page_content = "\n\n".join([doc.page_content for doc in response])
    return page_content


def validate_upload_file(file: UploadFile):
    content = file.file.read(MAX_FILE_SIZE + 1)
    file.file.seek(0)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File tool large, 100MB max limit")

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF file allowed")

    return file