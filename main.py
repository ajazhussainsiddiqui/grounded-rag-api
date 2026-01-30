from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title='RAG API',
    description="rag tool and semantic searching (only for demonstration purpose)",
    version='1.0.0'
)

app.include_router(router)


@app.get('/')
def home():
    return {'message': 'Dark home page...'}


