# app/routes/index.py

from fastapi import APIRouter, Body
from app.models import IndexRequest
from app.dependencies import ollama_service, es
from app.utils import split_text_semantically, generate_unique_id

router = APIRouter()

@router.post("/api/index/")
async def index_text(index_request: IndexRequest = Body(embed=True)):
    index, text = index_request.index_name, index_request.text

    for chunk in split_text_semantically(text):
        if chunk:
            doc_id = generate_unique_id(chunk)
            body = {"text": chunk, "embedding": ollama_service.generate_embedding(chunk)}
            es.index(index=index, id=doc_id, body=body)

    return {"message": "Text indexed successfully"}
