# app/routes/index.py

from fastapi import APIRouter, Body
from app.models import IndexRequest
from app.dependencies import model, es
from app.utils import split_text_semantically, generate_unique_id, clean_text

router = APIRouter()

@router.post("/api/index/")
async def index_text(index_request: IndexRequest = Body(embed=True)):
    index, text = index_request.index_name, index_request.text
    clean_text_content = clean_text(text)

    for chunk in split_text_semantically(clean_text_content):
        if chunk:
            doc_id = generate_unique_id(chunk)
            embedding = model.encode(chunk).tolist()
            body = {"text": chunk, "embedding": embedding}
            es.index(index=index, id=doc_id, body=body)

    return {"message": "Text indexed successfully"}
