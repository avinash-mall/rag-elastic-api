# app/routes/upload.py

from fastapi import APIRouter, HTTPException, UploadFile, File
from app.dependencies import ollama_service, es
from app.utils import split_text_semantically, generate_unique_id, extract_text_from_pdf, extract_text_from_word

router = APIRouter()

@router.post("/api/upload/")
async def upload_file(index_name: str, file: UploadFile = File(...)):
    if file.content_type == "application/pdf":
        text = extract_text_from_pdf(file)
    elif file.content_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        text = extract_text_from_word(file)
    elif file.content_type == "text/plain":
        text = (await file.read()).decode('utf-8')
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    for chunk in split_text_semantically(text):
        if chunk:
            doc_id = generate_unique_id(chunk)
            body = {"text": chunk, "embedding": ollama_service.generate_embedding(chunk)}
            es.index(index=index_name, id=doc_id, body=body)

    return {"message": "File processed and indexed successfully"}
