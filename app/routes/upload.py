# app/routes/upload.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.dependencies import model, es
from app.utils import split_text_semantically, generate_unique_id, extract_text_from_pdf, extract_text_from_word, clean_text

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

    clean_text_content = clean_text(text)

    for chunk in split_text_semantically(clean_text_content):
        if chunk:
            doc_id = generate_unique_id(chunk)
            embedding = model.encode(chunk).tolist()
            body = {"text": chunk, "embedding": embedding}
            es.index(index=index_name, id=doc_id, body=body)

    return {"message": "File processed and indexed successfully"}
