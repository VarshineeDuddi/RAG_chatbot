"""
main.py — FastAPI Entry Point
Supports: OpenAI API and Ollama (local) modes.
After upload, extracts document-specific synonyms using the LLM.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest_file, get_collection, SUPPORTED_TYPES
from retriever import retrieve, format_context, extract_synonyms_from_doc
from llm import generate_answer

app = FastAPI(title="DocMind API", version="10.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

documents: dict = {}
ALL_EXTENSIONS  = [ext for exts in SUPPORTED_TYPES.values() for ext in exts]


class ChatRequest(BaseModel):
    question: str
    mode:     str = "openai"    # "openai" or "local"
    api_key:  str = ""
    doc_id:   str | None = None


class ChatResponse(BaseModel):
    answer:   str
    chunks:   list[dict]
    searched: str
    mode:     str


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(documents)}


@app.get("/documents")
def list_documents():
    return {"documents": list(documents.values())}


@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Form(default=""),
    mode: str = Form(default="openai"),
):
    import os
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALL_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload a PDF (.pdf) or Word document (.docx)."
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        stats = ingest_file(file_bytes, file.filename, api_key)
        documents[stats["doc_id"]] = stats

        # Extract document-specific synonyms in background (non-blocking)
        sample_text = " ".join(
            chunk for chunk in [] # will be populated from ChromaDB
        )
        # Get sample text from the first few chunks in ChromaDB
        collection = get_collection()
        sample_chunks = collection.get(
            where={"doc_id": stats["doc_id"]},
            limit=5,
            include=["documents"]
        )
        sample_text = " ".join(sample_chunks.get("documents", []))

        background_tasks.add_task(
            extract_synonyms_from_doc,
            stats["doc_id"],
            sample_text,
            mode,
            api_key,
        )

        return {"success": True, **stats}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        collection = get_collection()
        existing   = collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        del documents[doc_id]
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if req.mode == "openai" and not req.api_key.strip():
        raise HTTPException(status_code=400, detail="OpenAI API key is required.")
    if not documents:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    try:
        chunks = retrieve(req.question, doc_id=req.doc_id, top_k=5)
        if not chunks:
            return ChatResponse(
                answer="That information is not mentioned in the uploaded documents.",
                chunks=[],
                searched=req.doc_id or "all",
                mode=req.mode,
            )

        context = format_context(chunks)
        answer  = generate_answer(req.question, context, req.mode, req.api_key)
        return ChatResponse(
            answer=answer,
            chunks=chunks,
            searched=req.doc_id or "all",
            mode=req.mode,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/reset")
def reset():
    global documents
    try:
        collection = get_collection()
        all_items  = collection.get()
        if all_items["ids"]:
            collection.delete(ids=all_items["ids"])
        documents = {}
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))