# DocMind — AI Document Chatbot

> RAG-powered chatbot for PDF and Word documents · Google Gemini API · Ollama (local) · ChromaDB · React + FastAPI

---

## What is DocMind?

DocMind is an AI-powered document chatbot that lets you upload PDF or Word (.docx) files and ask questions about them in natural language. It uses a **Retrieval-Augmented Generation (RAG)** pipeline to find the most relevant sections of your document and generate accurate, grounded answers.

You can choose between two LLM modes:
- **✨ Gemini API** — Google's Gemini 1.5 Flash (free, cloud-based, best quality)
- **💻 Local (Ollama)** — llama3.2 running entirely on your machine (private, offline)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + Vite |
| Backend | Python + FastAPI |
| PDF Parsing | PyMuPDF |
| Word Parsing | python-docx |
| Text Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (in-memory) |
| LLM (API) | Google Gemini 1.5 Flash |
| LLM (Local) | Ollama · llama3.2 |

---

## RAG Pipeline

```
Upload PDF / DOCX
       │
       ▼
  Extract Text
  (PyMuPDF / python-docx)
       │
       ▼
  Split into Chunks
  (400 chars, 75 overlap)
       │
       ▼
  Generate Embeddings
  (sentence-transformers)
       │
       ▼
  Store in ChromaDB
  (in-memory vector store)
       │
       ▼
  User asks a question
       │
       ▼
  Query ChromaDB
  (cosine similarity, top-5 chunks)
       │
       ▼
  Send context + question to LLM
  (OPEN AI or Ollama)
       │
       ▼
  Return grounded answer
```

---

## Project Structure

```
docmind_gemini/
├── backend/
│   ├── main.py          ← FastAPI routes (/upload /chat /reset /health)
│   ├── ingest.py        ← PDF & DOCX extraction + ChromaDB storage
│   ├── retriever.py     ← Cosine similarity search
│   ├── llm.py           ← Gemini API + Ollama calls
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── App.jsx      ← Main UI with mode toggle
    │   ├── App.css      ← Light theme styles
    │   ├── api.js       ← Axios calls to backend
    │   ├── index.css    ← Global styles
    │   └── main.jsx     ← React entry point
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Gemini API key** (for API mode) — free at [aistudio.google.com](https://aistudio.google.com)
- **Ollama** (for local mode) — free at [ollama.com](https://ollama.com)

---

## Getting Started

### 1. Backend Setup

```bash
cd docmind_gemini/backend

# Create virtual environment
py -3.10 -m venv venv

# Activate
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8000
```

Backend runs at: `http://localhost:8000`

---

### 2. Frontend Setup

```bash
cd docmind_gemini/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Frontend runs at: `http://localhost:5173`

---

### 3. Get a Gemini API Key (for API mode)

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key** → **Create API Key**
3. Copy the key — it starts with `AIzaSy...`
4. Paste it into the UI

Free tier: **1500 requests/day** — more than enough for development and demos.

---

### 4. Set Up Ollama (for local mode)

```bash
# Download Ollama from https://ollama.com/download and install

# Pull the model (~2GB download)
ollama pull llama3.2

# Start Ollama server
ollama serve

# Verify it works
ollama run llama3.2 "say hello"
```

---

## Usage

1. Open `http://localhost:5173`
2. Choose your LLM mode:
   - **✨ Gemini API** → paste your `AIzaSy...` key
   - **💻 Local (Ollama)** → make sure `ollama serve` is running
3. Upload a PDF or DOCX file (drag & drop or click)
4. Ask anything about your document
5. Switch between documents or search across all using the scope bar

---

## Supported File Types

| Type | Extension |
|------|-----------|
| PDF | `.pdf` |
| Word | `.docx` |

> ⚠️ Old `.doc` format is not supported. Open in Word → Save As → `.docx`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/documents` | List uploaded documents |
| `POST` | `/upload` | Upload and ingest a file |
| `POST` | `/chat` | Ask a question (RAG + LLM) |
| `DELETE` | `/documents/{doc_id}` | Remove a document |
| `POST` | `/reset` | Clear all documents |

### Chat Request Body

```json
{
  "question": "What are the key findings?",
  "mode": "gemini",
  "api_key": "AIzaSy...",
  "doc_id": null
}
```

`mode` options: `"gemini"` or `"local"`
`doc_id`: `null` = search all documents, or pass a specific doc ID

---

## Features

- Upload multiple PDF and DOCX files
- Switch between **Gemini API** and **local Ollama** mode at any time
- Search across all documents or focus on a single one
- Source chunk badges show which file each answer came from
- Live RAG pipeline tracker (5 steps)
- Clean light theme — fully readable
- No document history — data clears on server restart (privacy-friendly)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `chromadb` SQLite error | Use Python 3.10+ (has SQLite 3.35+) |
| `httpx` proxies error | `pip install httpx==0.27.0` |
| Ollama not connecting | Run `ollama serve` before starting backend |
| Gemini key invalid | Key must start with `AIzaSy` |
| Blank white page in browser | Run `npm install` again in frontend folder |
| `.doc` file not uploading | Save as `.docx` in Microsoft Word first |
