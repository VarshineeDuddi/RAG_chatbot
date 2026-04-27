"""
llm.py — RAG Step 5
Supports two modes:
  - openai: OpenAI API (gpt-4o-mini)
  - local:  Ollama (llama3.2 running locally)
"""

import requests
from openai import OpenAI

OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are DocMind, a smart and knowledgeable document assistant.

Answer the user's question naturally and confidently, as if you are deeply familiar with the document.

GUIDELINES:
- Focus on the meaning of the question, not just exact wording.
- If the exact phrase is not present but a closely related concept exists (e.g., "growth rate" instead of "GDP growth rate"), use that information.
- Prefer the most relevant and semantically similar information available.
- Do NOT ignore correct answers just because wording differs slightly.

STRICT RULES:
- NEVER say "based on the context", "based on the chunks", "according to Chunk 1", "the provided context", or anything that reveals retrieval.
- NEVER mention sources, chunks, or internal processing.

- Do NOT substitute with unrelated numbers or facts (e.g., fiscal deficit, debt-to-GDP) unless they directly answer the question.

- If multiple relevant pieces of information exist, present them clearly using bullet points.

- If no relevant or related information exists at all, respond with:
  "I couldn't find this information in the document."

STYLE:
- Use a clean, professional, and confident tone.
- Use **bold formatting** for important numbers, terms, and key facts.
- Keep answers concise but complete.
"""

def ask_openai(question: str, context: str, api_key: str) -> str:
    """Call OpenAI API (gpt-4o-mini)."""
    client = OpenAI(api_key=api_key)

    user_message = f"""Use the following document information to answer the question.
Do not mention that you are reading from chunks or context — just answer naturally.

Document Information:
{context}

Question: {question}"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def ask_ollama(question: str, context: str) -> str:
    """Call local Ollama model."""
    prompt = f"""{SYSTEM_PROMPT}

Use the following document information to answer the question.
Do not mention chunks or context — just answer naturally.

Document Information:
{context}

Question: {question}

Answer:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 1024}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "Sorry, I could not generate a response.")
    except requests.exceptions.ConnectionError:
        raise Exception("Ollama is not running. Please start it with: ollama serve")
    except requests.exceptions.Timeout:
        raise Exception("Ollama took too long to respond.")
    except Exception as e:
        raise Exception(f"Ollama error: {str(e)}")


def generate_answer(question: str, context: str, mode: str = "openai", api_key: str = "") -> str:
    """
    Unified entry point.
    mode: 'openai' → OpenAI gpt-4o-mini
          'local'  → Ollama (local)
    """
    if mode == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        return ask_openai(question, context, api_key)
    elif mode == "local":
        return ask_ollama(question, context)
    else:
        raise ValueError(f"Unknown mode: {mode}")