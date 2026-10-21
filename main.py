import numpy as np, psycopg, torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

DSN = "postgresql://postgres:Password%402255@localhost:5432/AlbinDB"
LLM = "Qwen/Qwen2.5-1.5B-Instruct"

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tok = AutoTokenizer.from_pretrained(LLM)
llm = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype="auto")

# Load all chunks from Postgres into memory once at startup
with psycopg.connect(DSN) as conn:
    rows = conn.execute("SELECT content, source, page, embedding FROM docs").fetchall()

if not rows:
    raise RuntimeError("The docs table is empty. Run ingest.py first.")

chunks = [(r[0], r[1], r[2]) for r in rows]
matrix = np.array([r[3] for r in rows], dtype=np.float32)   # shape: (num_chunks, 384)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(q: Query):
    qvec = embedder.encode(q.question, normalize_embeddings=True)
    scores = matrix @ qvec                      # cosine similarity (vectors are normalized)
    top = np.argsort(scores)[::-1][:4]          # indices of the 4 best chunks

    context = "\n\n".join(chunks[i][0] for i in top)
    messages = [
        {"role": "system", "content":
         "Answer ONLY using the context. If the answer is not in the context, say you don't know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q.question}"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=300, do_sample=False)
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {"answer": answer,
            "sources": [{"file": chunks[i][1], "page": chunks[i][2]} for i in top]}