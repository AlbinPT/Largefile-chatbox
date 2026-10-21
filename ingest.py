import glob, os, psycopg
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:Password%402255@localhost:5432/AlbinDB"
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk(text, size=800, overlap=150):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out

with psycopg.connect(DSN, autocommit=True) as conn:
    conn.execute("DROP TABLE IF EXISTS docs")
    conn.execute("""
        CREATE TABLE docs (
            id SERIAL PRIMARY KEY,
            source TEXT, page INT, content TEXT,
            embedding DOUBLE PRECISION[]
        )""")

    for path in glob.glob("pdfs/*.pdf"):
        reader = PdfReader(path)
        for pno, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            pieces = chunk(text)
            vecs = embedder.encode(pieces, normalize_embeddings=True)
            for piece, vec in zip(pieces, vecs):
                conn.execute(
                    "INSERT INTO docs (source, page, content, embedding) VALUES (%s,%s,%s,%s)",
                    (os.path.basename(path), pno, piece, vec.tolist()),
                )
        print("Done:", path)