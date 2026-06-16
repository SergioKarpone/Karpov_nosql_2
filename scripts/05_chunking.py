# 5. Розбиття довгих анотацій на чанки двома стратегіями (fixed-size та semantic),
# завантаження у два окремі індекси Pinecone і пошук по чанках.

import os
os.environ["USE_TF"] = "0"       # не вантажити TensorFlow через transformers
os.environ["USE_TORCH"] = "1"    # використовувати PyTorch-бекенд

import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
METRIC     = "cosine"
BATCH_SIZE = 100

INDEX_FIXED    = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"

# Параметри chunking
FIXED_SIZE     = 60      # слів у чанку
FIXED_OVERLAP  = 15      # перекриття у словах
SEMANTIC_MAX   = 60      # макс. слів у семантичному чанку
N_PAPERS       = 30      # скільки найдовших анотацій беремо

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


# Стратегії розбиття - фіксована кількість слів зі ковзним вікном і перекриттям
def chunk_fixed(text: str, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP):
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return chunks

# Просте розбиття на речення по .!? (достатньо для анотацій).
def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]

# Накопичуємо цілі речення, доки не перевищимо ліміт слів.
def chunk_semantic(text: str, max_words: int = SEMANTIC_MAX):
    chunks, current, count = [], [], 0
    for sent in split_sentences(text):
        n = len(sent.split())
        if count + n > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks

# Pinecone
def ensure_index(name: str):
    if name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=name, dimension=VECTOR_DIM, metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    return pc.Index(name)

# Розбиваємо, кодуємо й завантажуємо чанки у відповідний індекс.
def build_and_upload(index, chunker, label: str, papers: pd.DataFrame):
    records = []
    for _, row in papers.iterrows():
        for ci, chunk_text in enumerate(chunker(row["abstract"])):
            records.append({
                "uid":      f"{row['id']}_chunk_{ci}",
                "text":     chunk_text,
                "arxiv_id": str(row["id"]),
                "title":    str(row["title"])[:300],
                "chunk_no": ci,
                "year":     int(row["year"]),
                "category": str(row["category"]),
            })

    print(f"[{label}] усього чанків: {len(records)} "
          f"(в середньому {len(records)/len(papers):.1f} на статтю)")

    texts = [r["text"] for r in records]
    vecs = model.encode(texts, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)

    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=f"upsert {label}"):
        end = min(start + BATCH_SIZE, len(records))
        batch = [{
            "id": records[i]["uid"],
            "values": vecs[i].tolist(),
            "metadata": {
                "arxiv_id": records[i]["arxiv_id"],
                "title":    records[i]["title"],
                "text":     records[i]["text"][:500],
                "chunk_no": records[i]["chunk_no"],
                "year":     records[i]["year"],
                "category": records[i]["category"],
            },
        } for i in range(start, end)]
        index.upsert(vectors=batch)


def search_chunks(index, query: str, top_k: int = 5):
    qv = model.encode([query], normalize_embeddings=True,
                      convert_to_numpy=True)[0]
    res = index.query(vector=qv.tolist(), top_k=top_k, include_metadata=True)
    for rank, m in enumerate(res["matches"], 1):
        md = m["metadata"]
        print(f"  {rank}. [{m['score']:.3f}] {md['title'][:55]} "
              f"(чанк {int(md['chunk_no'])})")
        print(f"     {md['text'][:140]}…")


def main() -> None:
    # 1. Беремо 30 статей із найдовшими анотаціями
    df["abs_len"] = df["abstract"].str.len()
    longest = df.sort_values("abs_len", ascending=False).head(N_PAPERS)
    print(f"Обрано {len(longest)} статей. Довжина анотацій: "
          f"{int(longest['abs_len'].min())}–{int(longest['abs_len'].max())} символів.\n")

    # 2-5. Два індекси, дві стратегії
    idx_fixed    = ensure_index(INDEX_FIXED)
    idx_semantic = ensure_index(INDEX_SEMANTIC)

    build_and_upload(idx_fixed,    chunk_fixed,    "fixed",    longest)
    build_and_upload(idx_semantic, chunk_semantic, "semantic", longest)

    # 6. Пошук по чанках
    queries = [
        "neural network architecture for image classification",
        "statistical method for high energy physics",
    ]
    for q in queries:
        print(f"\nЗАПИТ: «{q}»")
        print("\n[FIXED-SIZE]")
        search_chunks(idx_fixed, q)
        print("\n[SEMANTIC]")
        search_chunks(idx_semantic, q)


if __name__ == "__main__":
    main()
