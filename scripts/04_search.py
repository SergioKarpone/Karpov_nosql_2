# 4. Три види пошуку:
# 1) чистий семантичний пошук у Pinecone;
# 2) пошук із фільтрами по метаданих (рік, категорія);
# 3) порівняння метрик cosine / dot / L2 на локальних ембеддингах.

import os
os.environ["USE_TF"] = "0"       # не вантажити TensorFlow через transformers
os.environ["USE_TORCH"] = "1"    # використовувати PyTorch-бекенд

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
CURRENT_YEAR = 2026

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# Локальні ембеддинги для порівняння метрик (рядок i ↔ paper_i)
EMB = np.load("embeddings/embeddings.npy")


# Допоміжні функції

# Кодуємо запит у нормалізований вектор (як і документи).
def encode_query(query: str) -> np.ndarray:
    return model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)


def show_pinecone_hits(matches) -> None:
    for rank, m in enumerate(matches, 1):
        md = m["metadata"]
        abstract = md.get("abstract", "")[:160]
        print(f"  {rank}. [{m['score']:.3f}] {md['title']}")
        print(f"     {md['category']} · {int(md['year'])} · {md.get('authors','')[:60]}")
        print(f"     {abstract}…\n")


def semantic_search(query: str, top_k: int = TOP_K, flt: dict | None = None):
    qv = encode_query(query)
    return index.query(
        vector=qv.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=flt,
    )["matches"]


# 1. Чистий семантичний пошук
def part_pure_semantic() -> None:
    query = "teaching machines to recognize objects in pictures"
    print(f"\n1) СЕМАНТИЧНИЙ ПОШУК: «{query}»")
    show_pinecone_hits(semantic_search(query))


# 2. Пошук із фільтрацією по метаданих
def part_filtered_search() -> None:
    query = "reinforcement learning"

    print("\n2A) reinforcement learning · останні 5 років · category = cs.LG")
    flt_a = {
        "year":     {"$gte": CURRENT_YEAR - 5},
        "category": {"$eq": "cs.LG"},
    }
    show_pinecone_hits(semantic_search(query, flt=flt_a))

    print("\n2B) reinforcement learning · статті до 2015 року · будь-яка категорія")
    flt_b = {"year": {"$lt": 2015}}
    show_pinecone_hits(semantic_search(query, flt=flt_b))

    print(">> Порівняння: фільтр A повертає свіжі deep-RL роботи саме з cs.LG,")
    print(">> фільтр B - ранні/класичні згадки RL із різних категорій. Семантика")
    print(">> запиту однакова, але метадані-фільтр звужує кандидатів ДО ранжування.\n")


# 3. Порівняння метрик на локальних ембеддингах
def topk_indices_by_metric(qv: np.ndarray, k: int = TOP_K):
    # cosine: вектори нормалізовані, тож cosine == dot, але рахуємо явно
    norms = np.linalg.norm(EMB, axis=1)
    cosine = (EMB @ qv) / (norms * np.linalg.norm(qv) + 1e-12)
    dot    = EMB @ qv
    l2     = np.linalg.norm(EMB - qv, axis=1)

    return {
        "cosine": np.argsort(-cosine)[:k],   # більше = краще
        "dot":    np.argsort(-dot)[:k],      # більше = краще
        "l2":     np.argsort(l2)[:k],        # менше = краще
    }


def part_metric_comparison() -> None:
    query = "teaching machines to recognize objects in pictures"
    print(f"\n3) ПОРІВНЯННЯ МЕТРИК (локально): «{query}»")
    qv = encode_query(query)
    tops = topk_indices_by_metric(qv)

    for metric, idxs in tops.items():
        print(f"\n{metric.upper()}")
        for rank, i in enumerate(idxs, 1):
            print(f"  {rank}. {df.iloc[i]['title'][:70]}")

    same = (list(tops["cosine"]) == list(tops["dot"]) == list(tops["l2"]))
    print(f"\n>> Топ-5 збігаються по всіх трьох метриках: {same}")
    print(">> Причина: вектори нормалізовані. Для |a|=|b|=1 маємо")
    print(">> ||a-b||² = 2 − 2·(a·b), тож мінімум L2 = максимум dot = максимум cosine.\n")


if __name__ == "__main__":
    part_pure_semantic()
    part_filtered_search()
    part_metric_comparison()
