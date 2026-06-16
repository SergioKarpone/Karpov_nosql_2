# 6. Гібридний пошук: BM25 (лексичний) + Pinecone (векторний),
# об'єднані через Reciprocal Rank Fusion (RRF).

import os
os.environ["USE_TF"] = "0"       # не вантажити TensorFlow через transformers
os.environ["USE_TORCH"] = "1"    # використовувати PyTorch-бекенд

import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10        # беремо ширше, щоб RRF мав що переранжувати
RRF_K = 60        # стандартне значення константи RRF

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


# BM25-індекс по title + abstract
def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", str(text).lower())


CORPUS = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
BM25 = BM25Okapi([tokenize(doc) for doc in CORPUS])

# Повертає список (doc_index, score), відсортований за релевантністю.
def bm25_search(query: str, top_k: int = TOP_K):
    scores = BM25.get_scores(tokenize(query))
    order = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in order]


# Векторний пошук (Pinecone) → індекси рядків df
def vector_search(query: str, top_k: int = TOP_K):
    qv = model.encode([query], normalize_embeddings=True,
                      convert_to_numpy=True)[0]
    res = index.query(vector=qv.tolist(), top_k=top_k, include_metadata=False)
    out = []
    for m in res["matches"]:
        # id має вигляд "paper_<номер рядка>"
        i = int(m["id"].split("_")[1])
        out.append((i, float(m["score"])))
    return out


# Reciprocal Rank Fusion
# Кожен список — [(doc_id, score), ...] у порядку спадання релевантності.
# RRF використовує лише РАНГ: score = Σ 1 / (k + rank).
def rrf_fuse(*ranked_lists, k: int = RRF_K, top_k: int = 5):
    fused: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, (doc_id, _score) in enumerate(lst, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda x: -x[1])
    return ordered[:top_k]


# Вивід
def show(title: str, results, with_score=True):
    print(f"\n[{title}]")
    for rank, item in enumerate(results, 1):
        doc_id = item[0]
        score = item[1]
        t = df.iloc[doc_id]["title"][:62]
        if with_score:
            print(f"  {rank}. [{score:.4f}] {t}")
        else:
            print(f"  {rank}. {t}")


def run_query(query: str):
    print(f"\nЗАПИТ: «{query}»")

    bm = bm25_search(query)
    vec = vector_search(query)
    hybrid = rrf_fuse(bm, vec, k=RRF_K, top_k=5)

    show("BM25 (top-5)", bm[:5])
    show("VECTOR (top-5)", vec[:5])
    show("HYBRID · RRF (top-5)", hybrid)

    # Документи, яких немає в топ-5 жодного окремого методу
    bm_ids  = {i for i, _ in bm[:5]}
    vec_ids = {i for i, _ in vec[:5]}
    new_in_hybrid = [d for d, _ in hybrid if d not in bm_ids and d not in vec_ids]
    if new_in_hybrid:
        print(f"\n  ↳ У гібридному топ-5 з'явились документи поза топ-5 окремих методів: "
              f"{[df.iloc[i]['title'][:40] for i in new_in_hybrid]}")
    print()


if __name__ == "__main__":
    queries = [
        "BERT fine-tuning",                                   # точний термін
        "Yann LeCun convolutional networks",                  # ім'я автора
        "making computers understand human emotions from text",  # перефразування
    ]
    for q in queries:
        run_query(q)
