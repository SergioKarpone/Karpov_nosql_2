# 3. Створюємо serverless-індекс arxiv-papers у Pinecone (якщо немає)
# і завантажуємо в нього вектори з метаданими батчами по 200.

import os
os.environ["USE_TF"] = "0"       # не вантажити TensorFlow через transformers
os.environ["USE_TORCH"] = "1"    # використовувати PyTorch-бекенд

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

INPUT_PARQUET    = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200          # Pinecone рекомендує батчі до 200 векторів
METRIC     = "cosine"     # вектори нормалізовані → cosine коректна метрика

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def ensure_index() -> "Pinecone.Index":
    # Створюємо індекс, якщо його ще немає, і чекаємо готовності.
    existing = [idx["name"] for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Створюємо індекс '{INDEX_NAME}' (dim={VECTOR_DIM}, metric={METRIC}) ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # чекаємо, доки індекс перейде у стан ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("Індекс готовий.")
    else:
        print(f"Індекс '{INDEX_NAME}' уже існує — використовуємо його.")
    return pc.Index(INDEX_NAME)

# Метадані одного вектора. abstract обрізаємо до 500 символів:
# Pinecone обмежує сумарний розмір метаданих вектора до 40 KB,
# повний текст лишається в parquet і підтягується за ID.
def build_metadata(row: pd.Series) -> dict:
    return {
        "arxiv_id": str(row["id"]),
        "title":    str(row["title"])[:500],
        "abstract": str(row["abstract"])[:500],
        "authors":  str(row["authors"])[:200],
        "year":     int(row["year"]),
        "category": str(row["category"]),
    }


def main() -> None:
    index = ensure_index()

    # Завантажуємо дані та вектори
    df = pd.read_parquet(INPUT_PARQUET).reset_index(drop=True)
    embeddings = np.load(INPUT_EMBEDDINGS)
    assert len(df) == len(embeddings), (
        f"Кількість записів ({len(df)}) != кількість векторів ({len(embeddings)})"
    )
    print(f"Завантажую {len(df)} векторів батчами по {BATCH_SIZE} ...")

    # Завантаження батчами
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Upsert"):
        end = min(start + BATCH_SIZE, len(df))
        batch_df  = df.iloc[start:end]
        batch_emb = embeddings[start:end]

        vectors = []
        for offset, (_, row) in enumerate(batch_df.iterrows()):
            vectors.append({
                "id":       f"paper_{start + offset}",
                "values":   batch_emb[offset].tolist(),
                "metadata": build_metadata(row),
            })
        index.upsert(vectors=vectors)

    # Підсумок
    stats = index.describe_index_stats()
    print(f"\nГотово. Загальна кількість векторів в індексі: {stats['total_vector_count']}")


if __name__ == "__main__":
    main()
