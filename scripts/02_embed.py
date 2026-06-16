# 2. Перетворення текстів (title + abstract) у вектори моделлю
# allenai/specter2_base і збереження їх у embeddings/embeddings.npy.

import os
os.environ["USE_TF"] = "0"       # не вантажити TensorFlow через transformers
os.environ["USE_TORCH"] = "1"    # використовувати PyTorch-бекенд

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE  = "data/arxiv_subset.parquet"
OUTPUT_DIR  = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")

MODEL_NAME  = "allenai/specter2_base"
BATCH_SIZE  = 64
EXPECTED_DIM = 768

# Об'єднуємо заголовок і анотацію у форматі, на якому навчена SPECTER2:
# title + " [SEP] " + abstract
# Токен [SEP] обов'язковий — модель розрізняє два сегменти саме по ньому.

def build_texts(df: pd.DataFrame) -> list[str]:
    titles    = df["title"].fillna("").astype(str)
    abstracts = df["abstract"].fillna("").astype(str)
    return (titles + " [SEP] " + abstracts).tolist()


def main() -> None:
    # 1. Завантажуємо підготовлений датасет
    df = pd.read_parquet(INPUT_FILE)
    print(f"Завантажено записів із {INPUT_FILE}: {len(df)}")

    # 2. Готуємо тексти
    texts = build_texts(df)

    # 3. Завантажуємо модель
    print(f"Завантажуємо модель {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Кодуємо всі тексти у вектори
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # одинична довжина → cosine == dot
        convert_to_numpy=True,
    ).astype(np.float32)

    # 5. Діагностика
    print(f"\nОброблено текстів:        {len(texts)}")
    print(f"Розмірність ембеддингів:  {embeddings.shape[1]}  (очікується {EXPECTED_DIM})")
    print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.6f}  (≈ 1.0)")

    if embeddings.shape[1] != EXPECTED_DIM:
        print("⚠️  УВАГА: розмірність не збігається з очікуваною — перевірте модель.")

    # 6. Зберігання
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)
    print(f"\nЗбережено {embeddings.shape} → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
