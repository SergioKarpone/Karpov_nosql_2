import json
import os
import pandas as pd
from tqdm import tqdm

INPUT_FILE  = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "data/arxiv_subset.parquet"
MAX_RECORDS = 10_000

os.makedirs("data", exist_ok=True)


def extract_year(paper: dict) -> int:
    """Рік першої версії статті (дата публікації на arXiv)."""
    try:
        versions = paper.get("versions", [])
        if versions:
            created = versions[0]["created"]  # "Mon, 2 Apr 2007 19:18:42 GMT"
            return int(created.split()[3])
    except (IndexError, ValueError, KeyError):
        pass
    return int(paper.get("update_date", "2000-01-01")[:4])


def format_authors(paper: dict) -> str:
    """authors_parsed → "Прізвище І., Прізвище І."."""
    parsed = paper.get("authors_parsed", [])
    if parsed:
        parts = []
        for entry in parsed[:10]:
            last  = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                parts.append(f"{last} {first}".strip())
        return ", ".join(parts)
    return paper.get("authors", "").replace("\n", " ")


records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Читаємо датасет"):
        if len(records) >= MAX_RECORDS:
            break
        line = line.strip()
        if not line:
            continue
        paper = json.loads(line)

        abstract = paper.get("abstract", "").strip()
        title    = paper.get("title", "").strip()
        if not abstract or not title:
            continue

        categories_raw = paper.get("categories", "unknown")
        primary_category = categories_raw.split()[0]

        records.append({
            "id":       paper["id"],
            "title":    title.replace("\n", " ").strip(),
            "abstract": abstract.replace("\n", " ").strip(),
            "authors":  format_authors(paper),
            "year":     extract_year(paper),
            "category": primary_category,
        })

df = pd.DataFrame(records)
print(f"\nЗавантажено статей: {len(df)}")
print(f"\nРозподіл за категоріями (топ-10):")
print(df["category"].value_counts().head(10))
print(f"\nРозподіл за роками:")
print(df["year"].value_counts().sort_index().tail(10))
print(f"\nПриклад запису:")
print(df.iloc[0].to_dict())

df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nЗбережено в {OUTPUT_FILE}")
