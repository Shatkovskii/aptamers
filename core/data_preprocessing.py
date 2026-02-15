"""
Preprocessing pipeline for the aptamer–antibody interaction dataset.

* Loads the raw CSV produced by the intersection-checking step.
* Removes exact duplicates based on the four sequence / ID columns that
  uniquely define an (antibody, aptamer) pair:
      Antibody Sequence, target_seq_ab, target_seq_apt, Aptamer Sequence
* Saves the deduplicated dataset and prints statistics.
"""

import os
from pathlib import Path

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "3_checked_intersections_180t.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "preprocessed_data.csv"

# Columns that together define a unique (antibody, aptamer) pair
DEDUP_COLUMNS = [
    "Antibody Sequence",
    "target_seq_ab",
    "target_seq_apt",
    "Aptamer Sequence",
]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are duplicates with respect to *DEDUP_COLUMNS*.

    Keeps the first occurrence and removes subsequent ones.
    Prints the number of duplicates found and the dataset size change.
    """
    n_before = len(df)
    n_duplicates = df.duplicated(subset=DEDUP_COLUMNS, keep="first").sum()

    df_deduped = df.drop_duplicates(subset=DEDUP_COLUMNS, keep="first")
    n_after = len(df_deduped)

    print(f"Размер датасета ДО удаления дубликатов : {n_before}")
    print(f"Найдено дубликатов                     : {n_duplicates}")
    print(f"Размер датасета ПОСЛЕ удаления         : {n_after}")
    print(f"Удалено строк                          : {n_before - n_after}")

    return df_deduped


def main() -> None:
    # ── load ─────────────────────────────────────────────────────────────
    print(f"Загрузка данных из {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Колонки: {list(df.columns)}\n")

    # ── deduplicate ──────────────────────────────────────────────────────
    df_clean = remove_duplicates(df)

    # ── save ─────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\nОчищенный датасет сохранён в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
