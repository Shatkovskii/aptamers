import pandas as pd

COLS = ["Antibody Sequence", "target_seq_ab", "target_seq_apt", "Aptamer Sequence"]

df = pd.read_csv("../data/3_checked_intersections_180t.csv", index_col=0)
total = len(df)

print(f"Total rows: {total}\n")

for col in COLS:
    n_none = df[col].isna().sum()
    print(f"  {col}: {n_none} None ({n_none / total:.1%})")

valid = df.dropna(subset=COLS)
print(f"\nRows with all 4 columns present: {len(valid)} / {total} ({len(valid) / total:.1%})")

print(f"\nUnique values:\n")
for col in COLS:
    n_unique = df[col].nunique()
    print(f"  {col}: {n_unique} unique")
