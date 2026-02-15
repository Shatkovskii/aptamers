import pandas as pd

df = pd.read_csv("../data/3_checked_intersections_180t.csv", index_col=0)

print(f"DF initial shape: {df.shape}")


def drop_nones_from_column(data: pd.DataFrame, col_name: str | list) -> pd.DataFrame:

    if isinstance(col_name, str):
        col_set = [col_name]
    else:
        col_set = col_name

    new_data = data.dropna(subset=col_set)

    for i, line in new_data.iterrows():
        for col_name in col_set:
            assert line[col_name] not in [None, False, ""]
            assert bool(line[col_name]) is True

    return new_data


df_1 = drop_nones_from_column(data=df, col_name="Aptamer Sequence")
print(f"Without NONE in Aptamer Sequence: {df_1.shape}")

df_2 = drop_nones_from_column(data=df, col_name="Antibody Sequence")
print(f"Without NONE in Antibody Sequence: {df_2.shape}")

df_3 = drop_nones_from_column(data=df, col_name="target_seq_ab")
print(f"Without NONE in target_seq_ab: {df_3.shape}")

df_4 = drop_nones_from_column(data=df, col_name="target_seq_apt")
print(f"Without NONE in target_seq_apt: {df_4.shape}")


df_5 = drop_nones_from_column(data=df, col_name=["Aptamer Sequence", "Antibody Sequence", "target_seq_ab", "target_seq_apt"])
print(f"Without NONE seqs: {df_5.shape}")
