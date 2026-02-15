import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import random
import re

#set embeddings and sequences paths 
data_path = Path("../data/")
#embeddings_path = Path("../../esm/embeds")
embeddings_path = Path.cwd().parent.parent / "esm" / "embeds"  #/mnt/tank/scratch/azaikina/esm/embeds

df_path = data_path / '3_checked_intersections.csv'
df = pd.read_csv(df_path, index_col = 0)

class AptamersDataset(Dataset):
    """    
    Returns:
    A tuple out of:
        embedding (type torch.Tensor), 
        ab_name, apt_name, tg_name, 
        ab_seq, apt_seq, tg_seq
    """
    def __init__(self, df: pd.DataFrame, embeddings_path: str,
                 ab_name_column: str = 'Name of Antibody', apt_name_column: str = 'Name of Aptamer', tg_name_column: str = 'Target_ab',
                 ab_seq_column: str = 'Antibody Sequence', apt_seq_column: str = 'Aptamer Sequence', tg_seq_column: str = 'target_seq_ab',
                 with_names: bool =True):
        self.df = df
        self.embeddings_path = embeddings_path

        self.ab_name_column = ab_name_column
        self.apt_name_column = apt_name_column
        self.tg_name_column = tg_name_column
        self.ab_seq_column = ab_seq_column
        self.apt_seq_column = apt_seq_column
        self.tg_seq_column = tg_seq_column
        self.with_names = with_names

        #Mapping {embedding_number: file_path}
        all_files = list(Path(embeddings_path).glob("*.npy"))
        self.id_to_path = {}
        for f in all_files:
            match = re.search(r"(\d+)(?=\.npy$)", f.name)
            if match:
                emb_id = int(match.group(1))
                self.id_to_path[emb_id] = f

        # --- Keep only DataFrame rows that have an embedding ---
        self.valid_indices = [i for i in self.df.index if i in self.id_to_path]
        if len(self.valid_indices) < len(self.df):
            print(f"⚠️ {len(self.df) - len(self.valid_indices)} rows skipped (no embedding file found).")

    def load_embedding(self, emb_index: int, dtype = torch.float32) -> torch.Tensor:
        "Transforms numpy to torch.Tensor."
        npy_path = self.id_to_path[emb_index]
        
        npy_embed = np.load(npy_path)
        tensor = torch.from_numpy(npy_embed).type(dtype)
        return tensor

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.valid_indices)
    
    def __getitem__(self, index: int):
        emb_index = self.valid_indices[index]
        embedding = self.load_embedding(emb_index)
        embedding_path = self.id_to_path[emb_index]


        ab_name = self.df.loc[emb_index, self.ab_name_column]
        apt_name = self.df.loc[emb_index, self.apt_name_column]
        tg_name = self.df.loc[emb_index, self.tg_name_column]
        
        ab_seq = self.df.loc[emb_index, self.ab_seq_column]
        apt_seq = self.df.loc[emb_index, self.apt_seq_column]
        tg_seq = self.df.loc[emb_index, self.tg_seq_column]

        if self.with_names:
            return embedding, embedding_path, ab_name, apt_name, tg_name, ab_seq, apt_seq, tg_seq
        else:
            return embedding # return tensor
    
    def get_by_id(self, emb_id: int):
        """Fetch a sample by its true embedding ID (DataFrame index)"""
        if emb_id not in self.id_to_path:
            raise KeyError(f"No embedding found for ID {emb_id}")

        embedding = self.load_embedding(emb_id)
        embedding_path = self.id_to_path[emb_id]


        ab_name = self.df.loc[emb_id, self.ab_name_column]
        apt_name = self.df.loc[emb_id, self.apt_name_column]
        tg_name = self.df.loc[emb_id, self.tg_name_column]
        
        ab_seq = self.df.loc[emb_id, self.ab_seq_column]
        apt_seq = self.df.loc[emb_id, self.apt_seq_column]
        tg_seq = self.df.loc[emb_id, self.tg_seq_column]

        if self.with_names:
            return embedding, embedding_path, ab_name, apt_name, tg_name, ab_seq, apt_seq, tg_seq
        else:
            return embedding
        
def collate_embeddings(batch):
    """
    DataLoader cannot batch Path objects or strings by default. 
    Collate function to batch embeddings (torch.Tensor) and keep other fields as lists.
    """
    embeddings = torch.stack([item[0] for item in batch])  # batch tensor
    paths = [item[1] for item in batch]                   # list of Paths
    ab_names = [item[2] for item in batch]
    apt_names = [item[3] for item in batch]
    tg_names = [item[4] for item in batch]
    ab_seqs = [item[5] for item in batch]
    apt_seqs = [item[6] for item in batch]
    tg_seqs = [item[7] for item in batch]

    return embeddings, paths, ab_names, apt_names, tg_names, ab_seqs, apt_seqs, tg_seqs

