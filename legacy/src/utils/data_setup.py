import torch
from torch.utils.data import Dataset
from pathlib import Path

import numpy as np
import pandas as pd
import re
from legacy.src.models.model_1 import causal_mask

#set embeddings and sequences paths 
data_path = Path("../data/")
#embeddings_path = Path("../../esm/embeds")
embeddings_path = '/mnt/tank/scratch/azaikina/esm/embeds'  #/mnt/tank/scratch/azaikina/esm/embeds

df_path = '/mnt/tank/scratch/azaikina/Model/data/3_checked_intersections.csv'
df = pd.read_csv(df_path, index_col = 0)

class AptamersDataset(Dataset):
    """    
    Returns:
    A tuple out of:
        embedding (type torch.Tensor), 
        ab_name, apt_name, tg_name, 
        ab_seq, apt_seq, tg_seq
    """
    def __init__(self, df: pd.DataFrame, tokenizer, seq_len: int, embeddings_path: str,
                 ab_name_column: str = 'Name of Antibody', apt_name_column: str = 'Name of Aptamer', tg_name_column: str = 'Target_ab',
                 ab_seq_column: str = 'Antibody Sequence', apt_seq_column: str = 'Aptamer Sequence', tg_seq_column: str = 'target_seq_ab',
                 with_names: bool =True):
        self.df = df
        self.tokenizer = tokenizer
        self.seq_len = seq_len
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

        tokens = self.tokenizer.encode(apt_seq)  # already [SOS] ... [EOS]
        num_pad = self.seq_len - len(tokens)
        if num_pad < 0:
            raise ValueError("Sequence too long")

        # decoder_input = tokens[:-1] (everything except final EOS)
        decoder_input = torch.cat([
            torch.tensor(tokens[:-1], dtype=torch.long),
            torch.tensor([self.tokenizer.pad_id] * num_pad, dtype=torch.long)
        ])

        # label = tokens[1:] (everything except SOS)
        label = torch.cat([
            torch.tensor(tokens[1:], dtype=torch.long),
            torch.tensor([self.tokenizer.pad_id] * num_pad, dtype=torch.long)
        ])
        
        seq_len_for_causal_mask = decoder_input.size(0)
        # masks
        mask = (decoder_input != self.tokenizer.pad_id).unsqueeze(0).unsqueeze(0).int()
        causal = causal_mask(seq_len_for_causal_mask)
        decoder_mask = mask & causal

        if self.with_names:
            return   {
            'embedding': embedding,
            'embedding_path': embedding_path,
            'decoder_input': decoder_input,
            'decoder_mask': decoder_mask,
            'label': label,
            'ab_name': ab_name,
            'apt_name': apt_name,
            'tg_name': tg_name,
            'ab_seq': ab_seq,
            'apt_seq': apt_seq,
            'tg_seq': tg_seq}            
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
    Collate function for dict-based dataset samples.
    Stacks tensors and groups other fields into lists.
    """
    # Stack tensors
    embeddings = torch.stack([item['embedding'] for item in batch])
    decoder_inputs = torch.stack([item['decoder_input'] for item in batch])
    decoder_masks = torch.stack([item['decoder_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # Non-tensor metadata
    embedding_paths = [item['embedding_path'] for item in batch]
    ab_names = [item['ab_name'] for item in batch]
    apt_names = [item['apt_name'] for item in batch]
    tg_names = [item['tg_name'] for item in batch]
    ab_seqs = [item['ab_seq'] for item in batch]
    apt_seqs = [item['apt_seq'] for item in batch]
    tg_seqs = [item['tg_seq'] for item in batch]

    return {
        'embedding': embeddings,
        'embedding_path': embedding_paths,
        'decoder_input': decoder_inputs,
        'decoder_mask': decoder_masks,
        'label': labels,
        'ab_name': ab_names,
        'apt_name': apt_names,
        'tg_name': tg_names,
        'ab_seq': ab_seqs,
        'apt_seq': apt_seqs,
        'tg_seq': tg_seqs
    }
