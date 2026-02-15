import torch
from torch.utils.data import Dataset
from pathlib import Path

import numpy as np
import pandas as pd
import re
from legacy.src.models.model_1 import causal_mask

class AptamersDataset(Dataset):
    """    
    Returns:
    A tuple out of:
        embedding (type torch.Tensor), 
        ab_name, apt_name, tg_name, 
        ab_seq, apt_seq, tg_seq
    """
    def __init__(self, df: pd.DataFrame, tokenizer, seq_len: int, embeddings_path: str,
                 ab_name_column: str = 'Name of Antibody', apt_name_column: str = 'Interactor1.Symbol', tg_name_column: str = 'Target_ab',
                 ab_seq_column: str = 'Antibody Sequence', apt_seq_column: str = 'Aptamer Sequence', tg_seq_column: str = 'target_seq_ab',
                 with_names: bool =True):

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

        # Mapping {embedding_number: file_path}
        all_files = list(Path(embeddings_path).glob("*.npy"))
        self.id_to_path = {}
        for f in all_files:
            match = re.search(r"(\d+)(?=\.npy$)", f.name)
            if match:
                emb_id = int(match.group(1))
                self.id_to_path[emb_id] = f

        embedding_ids = set(self.id_to_path.keys())
        df = df[df.index.isin(embedding_ids)].copy()
        # Drop rows with missing aptamer sequences
        df = df.dropna(subset=[apt_seq_column])
        df['original_index'] = df.index  # <- preserve original IDs before reset_index
        df = df.reset_index(drop=True)

        # Keep only DataFrame rows that have an embedding
        if len(df) < len(embedding_ids): 
            print(f"⚠️ {len(df) - len(embedding_ids)} rows skipped (no matching embedding).")


        unique_seqs = df[self.apt_seq_column].unique()
        seq_to_label = {seq: i for i, seq in enumerate(unique_seqs)}

        df['aptamer_class'] = df[self.apt_seq_column].map(seq_to_label)
        self.df = df

        self.apt_classes = torch.tensor(df['aptamer_class'].to_list())
        self.valid_indices = df['original_index'].to_list() # <- embedding IDs
        
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


        ab_name = self.df.loc[index, self.ab_name_column]
        apt_name = self.df.loc[index, self.apt_name_column]
        tg_name = self.df.loc[index, self.tg_name_column]
        
        ab_seq = self.df.loc[index, self.ab_seq_column]
        apt_seq = self.df.loc[index, self.apt_seq_column]
        tg_seq = self.df.loc[index, self.tg_seq_column]

        tokens = self.tokenizer.encode(apt_seq)  # already [SOS] ... [EOS]
        num_pad = self.seq_len - len(tokens)
        if num_pad < 0:
            print(f"⚠️ Sequence too long! Length: {len(tokens)}, allowed: {self.seq_len}")
            print(f"Problematic sequence (apt_seq): {apt_seq}")
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

        class_idx = int(self.apt_classes[index])

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
            'tg_seq': tg_seq,
            'class_idx': class_idx
            }            
        else:
            return embedding # return tensor
        

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
    class_idx = [item['class_idx'] for item in batch]

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
        'tg_seq': tg_seqs,
        'class_idx': class_idx
    }
