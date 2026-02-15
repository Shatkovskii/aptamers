import torch
import esm
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv(rf"/data/3_checked_intersections.csv", index_col=0)
# Filter out rows with missing sequences, but keep original indices
valid_df = df[~df['target_seq_ab'].isna()]

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval()  # disables dropout for deterministic results

manifest = []

print('with torch.no_grad():')
# Extract per-residue representations (on CPU)
with torch.no_grad():
    for idx, label, seq in tqdm(zip(valid_df.index, valid_df["Target_ab"], valid_df["target_seq_ab"]),
                            total=len(valid_df)):
        tqdm.write(f"{idx} {seq}")
        batch_labels, batch_strs, batch_tokens = batch_converter([(str(label), str(seq))])
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        sequence_representation = token_representations[0, 1:batch_lens[0]-1].mean(0)  #0 - select one sequence, 1: - skip first token, -1 - skip last token, batch_lens[0] - length of the sequence, .mean(0) - over the sequence length dimension
        sequence_representation = sequence_representation.detach().cpu().numpy()
        filepath = f'/mnt/tank/scratch/azaikina/esm/embeds/esm_embed_{idx}.npy'
        np.save(filepath, sequence_representation)
        manifest.append({'idx': idx, 'label': label, 'filepath': filepath})

manifest_df = pd.DataFrame(manifest)
manifest_df.to_csv('/mnt/tank/scratch/azaikina/esm/embeds/embedding_manifest.csv', index=False)

