import dotenv

import pandas as pd
from tqdm import tqdm
from config import get_config
from legacy.src.models.model_Mult_Attention import build_transformer
from legacy.src.utils.utils import KMerTokenizer, levenshtein_distance, clean_sequence
from legacy.src.utils.data_setup_balanced import AptamersDataset, causal_mask, collate_embeddings

import torch
from torch.utils.data import DataLoader
import os

from timeit import default_timer as timer

dotenv.load_dotenv(".env")

device = "cuda"
print(torch.cuda.get_device_name())

DATA_PATH = os.environ["DATA_PATH"]
OUTPUTS_PATH = os.environ["OUTPUTS_PATH"]
CHECKPOINTS_PATH = os.environ["CHECKPOINTS_PATH"]
#MLRUNS_PATH = os.environ["MLRUNS_PATH"]


embeddings_path = "/mnt/tank/scratch/azaikina/esm/embeds"
#embeddings_path = os.path.join(DATA_PATH, "mirna_embeds")
#changed data
df = pd.read_csv('/mnt/tank/scratch/azaikina/Model/data/test_ds_Transformer_MultAttention_finetune', index_col=0)
# # # df_path = os.path.join(DATA_PATH, '3_checked_intersections_180t.csv')
# # # df = pd.read_csv(df_path, index_col = 0)




#Убрать строки без сиквенсов
df = df.dropna(subset=['Aptamer Sequence'])
df = df[df["type"] == 'RNA']

#Без антитела, поэтому значения-заглушки
apt_seq_column = 'Aptamer Sequence'
apt_name_column = 'Name of Aptamer'
ab_name_column = 'Name of Antibody'
ab_seq_column = 'Antibody Sequence'
tg_name_column = 'Target_ab'
tg_seq_column = 'target_seq_ab'


#For test###############################################
#df = df[:1000]


####################################################################################################
config = get_config(config_name='config_finetune_Mult_4')
d_model = config['d_model']    #1280
max_len = config['seq_len']    #182
N = config['num_layers']   #2
h = config['num_heads']    #8
dropout = config['dropout']   #0.1
d_ff = config['d_ff']   #512
exp_name = config['experiment_name']
lr = config['lr']

tokenizer = KMerTokenizer(k = config['kmer'])



dataset = AptamersDataset(df=df, tokenizer=tokenizer, seq_len=config['seq_len'], embeddings_path = embeddings_path, 
                            apt_name_column = apt_name_column, apt_seq_column = apt_seq_column, tg_name_column = tg_name_column,
                            tg_seq_column = tg_seq_column, ab_name_column = ab_name_column, ab_seq_column = ab_seq_column
)

dataloader = DataLoader(
    dataset,
    shuffle=False,
    collate_fn=collate_embeddings
)

################################################



def inference(model: torch.nn.Transformer, 
              dataloader: torch.utils.data.DataLoader, 
              tokenizer,
              output_file = "inference_results.csv"):
    
    model.eval()
    results_table = []

    total_levenshtein = 0
    total_normalized_lev = 0
    count = 0

    progress_bar = tqdm(dataloader, total=len(dataloader), desc="Generating", leave=True)
    with torch.inference_mode():
        for batch in progress_bar:
            encoder_output = batch['embedding'].to(device)
            decoder_input = torch.tensor(batch['decoder_input']).to(device)
            decoder_mask = batch['decoder_mask'].to(device)
 
            if encoder_output.dim() == 2:
                encoder_output = encoder_output.unsqueeze(1)
            decoder_output = model.decode(encoder_output, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            pred_ids = proj_output.argmax(dim=-1).detach().cpu().numpy()
            model_out_text = [clean_sequence(tokenizer.decode(ids)) for ids in pred_ids]
            # Retrieving source and target texts from the batch
            #source_text = [batch['ab_name'], batch['tg_name'], batch['ab_seq'], batch['tg_seq']]
            #target_name = [batch['apt_name']]
            target_text = [clean_sequence(s) for s in batch['apt_seq']]

            B = len(batch["apt_seq"])
            model_out_text = model_out_text[:B]
            target_text     = target_text[:B]

            for i in range(B):   #for all sequences in dataloader
                pred_seq   = model_out_text[i]
                target_seq = target_text[i]

                if len(target_seq) == 0:
                    lev_dist = 182
                    normalized_lev = 1
                else:
                    lev_dist = levenshtein_distance(pred_seq, target_seq)
                    normalized_lev = lev_dist / max(1, len(target_seq))

                lev_dist = levenshtein_distance(pred_seq, target_seq)
                normalized_lev = lev_dist / max(1, len(target_seq))
                total_levenshtein += lev_dist
                total_normalized_lev += normalized_lev
                count += 1

                results_table.append({
                        "ab_name":    batch['ab_name'][i],
                        "tg_name":    batch['tg_name'][i],
                        "apt_name":   batch['apt_name'][i],
                        "ab_seq":     batch['ab_seq'][i],
                        "tg_seq":     batch['tg_seq'][i],
                        "embeddings_path": batch['tg_seq'][i],
                        "pred_seq":   pred_seq,
                        "target_seq": target_seq,
                        "lev_dist":   lev_dist,
                        "normalized_lev": normalized_lev
                    })
        
        avg_levenshtein = total_levenshtein / count if count else 0
        avg_normalized_lev = total_normalized_lev / count if count else 0
        
        df = pd.DataFrame(results_table)
        df.to_csv(output_file, index=False)

        print(f"\nSaved {len(df)} rows to {output_file}")
        print(f"Avg Levenshtein: {avg_levenshtein:.3f}")
        print(f"Avg Normalized Levenshtein: {avg_normalized_lev:.3f}")
        
        return df


def autoregressive_decode(model, encoder_output, tokenizer, max_len=181):
    sos = tokenizer.token_to_id("[SOS]")
    eos = tokenizer.token_to_id("[EOS]")

    decoder_input = torch.tensor([[sos]], device=encoder_output.device)

    for _ in range(max_len):
        mask = causal_mask(decoder_input.size(1)).to(encoder_output.device)

        out = model.decode(encoder_output, decoder_input, mask)
        out = model.project(out[:, -1])     # only last step
        next_id = out.argmax(dim=-1).unsqueeze(0)

        decoder_input = torch.cat([decoder_input, next_id], dim=1)

        if next_id.item() == eos:
            break

    return decoder_input.squeeze().tolist()


def autoregressive_inference(model: torch.nn.Transformer, 
              dataloader: torch.utils.data.DataLoader, 
              tokenizer,
              output_file = "inference_results.csv"):
    
    model.eval()
    results_table = []

    total_levenshtein = 0
    total_normalized_lev = 0
    count = 0

    progress_bar = tqdm(dataloader, total=len(dataloader), desc="Generating", leave=True)
    with torch.inference_mode():
        for batch in progress_bar:
            encoder_output = batch['embedding'].to(device)

            if encoder_output.dim() == 2:
                encoder_output = encoder_output.unsqueeze(1)

            pred_ids = []
            for i in range(encoder_output.size(0)):
                ids = autoregressive_decode(model, encoder_output[i:i+1], tokenizer)
                pred_ids.append(ids)
            model_out_text = [clean_sequence(tokenizer.decode(ids)) for ids in pred_ids]
            # Retrieving source and target texts from the batch
            #source_text = [batch['ab_name'], batch['tg_name'], batch['ab_seq'], batch['tg_seq']]
            #target_name = [batch['apt_name']]
            target_text = [clean_sequence(s) for s in batch['apt_seq']]

            B = len(batch["apt_seq"])
            model_out_text = model_out_text[:B]
            target_text     = target_text[:B]

            for i in range(B):   #for all sequences in dataloader
                pred_seq   = model_out_text[i]
                target_seq = target_text[i]

                if len(target_seq) == 0:
                    lev_dist = 182
                    normalized_lev = 1
                else:
                    lev_dist = levenshtein_distance(pred_seq, target_seq)
                    normalized_lev = lev_dist / max(1, len(target_seq))

                total_levenshtein += lev_dist
                total_normalized_lev += normalized_lev
                count += 1

                results_table.append({
                        "ab_name":    batch['ab_name'][i],
                        "tg_name":    batch['tg_name'][i],
                        "apt_name":   batch['apt_name'][i],
                        "ab_seq":     batch['ab_seq'][i],
                        "tg_seq":     batch['tg_seq'][i],
                        "embedding_path": batch['embedding_path'][i],
                        "pred_seq":   pred_seq,
                        "target_seq": target_seq,
                        "lev_dist":   lev_dist,
                        "normalized_lev": normalized_lev
                    })
        
        avg_levenshtein = total_levenshtein / count if count else 0
        avg_normalized_lev = total_normalized_lev / count if count else 0
        
        df = pd.DataFrame(results_table)
        df.to_csv(output_file, index=False)

        print(f"\nSaved {len(df)} rows to {output_file}")
        print(f"Avg Levenshtein: {avg_levenshtein:.3f}")
        print(f"Avg Normalized Levenshtein: {avg_normalized_lev:.3f}")
        
        return df


vocab_size= len(tokenizer)

tokens = tokenizer.encode("ACGTACGT")
print("Encoded:", tokens)
print("Decoded:", tokenizer.decode(tokens))

model = build_transformer(vocab_size, max_len, d_model, N, h, dropout, d_ff)
model_name = "Transformer_MultAttention_finetune_model"
model.load_state_dict(torch.load(f"/mnt/tank/scratch/azaikina/Model/outputs/checkpoints/{model_name}.pth"))  # загружаем веса
model.to(device)
output_file=f"/mnt/tank/scratch/azaikina/Model/outputs/generated_aptamers/inference_results_{model_name}_test_ds_auto.csv"


torch.manual_seed(42) 
torch.cuda.manual_seed(42)


start_time = timer()
# Train model_0 
model_results = autoregressive_inference(model=model, dataloader=dataloader,
                          tokenizer=tokenizer,
                          output_file=output_file
                          )

end_time = timer()
print(f"Total generation time: {end_time-start_time:.3f} seconds")