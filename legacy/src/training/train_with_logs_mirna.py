import mlflow
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from utils.utils import KMerTokenizer, visualize_mismatch, levenshtein_distance
from utils.data_setup import AptamersDataset, collate_embeddings
from legacy.src.models.model_1 import build_transformer
from utils import get_config


device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings_path = "/mnt/tank/scratch/azaikina/esm/mirna_embeds"
df_path = '/mnt/tank/scratch/azaikina/Model/new_scripts_5_10/data/mirbase_clean.csv'
df = pd.read_csv(df_path, index_col = 0)

#Убрать строки без сиквенсов
df = df.dropna(subset=['mirna_sequence'])

#Без антитела, поэтому значения-заглушки
df['ab_name_column'] = '-'
df['ab_seq_column'] = '-'
apt_seq_column = 'mirna_sequence'
apt_name_column = 'Interactor1.Symbol_x'
ab_name_column = 'ab_name_column'
ab_seq_column = 'ab_seq_column'
tg_name_column = 'Interactor2.Symbol_x'
tg_seq_column = 'Protein_Sequence'



#For test###############################################
df = df[:10000]


####################################################################################################
config = get_config()

tokenizer = KMerTokenizer(k = config['kmer'])
dataset = AptamersDataset(df=df, tokenizer=tokenizer, seq_len=config['seq_len'], embeddings_path = embeddings_path, 
                            apt_name_column = apt_name_column, apt_seq_column = apt_seq_column, tg_name_column = tg_name_column,
                            tg_seq_column = tg_seq_column, ab_name_column = ab_name_column, ab_seq_column = ab_seq_column)


# Define split sizes
train_size = int(0.9 * len(dataset))  # 90% for training
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader_custom = DataLoader(dataset=train_dataset,
                                     batch_size=config['batch_size'],
                                     shuffle=True,
                                     collate_fn=collate_embeddings)

test_dataloader_custom = DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    collate_fn=collate_embeddings)


def test_step(model: torch.nn.Transformer, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              global_step:int = None):
    model.eval()

    test_loss = 0
    total_levenshtein = 0
    total_normalized_lev = 0
    all_sequences_list = []
    count = 0
    progress_bar = tqdm(dataloader, total=len(dataloader), desc="Testing", leave=True)
    with torch.inference_mode():
        for batch in progress_bar:
            encoder_output = batch['embedding'].to(device)
            decoder_input = torch.tensor(batch['decoder_input']).to(device)
            decoder_mask = batch['decoder_mask'].to(device)
 
            if encoder_output.dim() == 2:
                encoder_output = encoder_output.unsqueeze(1)
            decoder_output = model.decode(encoder_output, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)


            label = torch.tensor(batch['label']).to(device)

            pred_ids = proj_output.argmax(dim=-1).detach().cpu().numpy()
            model_out_text = [tokenizer.decode(ids) for ids in pred_ids]
            # Retrieving source and target texts from the batch
            source_text = [batch['ab_name'], batch['tg_name'], batch['ab_seq'], batch['tg_seq']]
            target_name = [batch['apt_name']]
            target_text = batch['apt_seq']
   

            tqdm.write(f'SOURCE: {source_text}')
            tqdm.write(f'TARGET: {target_text}')
            tqdm.write(f'TARGET NAME: {target_name}')
            tqdm.write(f'PREDICTED: {model_out_text}')
            

            loss = loss_fn(proj_output.view(-1, len(tokenizer)), label.view(-1))
            tqdm.write(f'loss {loss}')

            pred_seq = model_out_text[0]  # get the first (and only) sequence
            target_seq = target_text[0]

            lev_dist = levenshtein_distance(pred_seq, target_seq)
            normalized_lev = lev_dist / len(target_seq)

            total_levenshtein += lev_dist
            total_normalized_lev += normalized_lev

            tqdm.write(f'LEVENSTEIN:{lev_dist}')
            tqdm.write(f'NORM_LEVENSTEIN:{normalized_lev:.2f}')
            count += 1
            all_sequences_list.append(model_out_text)
            test_loss += loss.item()

        test_loss = test_loss / len(dataloader)
        avg_levenshtein = total_levenshtein / len(dataloader)
        avg_normalized_lev = total_normalized_lev / len(dataloader)

        mlflow.log_artifact("mismatch.txt")
        mlflow.log_metric('Validation/Test_loss', test_loss, step=global_step)
        mlflow.log_metric('Validation/Levenshtein', avg_levenshtein, step=global_step)
        mlflow.log_metric('Validation/Normalized_Levenshtein', avg_normalized_lev, step=global_step)

        mismatch_str = visualize_mismatch(target_seq, pred_seq)
        with open("mismatch.txt", "a") as f:                   # at the end of test_step write mismatch for visualization
            f.write(f"Step {global_step}\n{mismatch_str}\n\n")

        return test_loss, avg_levenshtein, avg_normalized_lev
    
def train_step(model: torch.nn.Transformer, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               global_step: int = None):
    model.train()
    

    train_loss = 0
    total_levenshtein = 0
    total_normalized_lev = 0
    total_sequences = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=True)
    # Loop through data loader data batches
    for i, batch in progress_bar:
        encoder_output = batch['embedding'].to(device)
        decoder_input = torch.tensor(batch['decoder_input']).to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        target_text = batch['apt_seq']
        
        
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1)
        decoder_output = model.decode(encoder_output, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)
        pred_ids = proj_output.argmax(dim=-1).detach().cpu().numpy()
        model_out_text = [tokenizer.decode(ids) for ids in pred_ids]
        pred_seq = model_out_text[0]
        target_seq = target_text[0]


        label = torch.tensor(batch['label']).to(device)
        loss = loss_fn(proj_output.view(-1, len(tokenizer)), label.view(-1))
        print('loss', loss)
        lev_dist = levenshtein_distance(pred_seq, target_seq)
        normalized_lev = lev_dist / len(target_seq)

        total_levenshtein += lev_dist
        total_normalized_lev += normalized_lev
        total_sequences += 1
        tqdm.write(f'LEVENSTEIN:{lev_dist}')
        tqdm.write(f'NORM_LEVENSTEIN:{normalized_lev:.2f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        global_step += 1

    train_loss = train_loss / len(dataloader)
    avg_levenshtein = total_levenshtein / len(dataloader)
    avg_normalized_lev = total_normalized_lev / len(dataloader)
    mlflow.log_metric('Train/Train_loss', train_loss, step=global_step)
    mlflow.log_metric('Train/Levenshtein', avg_levenshtein, step=global_step)
    mlflow.log_metric('Train/Normalized_Levenshtein', avg_normalized_lev, step=global_step)
    return train_loss, avg_levenshtein, avg_normalized_lev, global_step


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    mlflow.set_tracking_uri("/mnt/tank/scratch/azaikina/Model/mlruns")
    mlflow.set_experiment('Experiment')
    with mlflow.start_run(run_name="Experiment_run"):

        results = {"train_loss": [],
            "test_loss": [],
            "train_avg_levenshtein": [],
            "train_normalized_levenshtein": [],
            "test_avg_levenshtein": [],
            "test_normalized_levenshtein": []
        }
        global_step = 0
        #Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            train_loss, train_avg_levenshtein, train_normalized_levenshtein, global_step = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer, global_step= global_step)
            test_loss, test_avg_levenshtein, test_normalized_levenshtein = test_step(model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn, global_step=global_step)
            

            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"test_loss: {test_loss:.4f} | " 
                f"train_avg_levenshtein: {train_avg_levenshtein} | "
                f"train_normalized_levenshtein: {train_normalized_levenshtein} | "
                f"test_avg_levenshtein: {test_avg_levenshtein} | "  
                f"test_normalized_levenshtein: {test_normalized_levenshtein} | "    
            )

            results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
            results["train_avg_levenshtein"].append(train_avg_levenshtein.item() if isinstance(train_avg_levenshtein, torch.Tensor) else train_avg_levenshtein)
            results["train_normalized_levenshtein"].append(train_normalized_levenshtein.item() if isinstance(train_normalized_levenshtein, torch.Tensor) else train_normalized_levenshtein)
            results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
            results["test_avg_levenshtein"].append(test_avg_levenshtein.item() if isinstance(test_avg_levenshtein, torch.Tensor) else test_avg_levenshtein)
            results["test_normalized_levenshtein"].append(test_normalized_levenshtein.item() if isinstance(test_normalized_levenshtein, torch.Tensor) else test_normalized_levenshtein)

    # 6. Return the filled results at the end of the epochs
    return results

vocab_size= len(tokenizer)

d_model = config['d_model']    #1280
max_len = config['seq_len']    #100
N = config['num_layers']   #2
h = config['num_heads']    #8
dropout = config['dropout']   #0.1
d_ff = config['d_ff']   #512



model = build_transformer(vocab_size, max_len, d_model, N, h, dropout, d_ff)
model.to(device)


torch.manual_seed(42) 
torch.cuda.manual_seed(42)


loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Train model_0 
model_results = train(model=model, 
                        train_dataloader=train_dataloader_custom,
                        test_dataloader=test_dataloader_custom,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=config['num_epochs'])


results_df = pd.DataFrame(model_results)
results_df.to_csv("training_results.csv", index=False)
print("Training results saved to training_results.csv")

end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
