import os
from collections import defaultdict
from timeit import default_timer as timer 

import dotenv
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import mlflow

from config import get_config
from legacy.src.models.model_1_encode import build_transformer
from legacy.src.utils.utils import KMerTokenizer, save_model, visualize_mismatch, levenshtein_distance, clean_sequence
from legacy.src.utils.data_setup_balanced import AptamersDataset, collate_embeddings
from legacy.src.utils.pytorch_balanced_sampler.sampler import SamplerFactory

dotenv.load_dotenv(".env")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.get_device_name())

PYTHONPATH=rf"C:\Users\anna6\Aptamers_Model"
DATA_PATH=rf"C:\Users\anna6\Aptamers_Model\data"
OUTPUTS_PATH=rf"C:\Users\anna6\Aptamers_Model\outputs"
CHECKPOINTS_PATH=rf"C:\Users\anna6\Aptamers_Model\outputs\checkpoints"
MLRUNS_PATH=rf"C:\Users\anna6\Aptamers_Model\outputs\mlruns"
CONFIG_PATH=rf"C:\Users\anna6\Aptamers_Model\conf"

embeddings_path = rf"C:\Users\anna6\Aptamers_Model\data\esm"
#embeddings_path = os.path.join(DATA_PATH, "mirna_embeds")
DATA_PATH = Path('/c/Aptamers_Model/data')
df_path = DATA_PATH / '3_checked_intersections_180t.csv'
df_path = rf'C:\Users\anna6\Aptamers_Model\data\3_checked_intersections_180t.csv'

df = pd.read_csv(df_path, index_col = 0)


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
#df = df[:10000]


####################################################################################################
config = get_config(config_name='config_finetune_encode')
exp_name = config['experiment_name']


tokenizer = KMerTokenizer(k = config['kmer'])
indices = torch.randperm(len(df)).tolist()
train_size = int(0.9 * len(df))

train_indices = indices[:train_size]

test_indices = indices[train_size:]

# 4. Создание датасетов
train_ds = AptamersDataset(df=df.iloc[train_indices], tokenizer=tokenizer, seq_len=config['seq_len'], embeddings_path = embeddings_path, 
                            apt_name_column = apt_name_column, apt_seq_column = apt_seq_column, tg_name_column = tg_name_column,
                            tg_seq_column = tg_seq_column, ab_name_column = ab_name_column, ab_seq_column = ab_seq_column)


# print(train_ds)
test_ds = AptamersDataset(df=df.iloc[test_indices], tokenizer=tokenizer, seq_len=config['seq_len'], embeddings_path = embeddings_path, 
                            apt_name_column = apt_name_column, apt_seq_column = apt_seq_column, tg_name_column = tg_name_column,
                            tg_seq_column = tg_seq_column, ab_name_column = ab_name_column, ab_seq_column = ab_seq_column
)

# допустим, у тебя есть метки классов
apt_classes_train = train_ds.df['aptamer_class'].values  # shape (N,)
class_idxs_dict_train = defaultdict(list)

# группируем индексы по классам
for idx, cls in enumerate(apt_classes_train):
    print(idx, cls)
    class_idxs_dict_train[int(cls)].append(idx)

class_idxs_train = list(class_idxs_dict_train.values())  # По формату нужен список списков

# допустим, у тебя есть метки классов
apt_classes_test = test_ds.df['aptamer_class'].values  # shape (N,)
class_idxs_dict_test = defaultdict(list)

# группируем индексы по классам
for idx, cls in enumerate(apt_classes_test):
    print(idx, cls)
    class_idxs_dict_test[int(cls)].append(idx)

class_idxs_test = list(class_idxs_dict_test.values())  # По формату нужен список списков

print(f"Number of classes: {len(class_idxs_train)}")
for i, class_indices in enumerate(class_idxs_train):
    print(f"Class {i}: {len(class_indices)} samples")

n_train_batches = len(train_ds) // config['batch_size']
n_test_batches = len(test_ds) // config['batch_size']

train_sampler = SamplerFactory().get(
    class_idxs=class_idxs_train,
    batch_size=config['batch_size'],
    n_batches=n_train_batches,
    alpha=1,  # Balance parameter (0.0 = no balance, 1.0 = perfect balance)
    kind='random'  # 'fixed' or 'random'
)

test_sampler = SamplerFactory().get(
    class_idxs=class_idxs_test,
    batch_size=config['batch_size'],
    n_batches=n_test_batches,
    alpha=1,
    kind='random'
)


# 5. Создание DataLoader'ов
train_dataloader = DataLoader(
    train_ds,
    shuffle=False,
    collate_fn=collate_embeddings,
    batch_sampler = train_sampler
)

test_dataloader = DataLoader(
    test_ds,
    shuffle=False,
    collate_fn=collate_embeddings,
    batch_sampler = test_sampler
)


def test_step(model: torch.nn.Transformer, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              global_step:int = None,
              epoch: int = None):
    model.eval()

    test_loss = 0
    total_levenshtein = 0
    total_normalized_lev = 0
    all_sequences_list = []
    count = 0
    mismatch_examples = []
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
            
            
            for pred_seq, target_seq in zip(model_out_text, target_text):   #for all sequences in dataloader
                pred_seq = clean_sequence(pred_seq)
                target_seq = clean_sequence(target_seq)

                lev_dist = levenshtein_distance(pred_seq, target_seq)
                normalized_lev = lev_dist / len(target_seq)
                total_levenshtein += lev_dist
                total_normalized_lev += normalized_lev
                count += 1

                if len(mismatch_examples) < 5:
                    mismatch_str = visualize_mismatch(target_seq, pred_seq)
                    mismatch_examples.append(mismatch_str)
            


            loss = loss_fn(proj_output.view(-1, len(tokenizer)), label.view(-1))
            tqdm.write(f'loss {loss}')
            all_sequences_list.append(model_out_text)
            test_loss += loss.item()

            # ---- собираем 5 примеров mismatch ----
            if len(mismatch_examples) < 5:
                mismatch_str = visualize_mismatch(target_seq, pred_seq)
                mismatch_examples.append(mismatch_str)

        test_loss = test_loss / len(dataloader)
        avg_levenshtein = total_levenshtein / len(dataloader)
        avg_normalized_lev = total_normalized_lev / len(dataloader)


        mlflow.log_metric('Validation/Test_loss', test_loss, step=global_step)
        mlflow.log_metric('Validation/Levenshtein', avg_levenshtein, step=global_step)
        mlflow.log_metric('Validation/Normalized_Levenshtein', avg_normalized_lev, step=global_step)

        # ---- записываем все 5 примеров mismatch ----
        with open(f"{exp_name}_mismatch.txt", "a") as f:
            f.write(f"Epoch {epoch}, Step {global_step}\n")
            for i, mismatch_str in enumerate(mismatch_examples, 1):
                f.write(f"\nExample {i}:\n{mismatch_str}\n")
            f.write("\n" + "="*80 + "\n\n")
        mlflow.log_artifact(f"{exp_name}_mismatch.txt")
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
    count = 0

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

        for pred_seq, target_seq in zip(model_out_text, target_text):   #for all sequences in dataloader
            pred_seq = clean_sequence(pred_seq)
            target_seq = clean_sequence(target_seq)
            if len(pred_seq) == 0 or len(target_seq) == 0:
                empty_seq_warnings += 1
                tqdm.write(f"[Warning] Empty sequence at batch {i}: pred='{pred_seq}', target='{target_seq}'")
                continue
        
            lev_dist = levenshtein_distance(pred_seq, target_seq)
            normalized_lev = lev_dist / len(target_seq)

            total_levenshtein += lev_dist
            total_normalized_lev += normalized_lev
            count += 1



        label = torch.tensor(batch['label']).to(device)
        loss = loss_fn(proj_output.view(-1, len(tokenizer)), label.view(-1))
        print('loss', loss)
        
        optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        mlflow.log_metric("Train/Grad_norm", total_norm, step=global_step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        #print(f"[Step {global_step}] Grad norm = {total_norm:.3f}")
        mlflow.log_metric("Train/Grad_norm_after_clip", total_norm, step=global_step)
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

len_df = len(df)

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    mlflow.set_tracking_uri(rf"file:///C:/Users/anna6/Aptamers_Model/outputs/mlruns")
    mlflow.set_experiment('encoder_Experiment')
    with mlflow.start_run(run_name="Experiment_run"):
        with open(f"{exp_name}_config.txt", "a") as f:                   # at the end of test_step write mismatch for visualization
            json.dump(config, f, indent=4)
        mlflow.log_artifact(f"{exp_name}_config.txt")
        mlflow.log_metric('Length of df', len_df)

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
                loss_fn=loss_fn, global_step=global_step, epoch=epoch)
            

            results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
            results["train_avg_levenshtein"].append(train_avg_levenshtein.item() if isinstance(train_avg_levenshtein, torch.Tensor) else train_avg_levenshtein)
            results["train_normalized_levenshtein"].append(train_normalized_levenshtein.item() if isinstance(train_normalized_levenshtein, torch.Tensor) else train_normalized_levenshtein)
            results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
            results["test_avg_levenshtein"].append(test_avg_levenshtein.item() if isinstance(test_avg_levenshtein, torch.Tensor) else test_avg_levenshtein)
            results["test_normalized_levenshtein"].append(test_normalized_levenshtein.item() if isinstance(test_normalized_levenshtein, torch.Tensor) else test_normalized_levenshtein)
            if epoch % config['save_every']== 0:
                save_model(model=model, target_dir=CHECKPOINTS_PATH, model_name=f'{exp_name}_model.pth')

    # 6. Return the filled results at the end of the epochs
    return results

vocab_size= len(tokenizer)

d_model = config['d_model']    #1280
max_len = config['seq_len']    #182
N = config['num_layers']   #2
h = config['num_heads']    #8
dropout = config['dropout']   #0.1
d_ff = config['d_ff']   #512
encoder_hidden_dim = config['encoder_hidden_dim']

model = build_transformer(vocab_size, max_len, d_model, N, h, dropout, d_ff, encoder_hidden_dim)
model.load_state_dict(torch.load(rf"C:\Users\anna6\Aptamers_Model\outputs\checkpoints\Transformer_with_mirna_encode_early_stopped.pth"))  # загружаем веса
model.to(device)


torch.manual_seed(42) 
torch.cuda.manual_seed(42)


loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Start the timer
start_time = timer()

# Train model_0 
model_results = train(model=model, 
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=config['num_epochs'])


results_df = pd.DataFrame(model_results)
results_df.to_csv(os.path.join(OUTPUTS_PATH, f"{exp_name}_training_results.csv"), index=False)
print(f"Training results saved to {exp_name}_training_results.csv")

end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
