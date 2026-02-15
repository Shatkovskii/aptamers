import dotenv

import pandas as pd
import json
from tqdm import tqdm
from config import get_config
from legacy.src.models.model_Mult_Attention import build_transformer
from legacy.src.utils.utils import KMerTokenizer, save_model, visualize_mismatch, levenshtein_distance, clean_sequence
from legacy.src.utils.data_setup_balanced import AptamersDataset, collate_embeddings

import torch
from torch import nn
from torch.utils.data import DataLoader
import os

from legacy.src.utils.pytorch_balanced_sampler.sampler import SamplerFactory
from collections import defaultdict

import mlflow


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
df_path = os.path.join(DATA_PATH, '3_checked_intersections_180t.csv')
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
indices = torch.randperm(len(df)).tolist()
train_size = int(0.9 * len(df))

train_indices = indices[:train_size]

test_indices = indices[train_size:]

df.iloc[train_indices].to_csv(f'/mnt/tank/scratch/azaikina/Model/data/train_ds_{exp_name}')
df.iloc[test_indices].to_csv(f'/mnt/tank/scratch/azaikina/Model/data/test_ds_{exp_name}')
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
    empty_seq_warnings = 0
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

            pred_ids = proj_output.argmax(dim=-1).detach().cpu().tolist()
            model_out_text = [tokenizer.decode_tensors(ids) for ids in pred_ids]
            # Retrieving source and target texts from the batch
            #source_text = [batch['ab_name'], batch['tg_name'], batch['ab_seq'], batch['tg_seq']]
            #target_name = [batch['apt_name']]
            target_text = batch['apt_seq']
   
            for pred_seq, target_seq in zip(model_out_text, target_text):   #for all sequences in dataloader
                pred_seq = clean_sequence(pred_seq)
                target_seq = clean_sequence(target_seq)
                if len(pred_seq) == 0 or len(target_seq) == 0:
                    empty_seq_warnings += 1
                    tqdm.write(f"[Warning] Empty sequence: pred='{pred_seq}', target='{target_seq}'")
                    continue

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

        test_loss = test_loss / len(dataloader)
        avg_levenshtein = total_levenshtein / count if count > 0 else 0.0
        avg_normalized_lev = total_normalized_lev / count if count > 0 else 0.0


        mlflow.log_metric('Validation/Test_loss', test_loss, step=global_step)
        mlflow.log_metric('Validation/Levenshtein', avg_levenshtein, step=global_step)
        mlflow.log_metric('Validation/Normalized_Levenshtein', avg_normalized_lev, step=global_step)
        if empty_seq_warnings > 0:
            mlflow.log_metric('Empty_sequences', empty_seq_warnings, step=global_step)
        
        mismatch_file = f"{exp_name}_mismatch.txt"
        with open(mismatch_file, "a") as f:
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
    count = 0
    empty_seq_warnings = 0

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
        pred_ids = proj_output.argmax(dim=-1).detach().cpu().tolist()
        model_out_text = [tokenizer.decode_tensors(ids) for ids in pred_ids]
        
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
    avg_levenshtein = total_levenshtein / count if count > 0 else 0.0
    avg_normalized_lev = total_normalized_lev / count if count > 0 else 0.0
    mlflow.log_metric('Train/Train_loss', train_loss, step=global_step)
    mlflow.log_metric('Train/Levenshtein', avg_levenshtein, step=global_step)
    mlflow.log_metric('Train/Normalized_Levenshtein', avg_normalized_lev, step=global_step)
    if empty_seq_warnings > 0:
        mlflow.log_metric('Empty_sequences', empty_seq_warnings, step=global_step)
    return train_loss, avg_levenshtein, avg_normalized_lev, global_step

len_df = len(df)

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5
          ):
    mlflow.set_tracking_uri("file:///mnt/tank/scratch/azaikina/Model/outputs/mlruns")
    mlflow.set_experiment('new_Experiment')

    mismatch_file = f"{exp_name}_mismatch.txt"
    with open(mismatch_file, "w") as f:
        pass

    with mlflow.start_run(run_name="Fine-tune_run"):
        config_file = f"{exp_name}_config.txt"
        with open(config_file, "w") as f:                   # at the end of test_step write mismatch for visualization
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
max_len = config['seq_len']    #100
N = config['num_layers']   #2
h = config['num_heads']    #8
dropout = config['dropout']   #0.1
d_ff = config['d_ff']   #512
lr = config['lr']

model = build_transformer(vocab_size, max_len, d_model, N, h, dropout, d_ff)

model.load_state_dict(torch.load("/mnt/tank/scratch/azaikina/Model/outputs/checkpoints/Transformer_MultAttention_early_stopped.pth"))  # загружаем веса
model.to(device)


torch.manual_seed(42) 
torch.cuda.manual_seed(42)


loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

# Start the timer
from timeit import default_timer as timer 
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
