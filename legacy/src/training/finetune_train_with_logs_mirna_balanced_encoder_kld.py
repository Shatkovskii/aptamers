import os
from collections import defaultdict
from timeit import default_timer as timer 

import dotenv
import pandas as pd
import json
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import mlflow

from config import get_config
from legacy.src.models.model_1_encode_kld import build_transformer
from legacy.src.utils.utils import KMerTokenizer, save_model, visualize_mismatch, levenshtein_distance, clean_sequence
from legacy.src.utils.data_setup_balanced import AptamersDataset, collate_embeddings
from legacy.src.utils.pytorch_balanced_sampler.sampler import SamplerFactory

dotenv.load_dotenv(".env")

device = "cpu"
#print(torch.cuda.get_device_name())

DATA_PATH = os.environ["DATA_PATH"]
OUTPUTS_PATH = os.environ["OUTPUTS_PATH"]
CHECKPOINTS_PATH = os.environ["CHECKPOINTS_PATH"]
MLRUNS_PATH = os.environ["MLRUNS_PATH"]


#-------SET-DATA---------------------------------------------------------------------------


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
#df = df[:2000]


####################################################################################################
config = get_config(config_name='config_kld_finetune')
d_model = config['d_model']    #1280
max_len = config['seq_len']    #182
N = config['num_layers']   #2
h = config['num_heads']    #8
dropout = config['dropout']   #0.1
d_ff = config['d_ff']   #512
encoder_hidden_dim = config['encoder_hidden_dim']
latent_dim = config['latent_dim']
beta = config['beta']
exp_name = config['experiment_name']
lr = config['lr']

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

print(len(train_dataloader))
print(len(test_dataloader))



def log_sequences_mlflow(batch_index, pred_ids_list, target_text_list, tokenizer, max_samples=5):
    """
    Логирует несколько сиквенсов (prediction + target) в MLflow
    с сохранением спец-токенов.
    """
    sequences_to_log = []

    for i, (pred_ids, target_seq) in enumerate(zip(pred_ids_list, target_text_list)):
        if i >= max_samples:
            break

        # Декодируем без удаления спец-токенов
        pred_seq = tokenizer.decode_keep_special_tokens(pred_ids)

        sequences_to_log.append({
            "batch_index": batch_index,
            "sample_index": i,
            "pred_seq": pred_seq,
            "target_seq": target_seq
        })

    # Логируем как json-строку
    mlflow.log_text(json.dumps(sequences_to_log, indent=2), f"sequences_batch_{batch_index}.json")



def test_step(model: torch.nn.Transformer, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              beta: float,
              global_step:int = None,
              epoch: int = None
              ):

    model.eval()

    test_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    total_beta_kld_loss = 0
    total_levenshtein = 0
    total_normalized_lev = 0
    all_sequences_list = []
    count = 0
    empty_seq_warnings = 0
    mismatch_examples = []
    progress_bar = tqdm(dataloader, total=len(dataloader), desc="Testing", leave=True)
    with torch.inference_mode():
        for batch in progress_bar:
            src_input  = batch['embedding'].to(device)
            decoder_input = torch.tensor(batch['decoder_input']).to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = torch.tensor(batch['label']).to(device)

            if src_input.dim() == 2:
                src_input  = src_input.unsqueeze(1)
            out, mu, logvar = model(src_input, decoder_input, decoder_mask, use_kld=True)

            recon_loss = loss_fn(out.view(-1, len(tokenizer)), label.view(-1))
            kld = kld_loss(mu, logvar)
            beta_kld = beta * kld
            loss = recon_loss + beta_kld

            pred_ids = out.argmax(dim=-1).detach().cpu().numpy()
            model_out_text = [tokenizer.decode(ids) for ids in pred_ids]
            # Retrieving source and target texts from the batch
            source_text = [batch['ab_name'], batch['tg_name'], batch['ab_seq'], batch['tg_seq']]
            target_name = [batch['apt_name']]
            target_text = batch['apt_seq']
            tqdm.write(f'loss {loss}')

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


                 # ---- собираем 5 примеров mismatch ----
                if len(mismatch_examples) < 5:
                    mismatch_str = visualize_mismatch(target_seq, pred_seq)
                    mismatch_examples.append(mismatch_str)
    
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld.item()
            total_beta_kld_loss += beta_kld.item()

            all_sequences_list.append(model_out_text)
            test_loss += loss.item()



        avg_loss = test_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_kld = total_kld_loss / len(dataloader)
        avg_beta_kld = total_beta_kld_loss / len(dataloader)
        avg_levenshtein = total_levenshtein / count if count > 0 else 0.0
        avg_normalized_lev = total_normalized_lev / count if count > 0 else 0.0

        
        mlflow.log_metric('Validation/Total_loss', avg_loss, step=global_step)
        mlflow.log_metric('Validation/Recon_loss', avg_recon, step=global_step)
        mlflow.log_metric('Validation/KLD_loss', avg_kld, step=global_step)
        mlflow.log_metric('Validation/beta_KLD_loss', avg_beta_kld, step=global_step)
        mlflow.log_metric('Validation/Levenshtein', avg_levenshtein, step=global_step)
        mlflow.log_metric('Validation/Normalized_Levenshtein', avg_normalized_lev, step=global_step)
        if empty_seq_warnings > 0:
            mlflow.log_metric('Empty_sequences', empty_seq_warnings, step=global_step)
        
        # ---- записываем все 5 примеров mismatch ----
        mismatch_file = f"{exp_name}_mismatch.txt"
        with open(mismatch_file, "a") as f:
            f.write(f"Epoch {epoch}, Step {global_step}\n")
            for i, mismatch_str in enumerate(mismatch_examples, 1):
                f.write(f"\nExample {i}:\n{mismatch_str}\n")
            f.write("\n" + "="*80 + "\n\n")
        mlflow.log_artifact(f"{exp_name}_mismatch.txt")

    return avg_loss, avg_recon, avg_kld, avg_levenshtein, avg_normalized_lev


log_every = 15

def train_step(model: torch.nn.Transformer, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               beta: float,
               global_step: int = None
               ):
    model.train()
    

    train_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    total_beta_kld_loss = 0
    total_levenshtein = 0
    total_normalized_lev = 0
    empty_seq_warnings = 0
    count = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=True)
    # Loop through data loader data batches
    for i, batch in progress_bar:
        src_input = batch['embedding'].to(device)
        decoder_input = torch.tensor(batch['decoder_input']).to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        target_text = batch['apt_seq']
        
        
        if src_input.dim() == 2:
            src_input = src_input.unsqueeze(1)
        out, mu, logvar = model(src_input, decoder_input, decoder_mask, use_kld=True)
        pred_ids = out.argmax(dim=-1).detach().cpu().numpy()
        
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
            
        if i % log_every == 0:
            log_sequences_mlflow(batch_index=i,
                                pred_ids_list=pred_ids,
                                target_text_list=batch['apt_seq'],
                                tokenizer=tokenizer,
                                max_samples=5)


        label = torch.tensor(batch['label']).to(device)

        recon_loss = loss_fn(out.view(-1, len(tokenizer)), label.view(-1))
        kld = kld_loss(mu, logvar)
        beta_kld = beta * kld
        loss = recon_loss + beta_kld
        print('loss', loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        mlflow.log_metric("Train/Grad_norm", total_norm, step=global_step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        ## log gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        #print(f"[Step {global_step}] Grad norm = {total_norm:.3f}")
        mlflow.log_metric("Train/Grad_norm_after_clip", total_norm, step=global_step)
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_kld_loss += kld.item()
        total_beta_kld_loss += beta_kld.item()
        train_loss += loss.item()
        global_step += 1

    avg_loss = train_loss / len(dataloader)    #averaged by batch
    avg_recon = total_recon_loss / len(dataloader)
    avg_kld = total_kld_loss / len(dataloader)
    avg_beta_kld = total_beta_kld_loss / len(dataloader)
    avg_levenshtein = total_levenshtein / count if count > 0 else 0.0
    avg_normalized_lev = total_normalized_lev / count if count > 0 else 0.0
    mlflow.log_metric('Train/Total_loss', avg_loss, step=global_step)
    mlflow.log_metric('Train/Recon_loss', avg_recon, step=global_step)
    mlflow.log_metric('Train/KLD_loss', avg_kld, step=global_step)
    mlflow.log_metric('Train/beta_KLD_loss', avg_beta_kld, step=global_step)
    mlflow.log_metric('Train/Levenshtein', avg_levenshtein, step=global_step)
    mlflow.log_metric('Train/Normalized_Levenshtein', avg_normalized_lev, step=global_step)
    if empty_seq_warnings > 0:
        mlflow.log_metric('Empty_sequences', empty_seq_warnings, step=global_step)
    return avg_loss, avg_recon, avg_kld, avg_levenshtein, avg_normalized_lev, global_step




def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          beta: float,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5
          ):
    print('BETA', beta)
    mlflow.set_tracking_uri("file:///mnt/tank/scratch/azaikina/Model/outputs/mlruns")
    mlflow.set_experiment('new_Experiment')
    mismatch_file = f"{exp_name}_mismatch.txt"
    with open(mismatch_file, "w") as f:
        pass
    with mlflow.start_run(run_name="Experiment_run"):
        config_file = f"{exp_name}_config.txt"
        with open(config_file, "w") as f:                   # at the end of test_step write mismatch for visualization
            json.dump(config, f, indent=4)
        mlflow.log_artifact(f"{exp_name}_config.txt")

        results = {
            "train_loss": [],
            "train_recon_loss": [],
            "train_kld_loss": [],
            "test_loss": [],
            "test_recon_loss": [],
            "test_kld_loss": [],
            "train_avg_levenshtein": [],
            "train_normalized_levenshtein": [],
            "test_avg_levenshtein": [],
            "test_normalized_levenshtein": []
        }
        global_step = 0
        #Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            train_loss, train_recon, train_kld, train_avg_levenshtein, train_normalized_levenshtein, global_step = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer, global_step= global_step, beta=beta)
            test_loss, test_recon, test_kld, test_avg_levenshtein, test_normalized_levenshtein = test_step(model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn, global_step=global_step, epoch=epoch, beta=beta)


            results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
            results["train_recon_loss"].append(train_recon.item() if isinstance(train_recon, torch.Tensor) else train_recon)
            results["train_kld_loss"].append(train_kld.item() if isinstance(train_kld, torch.Tensor) else train_kld)
            results["train_avg_levenshtein"].append(train_avg_levenshtein.item() if isinstance(train_avg_levenshtein, torch.Tensor) else train_avg_levenshtein)
            results["train_normalized_levenshtein"].append(train_normalized_levenshtein.item() if isinstance(train_normalized_levenshtein, torch.Tensor) else train_normalized_levenshtein)

            results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
            results["test_recon_loss"].append(test_recon.item() if isinstance(test_recon, torch.Tensor) else test_recon)
            results["test_kld_loss"].append(test_kld.item() if isinstance(test_kld, torch.Tensor) else test_kld)
            results["test_avg_levenshtein"].append(test_avg_levenshtein.item() if isinstance(test_avg_levenshtein, torch.Tensor) else test_avg_levenshtein)
            results["test_normalized_levenshtein"].append(test_normalized_levenshtein.item() if isinstance(test_normalized_levenshtein, torch.Tensor) else test_normalized_levenshtein)
            if epoch % config['save_every']== 0:
                save_model(model=model, target_dir=CHECKPOINTS_PATH, model_name=f'{exp_name}_model.pth')


    # 6. Return the filled results at the end of the epochs
    return results

vocab_size= len(tokenizer)


model = build_transformer(vocab_size, max_len, d_model, N, h, dropout, d_ff, encoder_hidden_dim, latent_dim)

model.load_state_dict(torch.load("/mnt/tank/scratch/azaikina/Model/outputs/checkpoints/Transformer_with_mirna_encode_kld_model.pth"))  # загружаем веса
model.to(device)


torch.manual_seed(42) 
torch.cuda.manual_seed(42)

def kld_loss(mu, logvar):
    # KL divergence between N(mu, sigma^2) and N(0,1)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

# Start the timer
start_time = timer()


model_results = train(model=model, 
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=config['num_epochs'],
                        beta=beta)


results_df = pd.DataFrame(model_results)
results_df.to_csv(os.path.join(OUTPUTS_PATH, f"{exp_name}_training_results.csv"), index=False)
print(f"Training results saved to {exp_name}_training_results.csv")

end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
