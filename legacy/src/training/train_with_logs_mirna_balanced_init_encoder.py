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
from seqshannon import shannon_entropy

import mlflow

from config import get_config
from legacy.src.models.model_1_init_encode import build_transformer
from legacy.src.utils.utils import KMerTokenizer, save_model, visualize_mismatch, levenshtein_distance, clean_sequence, compute_jsd, compute_emd, gc_content, EarlyStopping
from legacy.src.utils.data_setup_balanced import AptamersDataset, collate_embeddings
from legacy.src.utils.pytorch_balanced_sampler.sampler import SamplerFactory

dotenv.load_dotenv(".env")

device = "cuda"
print(torch.cuda.get_device_name())

DATA_PATH = os.environ["DATA_PATH"]
OUTPUTS_PATH = os.environ["OUTPUTS_PATH"]
CHECKPOINTS_PATH = os.environ["CHECKPOINTS_PATH"]
MLRUNS_PATH = os.environ["MLRUNS_PATH"]

embeddings_path = "/mnt/tank/scratch/azaikina/esm/mirna_embeds"
#embeddings_path = os.path.join(DATA_PATH, "mirna_embeds")
#changed data
df_path = os.path.join(DATA_PATH, 'mirna_df_clean.csv')
df = pd.read_csv(df_path, index_col = 1)


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
#df = df[:10000]

####################################################################################################
config = get_config(config_name='config_1_init_encode')
d_model = config['d_model']    #1280
max_len = config['seq_len']    #100
N = config['num_layers']   #2
h = config['num_heads']    #8
dropout = config['dropout']   #0.1
d_ff = config['d_ff']   #512
encoder_hidden_dim = config['encoder_hidden_dim']
exp_name = config['experiment_name']

mismatch_file = f"{exp_name}_mismatch.txt"

early_stopping = EarlyStopping(patience=config['patience'], delta=config['delta_for_early_stop'], verbose=True)


tokenizer = KMerTokenizer(k = config['kmer'])

vocab_size= len(tokenizer)


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
    empty_seq_warnings = 0
    mismatch_examples = []

    levenshtein_list = []
    gc_real, gc_pred = [], []
    len_real, len_pred = [], []
    entropy_real, entropy_pred = [], []

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
                levenshtein_list.append(lev_dist)

                gc_real.append(gc_content(target_seq))
                gc_pred.append(gc_content(pred_seq))          

                len_real.append(len(target_seq))
                len_pred.append(len(pred_seq))

                entropy_real.append(shannon_entropy(target_seq))
                entropy_pred.append(shannon_entropy(pred_seq))           
                count += 1


                if len(mismatch_examples) < 5:
                    mismatch_str = visualize_mismatch(target_seq, pred_seq)
                    mismatch_examples.append(mismatch_str)

            loss = loss_fn(proj_output.view(-1, len(tokenizer)), label.view(-1))
            tqdm.write(f'loss {loss}')

            all_sequences_list.append(model_out_text)
            test_loss += loss.item()

        with open(f"levenshtein_list.txt", "a") as f:                   # at the end of test_step write mismatch for visualization
            json.dump(levenshtein_list, f, indent=4)
        mlflow.log_artifact(f"levenshtein_list.txt")

        with open(f"levenshtein_list_0.txt", "a") as f:                   # at the end of test_step write mismatch for visualization
            json.dump([0]*len(levenshtein_list), f, indent=4)

        test_loss = test_loss / len(dataloader)
        avg_levenshtein = total_levenshtein / len(dataloader)
        avg_normalized_lev = total_normalized_lev / len(dataloader)

        avg_levenshtein = total_levenshtein / count if count > 0 else 0.0
        avg_normalized_lev = total_normalized_lev / count if count > 0 else 0.0

        jsd_lev, jsdist_lev = compute_jsd(levenshtein_list, [0]*len(levenshtein_list))
        emd_lev = compute_emd(levenshtein_list, [0]*len(levenshtein_list))

        jsd_gc, jsdist_gc = compute_jsd(gc_real, gc_pred)
        emd_gc = compute_emd(gc_real, gc_pred)

        jsd_len, jsdist_len = compute_jsd(len_real, len_pred)
        emd_len = compute_emd(len_real, len_pred)

        jsd_ent, jsdist_ent = compute_jsd(entropy_real, entropy_pred)
        emd_ent = compute_emd(entropy_real, entropy_pred)


        mlflow.log_metric('Validation/Test_loss', test_loss, step=global_step)
        mlflow.log_metric('Validation/Levenshtein', avg_levenshtein, step=global_step)
        mlflow.log_metric('Validation/Normalized_Levenshtein', avg_normalized_lev, step=global_step)
        
            # Levenshtein
        mlflow.log_metric('Validation_dist_comparison/JSD_Levenshtein', float(jsd_lev), step=global_step)
        mlflow.log_metric('Validation_dist_comparison/JSDdist_Levenshtein', jsdist_lev, step=global_step)
        mlflow.log_metric('Validation_dist_comparison/EMD_Levenshtein', emd_lev, step=global_step)

        # GC-content
        mlflow.log_metric('Validation_dist_comparison/JSD_GC', float(jsd_gc), step=global_step)
        mlflow.log_metric('Validation_dist_comparison/JSDdist_GC', jsdist_gc, step=global_step)
        mlflow.log_metric('Validation_dist_comparison/EMD_GC', emd_gc, step=global_step)

        # Length
        mlflow.log_metric('Validation_dist_comparison/JSD_Length', float(jsd_len), step=global_step)
        mlflow.log_metric('Validation_dist_comparison/JSDdist_Length', jsdist_len, step=global_step)
        mlflow.log_metric('Validation_dist_comparison/EMD_Length', emd_len, step=global_step)

        # Shannon entropy
        mlflow.log_metric('Validation_dist_comparison/JSD_Entropy', float(jsd_ent), step=global_step)
        mlflow.log_metric('Validation_dist_comparison/JSDdist_Entropy', jsdist_ent, step=global_step)
        mlflow.log_metric('Validation_dist_comparison/EMD_Entropy', emd_ent, step=global_step)

        
        if empty_seq_warnings > 0:
            mlflow.log_metric('Empty_sequences', empty_seq_warnings, step=global_step)
        
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
    total_sequences = 0
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



def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    mlflow.set_tracking_uri(MLRUNS_PATH)
    mlflow.set_experiment('Experiment')
    with open(mismatch_file, "w") as f:
        pass

    with mlflow.start_run(run_name="Experiment_run"):
        config_file = f"{exp_name}_config.txt"
        with open(f"{exp_name}_config.txt", "a") as f:                   # at the end of test_step write mismatch for visualization
            json.dump(config, f, indent=4)
        mlflow.log_artifact(f"{exp_name}_config.txt")

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
                        # Проверяем, улучшились ли все метрики по сравнению с предыдущим лучшим
            is_best = all(
                results[k][-1] < min(results[k][:-1]) if len(results[k]) > 1 else True
                for k in results
            )

            if is_best:
                save_model(model=model, target_dir=CHECKPOINTS_PATH, model_name=f'{exp_name}_best_model.pth')
                print(f"Epoch {epoch}: New best model saved!")

            if epoch % config['save_every']== 0:
                save_model(model=model, target_dir=CHECKPOINTS_PATH, model_name='test.pth')

            early_stopping.check_early_stop(test_loss)
    
            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch}")
                save_model(model=model, target_dir=CHECKPOINTS_PATH, model_name='early_stopped.pth')
                break

    # 6. Return the filled results at the end of the epochs
    return results


model = build_transformer(vocab_size, max_len, d_model, N, h, dropout, d_ff, encoder_hidden_dim)
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
