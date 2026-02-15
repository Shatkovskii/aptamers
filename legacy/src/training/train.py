from pathlib import Path

import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from utils.utils import (
    KMerTokenizer,
    visualize_mismatch,
    levenshtein_distance,
)
from utils.data_setup import AptamersDataset, collate_embeddings
from legacy.src.models.model_1 import build_transformer


device = "cuda" if torch.cuda.is_available() else "cpu"

# set embeddings and sequences paths
data_path = Path("/mnt/tank/scratch/azaikina/Model/data")
# embeddings_path = Path("../../esm/embeds")
embeddings_path = (
    "/mnt/tank/scratch/azaikina/esm/embeds"  # /mnt/tank/scratch/azaikina/esm/embeds
)

df_path = "/mnt/tank/scratch/azaikina/Model/data/3_checked_intersections.csv"
df = pd.read_csv(df_path, index_col=0)
seq_len = 100
tokenizer = KMerTokenizer(3)
dataset = AptamersDataset(
    df=df, embeddings_path=embeddings_path, seq_len=seq_len, tokenizer=tokenizer
)

# Define split sizes
train_size = int(0.9 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader_custom = DataLoader(
    dataset=train_dataset, batch_size=5, shuffle=True, collate_fn=collate_embeddings
)

test_dataloader_custom = DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=collate_embeddings
)


def test_step(
    model: torch.nn.Transformer,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    print_msg=None,
):
    model.eval()
    # progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing", leave=True)
    # Setup test loss and test accuracy values
    test_loss = 0
    total_levenshtein = 0  # Track total Levenshtein distance
    total_sequences = 0  # Count of sequences processed
    total_normalized_lev = 0
    all_sequences_list = []
    count = 0
    progress_bar = tqdm(dataloader, total=len(dataloader), desc="Testing", leave=True)
    with torch.inference_mode():
        # Loop through data loader data batches
        for batch in progress_bar:
            # embeddings, paths, ab_names, apt_names, tg_names, ab_seqs, apt_seqs, tg_seqs = batch
            encoder_output = batch["embedding"].to(device)
            ##print('encoder_output', encoder_output)
            decoder_input = torch.tensor(batch["decoder_input"]).to(device)
            ##print('decoder_input', decoder_input)
            decoder_mask = batch["decoder_mask"].to(device)
            ##print('decoder_mask', decoder_mask)

            # Running tensors through the Transformer
            # decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            ##print('encoder_output.size()', encoder_output.size(), 'decoder_input.size()', decoder_input.size(), 'decoder_mask.size()', decoder_mask.size())
            if encoder_output.dim() == 2:
                encoder_output = encoder_output.unsqueeze(
                    1
                )  # make shape [batch, seq_len=1, d_model]
            ##print('source size in greedy decode changed', source.size())
            decoder_output = model.decode(encoder_output, decoder_input, decoder_mask)
            ##################model_out = greedy_decode(model, src_embedding, tokenizer_tgt, max_len, device)
            ##print('decoder_output', decoder_output)
            proj_output = model.project(decoder_output)
            ##print('proj_output', proj_output.size(), proj_output)

            label = torch.tensor(batch["label"]).to(device)
            ##print('label', label.size(), label)
            ##print('proj_output_view', proj_output.view(-1, len(tokenizer)))
            ##print('label_view', label.view(-1))

            pred_ids = proj_output.argmax(dim=-1).detach().cpu().numpy()
            model_out_text = [tokenizer.decode(ids) for ids in pred_ids]
            # Retrieving source and target texts from the batch
            source_text = [
                batch["ab_name"],
                batch["tg_name"],
                batch["ab_seq"],
                batch["tg_seq"],
            ]
            target_name = [batch["apt_name"]]
            target_text = batch["apt_seq"]
            ##################model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy()) # Decoded, human-readable model output

            # Printing results
            # print('-'*console_width)
            tqdm.write(f"SOURCE: {source_text}")
            tqdm.write(f"TARGET: {target_text}")
            tqdm.write(f"TARGET NAME: {target_name}")
            tqdm.write(f"PREDICTED: {model_out_text}")

            # # After two examples, we break the loop
            # if count == num_examples:
            #     break

            loss = loss_fn(proj_output.view(-1, len(tokenizer)), label.view(-1))
            tqdm.write(f"loss {loss}")

            pred_seq = model_out_text[0]  # get the first (and only) sequence
            target_seq = target_text[0]  # if target_text is a list
            tqdm.write(visualize_mismatch(target_seq, pred_seq))

            lev_dist = levenshtein_distance(pred_seq, target_seq)
            normalized_lev = lev_dist / max(len(target_seq), 1)

            total_levenshtein += lev_dist
            total_normalized_lev += normalized_lev
            total_sequences += 1
            tqdm.write(f"LEVENSTEIN:{lev_dist}")
            tqdm.write(f"NORM_LEVENSTEIN:{normalized_lev:.2f}")
            count += 1
            all_sequences_list.append(model_out_text)
            test_loss += loss.item()
        test_loss = test_loss / len(dataloader)
        avg_levenshtein = total_levenshtein / len(dataloader)
        avg_normalized_lev = total_normalized_lev / len(dataloader)
        return test_loss, avg_levenshtein, avg_normalized_lev


def train_step(
    model: torch.nn.Transformer,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss = 0
    total_levenshtein = 0
    total_normalized_lev = 0
    total_sequences = 0

    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Training", leave=True
    )

    # Loop through data loader data batches
    for i, batch in progress_bar:
        # embeddings, paths, ab_names, apt_names, tg_names, ab_seqs, apt_seqs, tg_seqs = batch
        encoder_output = batch["embedding"].to(device)
        ##print('encoder_output', encoder_output)
        decoder_input = torch.tensor(batch["decoder_input"]).to(device)
        ##print('decoder_input', decoder_input)
        decoder_mask = batch["decoder_mask"].to(device)
        ##print('decoder_mask', decoder_mask)
        target_text = batch["apt_seq"]

        # Running tensors through the Transformer
        # decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        ##print('encoder_output.size()', encoder_output.size(), 'decoder_input.size()', decoder_input.size(), 'decoder_mask.size()', decoder_mask.size())
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(
                1
            )  # make shape [batch, seq_len=1, d_model]
        ##print('source size in greedy decode changed', source.size())
        decoder_output = model.decode(encoder_output, decoder_input, decoder_mask)
        ##print('decoder_output', decoder_output)
        proj_output = model.project(decoder_output)
        ##print('proj_output', proj_output.size(), proj_output)
        pred_ids = proj_output.argmax(dim=-1).detach().cpu().numpy()
        model_out_text = [tokenizer.decode(ids) for ids in pred_ids]
        pred_seq = model_out_text[0]  # get the first (and only) sequence
        target_seq = target_text[0]  # if target_text is a list

        label = torch.tensor(batch["label"]).to(device)
        ##print('label', label.size(), label)
        ##print('proj_output_view', proj_output.view(-1, len(tokenizer)))
        ##print('label_view', label.view(-1))
        loss = loss_fn(proj_output.view(-1, len(tokenizer)), label.view(-1))
        print("loss", loss)
        lev_dist = levenshtein_distance(pred_seq, target_seq)
        normalized_lev = lev_dist / max(len(target_seq), 1)

        total_levenshtein += lev_dist
        total_normalized_lev += normalized_lev
        total_sequences += 1
        tqdm.write(f"LEVENSTEIN:{lev_dist}")
        tqdm.write(f"NORM_LEVENSTEIN:{normalized_lev:.2f}")

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
    train_loss = train_loss / len(dataloader)
    avg_levenshtein = total_levenshtein / len(dataloader)
    avg_normalized_lev = total_normalized_lev / len(dataloader)
    return train_loss, avg_levenshtein, avg_normalized_lev


# 1. Take in various parameters required for training and test steps
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    epochs: int = 5,
):
    # 2. Create empty results dictionary
    results = {
        "train_loss": [],
        # "train_acc": [],
        "test_loss": [],
        # "test_acc": [],
        "train_avg_levenshtein": [],
        "train_normalized_levenshtein": [],
        "test_avg_levenshtein": [],
        "test_normalized_levenshtein": [],
    }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_avg_levenshtein, train_normalized_levenshtein = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        test_loss, test_avg_levenshtein, test_normalized_levenshtein = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn
        )

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            # f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            # f"test_acc: {test_acc:.4f}"
            f"train_avg_levenshtein: {train_avg_levenshtein} | "
            f"train_normalized_levenshtein: {train_normalized_levenshtein} | "
            f"test_avg_levenshtein: {test_avg_levenshtein} | "
            f"test_normalized_levenshtein: {test_normalized_levenshtein} | "
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(
            train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        )
        results["train_avg_levenshtein"].append(
            train_avg_levenshtein.item()
            if isinstance(train_avg_levenshtein, torch.Tensor)
            else train_avg_levenshtein
        )
        results["train_normalized_levenshtein"].append(
            train_normalized_levenshtein.item()
            if isinstance(train_normalized_levenshtein, torch.Tensor)
            else train_normalized_levenshtein
        )
        results["test_loss"].append(
            test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
        )
        results["test_avg_levenshtein"].append(
            test_avg_levenshtein.item()
            if isinstance(test_avg_levenshtein, torch.Tensor)
            else test_avg_levenshtein
        )
        results["test_normalized_levenshtein"].append(
            test_normalized_levenshtein.item()
            if isinstance(test_normalized_levenshtein, torch.Tensor)
            else test_normalized_levenshtein
        )

    # 6. Return the filled results at the end of the epochs
    return results


d_model = 1280
vocab_size = len(tokenizer)
max_len = 100
N = 6
h = 8
dropout = 0.1
d_ff = 512
model = build_transformer(vocab_size, max_len, d_model, N, h, dropout, d_ff)
model.to(device)


torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 1

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer.token_to_id("[PAD]"), label_smoothing=0.1
).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer

start_time = timer()

# Train model_0
model_results = train(
    model=model,
    train_dataloader=train_dataloader_custom,
    test_dataloader=test_dataloader_custom,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
)

import pandas as pd

# Convert results dict to DataFrame
results_df = pd.DataFrame(model_results)

# Save to CSV
results_df.to_csv("training_results.csv", index=False)

print("✅ Training results saved to training_results.csv")

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")

