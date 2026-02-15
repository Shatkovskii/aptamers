#tokenizer
import torch
from typing import List, Union
import itertools
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import numpy as np

class KMerTokenizer:
    """
    Tokenizer for DNA sequences that splits sequences into k-mers.

    Features:
    - Builds a vocabulary of all possible k-mers for given nucleotides.
    - Supports special tokens: [PAD], [SOS], [EOS].
    - Converts k-mers to integer IDs and vice versa.
    - Can pad sequences so that their length is a multiple of k.
    - Provides methods for tokenizing and decoding sequences.

    Parameters:
    ----------
    k : int
        Length of each k-mer (default: 5).
    nucleotides : str
        Allowed characters in sequences (default: "ATCG").

    Attributes:
    ----------
    vocab : List[str]
        List of all k-mers plus special tokens.
    token_to_id_map : dict
        Maps k-mer strings to integer IDs.
    id_to_token_map : dict
        Maps integer IDs back to k-mer strings.
    pad_id, sos_id, eos_id : int
        IDs of special tokens.
    """
    def __init__(self, k: int = 5, nucleotides: str = "AUCG"):
        self.k = k
        self.nucleotides = nucleotides

        # Строим основной словарь k-меров
        self.vocab = self._build_vocab()

        # Добавляем специальные токены в начало
        self.special_tokens = {'[PAD]': 0, '[SOS]': 1, '[EOS]': 2}
        self.vocab = list(self.special_tokens.keys()) + self.vocab


        self.token_to_id_map = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token_map = {idx: token for idx, token in enumerate(self.vocab)}

        self.pad_id = self.token_to_id_map['[PAD]']
        self.sos_id = self.token_to_id_map['[SOS]']
        self.eos_id = self.token_to_id_map['[EOS]']

    def _build_vocab(self) -> List[str]:
        """Makes a list of all combinations"""
        return [''.join(kmer) for kmer in itertools.product(self.nucleotides, repeat=self.k)]

    def token_to_id(self, token: str) -> int:
        """Возвращает ID для одного токена"""
        if token not in self.token_to_id_map:
            raise ValueError(f"Token '{token}' not in vocabulary")
        return self.token_to_id_map[token]

    def id_to_token(self, token_id: int) -> str:
        """Возвращает токен по ID"""
        return self.id_to_token_map[token_id]

    def tokenize(self, sequence: str) -> List[str]:
        sequence = self.pad_sequence(sequence)
        step = self.k
        kmers = []
        for i in range(0, len(sequence) - self.k + 1, step):
            kmer = sequence[i:i+self.k]
            if all(c in self.nucleotides or c == '[PAD]' for c in kmer):
                kmers.append(kmer)
        return kmers

    def encode(self, sequence: str) -> list[int]:
        """
        Convert a DNA sequence into a list of token IDs.
        Adds [SOS] at the start and [EOS] at the end.
        """
        kmers = self.tokenize(sequence)
        token_ids = [self.sos_id] + [self.token_to_id(kmer) for kmer in kmers] + [self.eos_id]
        return token_ids    

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode a tensor of token IDs back to a sequence string"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy()

        # Convert to integers and remove padding
        tokens = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.eos_id:  # Stop at EOS token
                break
            if token_id not in [self.pad_id, self.sos_id]:  # Skip PAD and SOS
                tokens.append(self.id_to_token_map[token_id])

        # Join k-mers back into sequence
        return ''.join(tokens).replace('[PAD]', '')

    def decode_tensors(self, token_ids: Union[torch.Tensor, List[int]]) -> str:
        """Decode a tensor of token IDs back to a sequence string"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        # Convert to integers and remove padding
        tokens = []
        for token_id in token_ids:
            if token_id == self.eos_id:  # Stop at EOS token
                break
            if token_id not in (self.pad_id, self.sos_id):  # Skip PAD and SOS
                tokens.append(self.id_to_token_map[token_id])

        # Join k-mers back into sequence
        return ''.join(tokens).replace('[PAD]', '')



    def decode_keep_special_tokens(self, token_ids) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return ''.join(self.id_to_token_map[int(tid)] for tid in token_ids)

    def pad_sequence(self, sequence: str) -> str:
        """Padding is calculated to make the length a multiple of k"""
        pad_length = (self.k - len(sequence) % self.k) % self.k
        return sequence + '[PAD]' * pad_length

    def __len__(self) -> int:
        """Number of kmer combinations"""
        return len(self.vocab)



def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def visualize_mismatch(target: str, predicted: str) -> str:
    output_lines = ["TARGET:    " + target]
    mismatch_line = "PREDICTED: "
    pointer_line = "           "

    max_len = max(len(target), len(predicted))
    for i in range(max_len):
        t_char = target[i] if i < len(target) else "-"
        p_char = predicted[i] if i < len(predicted) else "+"

        mismatch_line += p_char
        if t_char != p_char:
            pointer_line += "^"
        else:
            pointer_line += " "

    output_lines.append(mismatch_line)
    output_lines.append(pointer_line)
    return "\n".join(output_lines)

def clean_sequence(seq: str) -> str:
    for token in ['[PAD]', '[SOS]', '[EOS]', '[UNK]']:
        seq = seq.replace(token, '')
    return seq.strip()


def compute_jsd(p, q, bins=50):
    p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
    p_hist += 1e-12
    q_hist += 1e-12
    js_div = jensenshannon(p_hist, q_hist, base=2) ** 2  # JS divergence
    js_dist = np.sqrt(js_div)  # JS distance
    return js_div, js_dist

def compute_emd(p, q):
    return wasserstein_distance(p, q)

def gc_content(seq):
    seq = seq.upper()
    g = seq.count("G")
    c = seq.count("C")
    valid = sum(seq.count(x) for x in "ATGC")
    return (g + c) / valid if valid > 0 else 0.0

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
             

# Function to construct the path for saving and retrieving model weights
def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder'] # Extracting model folder from the config
    model_basename = config['model_basename'] # Extracting the base name for model files
    model_filename = f"{model_basename}{epoch}.pt" # Building filename
    return str(Path('') / model_folder / model_filename) # Combining current directory, the model folder, and the model filename


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")
