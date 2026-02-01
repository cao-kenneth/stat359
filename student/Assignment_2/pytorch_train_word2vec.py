import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        # Expect columns ['center', 'context'] with integer indices
        self.centers = torch.tensor(skipgram_df["center"].values, dtype=torch.long)
        self.contexts = torch.tensor(skipgram_df["context"].values, dtype=torch.long)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)   # center embeddings
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)  # context embeddings

        # small initialization for beginning
        init_range = 0.5 / embedding_dim
        self.in_embed.weight.data.uniform_(-init_range, init_range)
        self.out_embed.weight.data.uniform_(-init_range, init_range)

    def forward(self, center_idx, context_idx):
        """
        center_idx: (B,)
        context_idx:
          - (B,) for positive contexts
          - (B, K) for negative contexts
        returns logits:
          - (B,) or (B, K)
        """
        v = self.in_embed(center_idx)  # (B, D)

        if context_idx.dim() == 1:
            u = self.out_embed(context_idx)      # (B, D)
            logits = torch.sum(v * u, dim=1)     # (B,)
        else:
            u = self.out_embed(context_idx)      # (B, K, D)
            logits = torch.sum(u * v.unsqueeze(1), dim=2)  # (B, K)

        return logits

    def get_embeddings(self, as_numpy=False):
        emb = self.in_embed.weight.detach().cpu()
        return emb.numpy() if as_numpy else emb

# Load processed data
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

skipgram_df = data["skipgram_df"]
word2idx = data["word2idx"]
idx2word = data["idx2word"]
counter = data["counter"] 

vocab_size = len(word2idx)
print(f"Loaded {len(skipgram_df):,} skip-gram pairs")
print(f"Vocab size: {vocab_size:,}")

# Precompute negative sampling distribution below
def build_neg_dist(counter_obj, word2idx_map, power = 0.75):
    counts = np.zeros(len(word2idx_map), dtype = np.float64)
    for w, i in word2idx_map.items():
        counts[i] = counter_obj.get(w, 0)

    if counts.sum() == 0:
        counts += 1.0

    probs = counts ** power
    probs = probs / probs.sum()
    return torch.tensor(probs, dtype=torch.float)


neg_dist = build_neg_dist(counter, word2idx, power=0.75)

# Device selection: CUDA > MPS > CPU
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Using device:", device)


# Dataset and DataLoader
dataset = SkipGramDataset(skipgram_df)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# Model, Loss, Optimizer
model = Word2Vec(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def sample_negatives(neg_dist_cpu, batch_size, num_neg, pos_context_cpu=None):
    """
    neg_dist_cpu: (V,) probabilities on CPU
    pos_context_cpu: (B,) true context indices on CPU (optional); if provided, resample collisions
    returns: (B, K) negatives on CPU
    """
    negs = torch.multinomial(neg_dist_cpu, batch_size * num_neg, replacement=True).view(batch_size, num_neg)

    if pos_context_cpu is not None:
        pos = pos_context_cpu.view(-1, 1)
        mask = (negs == pos)
        while mask.any():
            n = int(mask.sum().item())
            repl = torch.multinomial(neg_dist_cpu, n, replacement=True)
            negs[mask] = repl
            mask = (negs == pos)

    return negs


def make_targets(center, context, vocab_size):
    return torch.ones(center.size(0), device=center.device)

# Training loop
model.train()
for epoch in range(1, EPOCHS + 1):
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for centers, contexts in pbar:
        centers = centers.to(device)
        contexts = contexts.to(device)
        bsz = centers.size(0)

        # Positive
        pos_logits = model(centers, contexts)  # (B,)
        pos_labels = make_targets(centers, contexts, vocab_size)  # (B,) ones

        # Negative samples (sample on CPU, then move to device)
        contexts_cpu = contexts.detach().to("cpu")
        neg_contexts_cpu = sample_negatives(neg_dist, bsz, NEGATIVE_SAMPLES, pos_context_cpu=contexts_cpu)
        neg_contexts = neg_contexts_cpu.to(device)  # (B, K)

        neg_logits = model(centers, neg_contexts)  # (B, K)
        neg_labels = torch.zeros_like(neg_logits, device=device)

        all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        all_labels = torch.cat([pos_labels.unsqueeze(1), neg_labels], dim=1)
        loss = criterion(all_logits, all_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")

# Save embeddings and mappings
embeddings = model.get_embeddings(as_numpy = True)
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
