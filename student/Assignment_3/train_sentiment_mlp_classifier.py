import os
import random
import numpy as np
import torch
import re
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from datasets import load_dataset
import gensim.models
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

SEED = 42

def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(SEED)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

FASTTEXT_MODEL_PATH = "student/Assignment_3/fasttext-wiki-news-subwords-300.model"
EMBED_DIM = 300
NUM_CLASSES = 3     
BATCH_SIZE = 64
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4
EARLY_STOP_PATIENCE = 8
HIDDEN_DIMS = [512, 256, 128]
OUTPUT_DIR = "outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_mlp_model.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# Load FastText Embeddings
# ─────────────────────────────────────────────

def load_fasttext(model_path: str):
    print(f"\n========== Loading FastText model from {model_path} ==========")
    kv = gensim.models.KeyedVectors.load(model_path)
    print(f"Vocabulary size: {len(kv.index_to_key):,}")
    print(f"Embedding dimension: {kv.vector_size}")
    return kv


# ─────────────────────────────────────────────
# Dataset Loading and Splitting
# ─────────────────────────────────────────────

def load_and_split_dataset():
    print("\n========== Loading Dataset ==========")
    dataset = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    print("Dataset loaded. Example:", dataset["train"][:3])

    # Creating own split
    sentences = dataset["train"]["sentence"]
    labels    = dataset["train"]["label"]

    # Split test set (15%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        sentences, labels,
        test_size=0.15,
        stratify=labels,
        random_state=SEED
    )

    # Then split train+val into train (85%) and val (15%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.15,
        stratify=y_trainval,
        random_state=SEED
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # Class distribution
    for split_name, split_labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = {c: sum(1 for l in split_labels if l == c) for c in range(3)}
        print(f"  {split_name} class distribution: {counts}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# ─────────────────────────────────────────────
# Sentence Embedding
# ─────────────────────────────────────────────

def simple_tokenize(sentence: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9']+", sentence.lower())
    return tokens


def sentence_to_embedding(sentence: str, kv) -> np.ndarray:
    tokens = simple_tokenize(sentence)
    vectors = []
    for token in tokens:
        if token in kv:
            vectors.append(kv[token])
    if vectors:
        return np.mean(vectors, axis=0).astype(np.float32)
    else:
        return np.zeros(kv.vector_size, dtype=np.float32)


def embed_sentences(sentences: list[str], kv) -> np.ndarray:
    embeddings = np.stack([sentence_to_embedding(s, kv) for s in sentences])
    return embeddings

# ─────────────────────────────────────────────
# Class Weights
# ─────────────────────────────────────────────

def compute_class_weights(labels: list[int]) -> torch.Tensor:
    labels_arr = np.array(labels)
    num_classes = len(np.unique(labels_arr))
    total = len(labels_arr)
    weights = []
    for c in range(num_classes):
        count = np.sum(labels_arr == c)
        weights.append(total / (num_classes * count))
    weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights: {weights}")
    return weights


# ─────────────────────────────────────────────
# MLP Model
# ─────────────────────────────────────────────

class SentimentMLP(nn.Module):
    """
    Architecture:
        Linear(300→512) → BatchNorm → ReLU → Dropout
        Linear(512→256) → BatchNorm → ReLU → Dropout
        Linear(256→128) → BatchNorm → ReLU → Dropout
        Linear(128→3)
    """

    def __init__(
        self,
        input_dim: int = EMBED_DIM,
        hidden_dims: list[int] = HIDDEN_DIMS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─────────────────────────────────────────────
# Training and Evaluation Helpers
# ─────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
):
    """Run one training epoch; return (loss, accuracy, macro-F1)."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_training_curves(history: dict, save_prefix: str = "mlp"):
    """Plot and save loss, accuracy and macro-F1 curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle("MLP Training Curves", fontsize=14, fontweight="bold")

    metrics = [
        ("Loss",     "train_loss", "val_loss"),
        ("Accuracy", "train_acc",  "val_acc"),
        ("Macro F1", "train_f1",   "val_f1"),
    ]

    for ax, (title, train_key, val_key) in zip(axes, metrics):
        ax.plot(epochs, history[train_key], label="Train", linewidth=1.8)
        ax.plot(epochs, history[val_key],   label="Val",   linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{save_prefix}_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves → {path}")


def plot_confusion_matrix(y_true, y_pred, save_prefix: str = "mlp"):
    """Plot and save the confusion matrix."""
    class_names = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("MLP – Test Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{save_prefix}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix  → {path}")


# ─────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────

def main():
    set_seeds(SEED)

    # Load FastText
    kv = load_fasttext(FASTTEXT_MODEL_PATH)

    # Dataset
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_dataset()

    # Embed sentences
    print("\n========== Computing sentence embeddings ==========")
    train_emb = embed_sentences(X_train, kv)
    val_emb   = embed_sentences(X_val,   kv)
    test_emb  = embed_sentences(X_test,  kv)
    print(f"Train embeddings: {train_emb.shape}")
    print(f"Val   embeddings: {val_emb.shape}")
    print(f"Test  embeddings: {test_emb.shape}")

    # DataLoaders
    def make_loader(embeddings, labels, shuffle=False):
        X_t = torch.tensor(embeddings, dtype=torch.float32)
        y_t = torch.tensor(labels,     dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(train_emb, list(y_train), shuffle=True)
    val_loader   = make_loader(val_emb,   list(y_val))
    test_loader  = make_loader(test_emb,  list(y_test))

    # Class weights
    class_weights = compute_class_weights(list(y_train)).to(DEVICE)

    # Model, criterion, optimiser, scheduler
    model = SentimentMLP(
        input_dim=EMBED_DIM,
        hidden_dims=HIDDEN_DIMS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    print(f"\nModel:\n{model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Training
    print("\n========== Training ==========")
    history = {k: [] for k in ["train_loss", "train_acc", "train_f1",
                                "val_loss",   "val_acc",   "val_f1"]}

    best_val_f1   = -1.0
    epochs_no_imp = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_f1)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Checkpoint best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_no_imp = 0
            tag = " ← best"
        else:
            epochs_no_imp += 1
            tag = ""

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} F1={tr_f1:.4f} | "
            f"Val   loss={val_loss:.4f} acc={val_acc:.4f} F1={val_f1:.4f}"
            f"{tag}"
        )

        # Early stopping (only after epoch 30)
        if epoch >= 30 and epochs_no_imp >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"\nBest validation Macro F1: {best_val_f1:.4f}")

    # Plot training curves
    plot_training_curves(history, save_prefix="mlp")

    # Final test evaluation (best checkpoint)
    print("\n========== Test Evaluation ==========")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, DEVICE)

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro F1 : {test_f1:.4f}")

    if test_f1 >= 0.65:
        print("✓ Performance requirement met (Macro F1 >= 0.65)")
    else:
        print("✗ Performance requirement NOT met (Macro F1 < 0.65)")
        print("  Try tuning: more epochs, different architecture, or learning rate.")

    # Confusion matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    plot_confusion_matrix(all_labels, all_preds, save_prefix="mlp")

    print(f"\nSaved best model → {MODEL_SAVE_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()