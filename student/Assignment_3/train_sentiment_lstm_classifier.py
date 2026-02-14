import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
FASTTEXT_MODEL_PATH = "student/Assignment_3/fasttext-wiki-news-subwords-300.model"
EMBED_DIM = 300
NUM_CLASSES = 3
BATCH_SIZE = 64
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.4
EARLY_STOP_PATIENCE = 8
MAX_LEN = 32
SEED = 42
MIN_EPOCHS = 30
HIDDEN_SIZE = 96
NUM_LAYERS = 1
BIDIRECTIONAL = True
OUTPUT_DIR = "outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_lstm_model.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")


# ────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────
def tokenize(sentence: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", sentence.lower())


def fasttext_has_token(ft: KeyedVectors, token: str) -> bool:
    if hasattr(ft, "key_to_index"):
        return token in ft.key_to_index
    return token in ft  # fallback for older gensim


# ─────────────────────────────────────────────
# Sequence builder(32, 300)
# ─────────────────────────────────────────────
def sentence_to_padded_vectors(
    text: str,
    ft: KeyedVectors,
    max_len: int = 32,) -> np.ndarray:

    tokens  = tokenize(text)
    vec_dim = ft.vector_size
    out     = np.zeros((max_len, vec_dim), dtype=np.float32)

    j = 0
    for token in tokens:
        if j >= max_len:
            break
        if fasttext_has_token(ft, token):
            out[j] = ft.get_vector(token)
            j += 1

    return out


def build_sequence_matrix(
    sentences: np.ndarray,
    ft: KeyedVectors,
    max_len: int,
    split_name: str = "",) -> np.ndarray:
    """
    Build an (N, max_len, vec_dim) float32 array for all sentences.
    """
    N = len(sentences)
    mat = np.zeros((N, max_len, ft.vector_size), dtype=np.float32)

    oov_tokens = 0
    total_tokens = 0

    for i, sent in enumerate(sentences):
        toks = tokenize(sent)
        total_tokens += len(toks)
        oov_tokens   += sum(1 for t in toks if not fasttext_has_token(ft, t))
        mat[i] = sentence_to_padded_vectors(sent, ft, max_len)

    oov_rate = 100.0 * oov_tokens / max(total_tokens, 1)
    print(f"  [{split_name}] OOV rate: {oov_rate:.2f}%  ({oov_tokens}/{total_tokens} tokens)")

    return mat


# ─────────────────────────────────────────────
# 4. Dataset
# ─────────────────────────────────────────────
class SentimentSequenceDataset(Dataset):
    """
    Wraps a pre-computed (N, 32, 300) matrix and integer labels.
    """
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        assert X_seq.ndim == 3, f"Expected 3-D array, got shape {X_seq.shape}"
        self.X = torch.from_numpy(X_seq).float()   # (N, 32, 300)
        self.y = torch.from_numpy(y).long()         # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# 5. Model
# ─────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """
    Input:  (B, 32, 300)  — batch of pre-computed word-vector sequences
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        input_size:    int   = EMBED_DIM,
        hidden_size:   int   = HIDDEN_SIZE,
        num_layers:    int   = NUM_LAYERS,
        bidirectional: bool  = BIDIRECTIONAL,
        dropout:       float = DROPOUT,
        num_classes:   int   = NUM_CLASSES,
    ):
        super().__init__()

        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = bidirectional,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 32, 300)

        h_n shape: (num_layers * num_directions, B, hidden_size)
        We take the last layer's hidden state(s).
        """
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # Last layer: forward = h_n[-2], backward = h_n[-1]
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2 * hidden_size)
        else:
            h_final = h_n[-1]                               # (B, hidden_size)

        h_final = self.dropout(h_final)
        return self.fc(h_final)


# ─────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────
def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = class_counts.sum() / (class_counts + 1e-8)
    weights = weights / weights.sum() * num_classes   # rescale to sum to num_classes
    w = torch.tensor(weights, dtype=torch.float32)
    print(f"Class counts : {class_counts.tolist()}")
    print(f"Class weights: {w.tolist()}")
    return w


# ─────────────────────────────────────────────
# Training / evaluation helpers
# ─────────────────────────────────────────────
def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,) -> tuple[float, float, float]:
    """One training pass. Returns (avg_loss, accuracy, macro_f1)."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.append(yb.detach().cpu().numpy())

    y_true   = np.concatenate(all_labels)
    y_pred   = np.concatenate(all_preds)
    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return avg_loss, acc, f1


def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,) -> tuple[float, float, float]:
    """Evaluation pass. Returns (avg_loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb  = xb.to(device), yb.to(device)
            logits  = model(xb)
            loss    = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    y_true   = np.concatenate(all_labels)
    y_pred   = np.concatenate(all_preds)
    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return avg_loss, acc, f1


def get_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) arrays without computing loss."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb     = xb.to(device)
            preds  = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_training_curves(history: dict, out_dir: str) -> None:
    """Save a single figure with loss, accuracy and macro-F1 subplots."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle("LSTM Training Curves", fontsize=14, fontweight="bold")

    panels = [
        ("Loss",     "train_loss", "val_loss"),
        ("Accuracy", "train_acc",  "val_acc"),
        ("Macro F1", "train_f1",   "val_f1"),
    ]

    for ax, (title, train_key, val_key) in zip(axes, panels):
        ax.plot(epochs, history[train_key], label="Train", linewidth=1.8)
        ax.plot(epochs, history[val_key],   label="Val",   linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "lstm_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves  → {path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str,) -> None:
    """Save the test-set confusion matrix."""
    class_names = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
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
    ax.set_title("LSTM – Test Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(out_dir, "lstm_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {path}")


# ─────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────
def main() -> None:
    set_seed(SEED)

    device = torch.device(DEVICE)

    # Dataset
    print("\n========== Loading Dataset ==========")
    dataset   = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    sentences = np.array(dataset["train"]["sentence"])
    labels    = np.array(dataset["train"]["label"])   # 0=neg, 1=neu, 2=pos

    # Stratified split: 15% test, then 15% of remainder for val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        sentences, labels,
        test_size=0.15, stratify=labels, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.15, stratify=y_trainval, random_state=SEED
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = {c: int(np.sum(ys == c)) for c in range(3)}
        print(f"  {name} class distribution: {counts}")

    # FastText
    print("\n========== Loading FastText ==========")
    ft = KeyedVectors.load(FASTTEXT_MODEL_PATH)
    print(f"Vocabulary size: {len(ft.index_to_key):,}")
    print(f"Embedding dim:   {ft.vector_size}")

    # Build (N, 32, 300) sequence matrices
    print("\n========== Building padded word-vector sequences ==========")
    X_train_seq = build_sequence_matrix(X_train, ft, MAX_LEN, "Train")
    X_val_seq   = build_sequence_matrix(X_val,   ft, MAX_LEN, "Val")
    X_test_seq  = build_sequence_matrix(X_test,  ft, MAX_LEN, "Test")

    print(f"Train matrix shape: {X_train_seq.shape}")   # (N, 32, 300)

    # DataLoaders
    train_loader = DataLoader(
        SentimentSequenceDataset(X_train_seq, y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        SentimentSequenceDataset(X_val_seq, y_val),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        SentimentSequenceDataset(X_test_seq, y_test),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # Class weights
    class_weights = compute_class_weights(y_train, NUM_CLASSES).to(device)

    # Model
    model = LSTMClassifier(
        input_size=ft.vector_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel:\n{model}")
    print(f"Trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight = class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    # Training loop
    print("\n========== Training ==========")
    history = {k: [] for k in [
        "train_loss", "train_acc", "train_f1",
        "val_loss",   "val_acc",   "val_f1",
    ]}

    best_val_f1   = -1.0
    epochs_no_imp = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_acc, va_f1 = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(va_f1)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        # Checkpoint
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            epochs_no_imp = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_f1": best_val_f1,
                    "class_weights": class_weights.cpu().tolist(),
                },
                MODEL_SAVE_PATH,
            )
            tag = " ← best"
        else:
            epochs_no_imp += 1
            tag = ""

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} F1={tr_f1:.4f} | "
            f"Val   loss={va_loss:.4f} acc={va_acc:.4f} F1={va_f1:.4f}"
            f"{tag}"
        )

        # Early stopping — only active after MIN_EPOCHS
        if epoch >= MIN_EPOCHS and epochs_no_imp >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"\nBest validation Macro F1: {best_val_f1:.4f}")
    plot_training_curves(history, OUTPUT_DIR)

    # Test evaluation
    print("\n========== Test Evaluation ==========")
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    te_loss, te_acc, te_f1 = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss     : {te_loss:.4f}")
    print(f"Test Accuracy : {te_acc:.4f}")
    print(f"Test Macro F1 : {te_f1:.4f}")

    if te_f1 >= 0.70:
        print("✓ Performance requirement met (Macro F1 >= 0.70)")
    else:
        print("✗ Performance requirement NOT met (Macro F1 < 0.70)")
        print("  Try: more hidden units, more layers, longer training, or tuning dropout/lr.")

    y_true, y_pred = get_predictions(model, test_loader, device)
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Negative", "Neutral", "Positive"],
        digits=4,
    ))

    plot_confusion_matrix(y_true, y_pred, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print(f"Best model saved to:  {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()