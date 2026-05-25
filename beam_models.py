"""Model and metric helpers for the GPS-based beam prediction workshop."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    """Make NumPy, Python, and PyTorch results easier to reproduce."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_knn_model(n_neighbors: int = 7, weights: str = "distance") -> Pipeline:
    """Create a KNN pipeline with scaling inside the pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
        ]
    )


def probabilities_to_topk_indices(probabilities: np.ndarray, max_k: int = 10) -> np.ndarray:
    """Return beam indices sorted from most to least likely."""
    return np.argsort(probabilities, axis=1)[:, ::-1][:, :max_k]


def knn_predict_proba_64(model: Pipeline, X: np.ndarray, n_beams: int = 64) -> np.ndarray:
    """Expand KNN probabilities to all 64 beam columns, including unseen classes."""
    knn = model.named_steps["knn"]
    raw_proba = model.predict_proba(X)
    proba = np.zeros((X.shape[0], n_beams), dtype=float)
    proba[:, knn.classes_.astype(int)] = raw_proba
    return proba


class BeamMLP(nn.Module):
    """A small fully connected classifier for GPS-to-beam prediction."""

    def __init__(self, input_dim: int = 2, n_classes: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainingHistory:
    train_loss: list[float]
    test_loss: list[float]
    train_acc: list[float]
    test_acc: list[float]


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return (predictions == y).float().mean().item()


def train_pytorch_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = False,
    normalize: bool = False,
) -> tuple[BeamMLP, TrainingHistory, StandardScaler | None]:
    """Train a PyTorch MLP and return the model, history, and fitted scaler.

    Set normalize=False when the caller has already pre-processed the features
    (e.g. applied MinMaxScaler externally). In that case the returned scaler is None
    and predict_proba_pytorch should be called with scaler=None as well.
    """
    set_seed(seed)
    if normalize:
        scaler: StandardScaler | None = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
    else:
        scaler = None
        X_train_scaled = X_train.astype(np.float32)
        X_test_scaled = X_test.astype(np.float32)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BeamMLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_train_t = torch.from_numpy(X_train_scaled).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.int64)).to(device)
    X_test_t = torch.from_numpy(X_test_scaled).to(device)
    y_test_t = torch.from_numpy(y_test.astype(np.int64)).to(device)

    history = TrainingHistory(train_loss=[], test_loss=[], train_acc=[], test_acc=[])
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            # 1. Forward pass: from GPS features to 64 beam logits.
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)

            # 2. Loss: compare logits with the true best beam index.
            loss = loss_fn(logits, yb)

            # 3. Backward pass: compute gradients and update weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation is done after each epoch without gradient tracking.
        model.eval()
        with torch.no_grad():
            train_logits = model(X_train_t)
            test_logits = model(X_test_t)
            train_loss = loss_fn(train_logits, y_train_t).item()
            test_loss = loss_fn(test_logits, y_test_t).item()
            history.train_loss.append(train_loss)
            history.test_loss.append(test_loss)
            history.train_acc.append(_accuracy_from_logits(train_logits, y_train_t))
            history.test_acc.append(_accuracy_from_logits(test_logits, y_test_t))
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
            print(
                f"epoch {epoch + 1:03d}/{epochs} | "
                f"train_loss={history.train_loss[-1]:.3f} | "
                f"val_loss={history.test_loss[-1]:.3f} | "
                f"val_acc={history.test_acc[-1]:.3f}"
            )

    return model, history, scaler


def predict_proba_pytorch(
    model: BeamMLP,
    scaler: StandardScaler | None,
    X: np.ndarray,
    device: str | None = None,
) -> np.ndarray:
    """Return class probabilities from a trained PyTorch classifier.

    Pass scaler=None when the features were normalized externally before training
    (i.e. train_pytorch_classifier was called with normalize=False).
    """
    if device is None:
        device = next(model.parameters()).device.type
    if scaler is not None:
        X_scaled = scaler.transform(X).astype(np.float32)
    else:
        X_scaled = X.astype(np.float32)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_scaled).to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    return probabilities


def topk_accuracy(y_true: np.ndarray, topk_indices: np.ndarray, max_k: int = 10) -> np.ndarray:
    """Compute Top-1 through Top-K accuracy from ranked predicted beam indices."""
    accuracies = []
    for k in range(1, max_k + 1):
        hit = np.any(topk_indices[:, :k] == y_true[:, None], axis=1)
        accuracies.append(hit.mean())
    return np.array(accuracies)


def power_loss_per_sample(
    power_matrix: np.ndarray,
    topk_indices: np.ndarray,
    k: int = 1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute per-sample power loss in dB for a Top-K beam list."""
    best_power = np.max(power_matrix, axis=1)
    chosen_power = np.array(
        [np.max(power_matrix[i, topk_indices[i, :k]]) for i in range(power_matrix.shape[0])]
    )
    return 10 * np.log10((best_power + eps) / (chosen_power + eps))


def mean_power_loss_db(
    power_matrix: np.ndarray,
    topk_indices: np.ndarray,
    max_k: int = 10,
) -> np.ndarray:
    """Compute mean power loss for Top-1 through Top-K."""
    return np.array(
        [power_loss_per_sample(power_matrix, topk_indices, k=k).mean() for k in range(1, max_k + 1)]
    )


def power_loss_exceedance(
    power_matrix: np.ndarray,
    topk_indices: np.ndarray,
    max_k: int = 10,
    threshold_db: float = 3.0,
) -> np.ndarray:
    """Percent of samples whose power loss is above a threshold for each K."""
    return np.array(
        [
            100 * np.mean(power_loss_per_sample(power_matrix, topk_indices, k=k) > threshold_db)
            for k in range(1, max_k + 1)
        ]
    )
