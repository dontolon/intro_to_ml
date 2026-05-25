"""Visualization helpers for the GPS-based beam prediction workshop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def _new_axis(figsize=(8, 5)):
    _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_beam_distribution(df: pd.DataFrame, title: str = "Distribuzione dei beam"):
    """Plot how often each best beam appears."""
    ax = _new_axis((10, 4))
    counts = df["beam_index"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel("Beam index")
    ax.set_ylabel("Numero di sample")
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_gps_by_beam(df: pd.DataFrame, title: str = "GPS colorato per best beam"):
    """Scatter plot of GPS coordinates colored by best beam."""
    ax = _new_axis((7, 6))
    sc = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["beam_index"],
        cmap="turbo",
        s=18,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(sc, ax=ax, label="Beam index")
    return ax


def plot_gps_by_sequence(df: pd.DataFrame, title: str = "GPS colorato per sequenza"):
    """Scatter plot of GPS coordinates colored by sequence id."""
    ax = _new_axis((7, 6))
    sc = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["seq_index"],
        cmap="viridis",
        s=18,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(sc, ax=ax, label="Seq index")
    return ax


def plot_power_profile(
    power_row: np.ndarray,
    true_beam: int,
    predicted_beams: np.ndarray | None = None,
    title: str = "Profilo di potenza sui 64 beam",
):
    """Show the power measured on every beam for one sample."""
    ax = _new_axis((10, 4))
    beams = np.arange(len(power_row))
    ax.plot(beams, power_row, marker="o", linewidth=1.5, color="#334155")
    ax.axvline(true_beam, color="#16a34a", linewidth=2, label=f"True best beam: {true_beam}")
    if predicted_beams is not None:
        for rank, beam in enumerate(predicted_beams[:3], start=1):
            ax.axvline(
                int(beam),
                color="#dc2626",
                alpha=0.35,
                linestyle="--",
                label=f"Pred Top-{rank}: {int(beam)}",
            )
    ax.set_title(title)
    ax.set_xlabel("Beam index")
    ax.set_ylabel("Received power")
    ax.grid(alpha=0.25)
    ax.legend()
    return ax


def plot_train_test_distribution(y_train: np.ndarray, y_test: np.ndarray):
    """Compare beam distributions after the split."""
    ax = _new_axis((10, 4))
    labels = np.arange(64)
    train_counts = np.bincount(y_train, minlength=64) / len(y_train)
    test_counts = np.bincount(y_test, minlength=64) / len(y_test)
    ax.plot(labels, train_counts, label="Train", linewidth=2)
    ax.plot(labels, test_counts, label="Test", linewidth=2)
    ax.set_title("Distribuzione delle label in train e test")
    ax.set_xlabel("Beam index")
    ax.set_ylabel("Frequenza relativa")
    ax.grid(alpha=0.25)
    ax.legend()
    return ax


def plot_topk_curve(results: dict[str, np.ndarray], title: str = "Top-K accuracy"):
    """Plot Top-1 through Top-K accuracy for multiple models."""
    ax = _new_axis((8, 5))
    for name, values in results.items():
        ks = np.arange(1, len(values) + 1)
        ax.plot(ks, values * 100, marker="o", linewidth=2, label=name)
    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel("Top-K accuracy (%)")
    ax.set_xticks(np.arange(1, max(len(v) for v in results.values()) + 1))
    ax.grid(alpha=0.25)
    ax.legend()
    return ax


def plot_power_loss_curve(results: dict[str, np.ndarray], title: str = "Mean power loss"):
    """Plot mean power loss for Top-1 through Top-K."""
    ax = _new_axis((8, 5))
    for name, values in results.items():
        ks = np.arange(1, len(values) + 1)
        ax.plot(ks, values, marker="o", linewidth=2, label=name)
    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel("Mean power loss (dB)")
    ax.set_xticks(np.arange(1, max(len(v) for v in results.values()) + 1))
    ax.grid(alpha=0.25)
    ax.legend()
    return ax


def plot_exceedance_curve(results: dict[str, np.ndarray], threshold_db: float = 3.0):
    """Plot percentage of samples above a power-loss threshold."""
    ax = _new_axis((8, 5))
    for name, values in results.items():
        ks = np.arange(1, len(values) + 1)
        ax.plot(ks, values, marker="o", linewidth=2, label=name)
    ax.set_title(f"Sample con power loss > {threshold_db:g} dB")
    ax.set_xlabel("K")
    ax.set_ylabel("Sample sopra soglia (%)")
    ax.set_xticks(np.arange(1, max(len(v) for v in results.values()) + 1))
    ax.grid(alpha=0.25)
    ax.legend()
    return ax


def plot_power_loss_histogram(loss_top1: np.ndarray, loss_top10: np.ndarray, model_name: str):
    """Compare Top-1 and Top-10 power-loss distributions."""
    ax = _new_axis((8, 5))
    ax.hist(loss_top1, bins=30, alpha=0.65, label="Top-1", color="#ef4444")
    ax.hist(loss_top10, bins=30, alpha=0.65, label="Top-10", color="#22c55e")
    ax.set_title(f"Distribuzione power loss - {model_name}")
    ax.set_xlabel("Power loss (dB)")
    ax.set_ylabel("Numero di sample")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return ax


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion matrix"):
    """Plot a compact confusion matrix for all beam indices."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(64))
    ax = _new_axis((9, 7))
    sns.heatmap(cm, cmap="Blues", cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Beam predetto")
    ax.set_ylabel("Beam vero")
    return ax


def plot_prediction_errors_on_gps(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Errori nello spazio GPS",
):
    """Show where predictions are correct or wrong on the GPS map."""
    ax = _new_axis((7, 6))
    errors = np.abs(y_true - y_pred)
    sc = ax.scatter(
        df_test["longitude"],
        df_test["latitude"],
        c=errors,
        cmap="magma",
        s=22,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(sc, ax=ax, label="|beam vero - beam predetto|")
    return ax


def plot_training_history(history):
    """Plot PyTorch loss and accuracy curves."""
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.train_loss, label="Train")
    axes[0].plot(history.test_loss, label="Val")
    axes[0].set_title("Loss durante training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(np.array(history.train_acc) * 100, label="Train")
    axes[1].plot(np.array(history.test_acc) * 100, label="Val")
    axes[1].set_title("Top-1 accuracy durante training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    return axes
