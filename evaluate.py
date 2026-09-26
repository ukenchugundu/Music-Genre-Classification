"""
evaluate.py - Comprehensive Evaluation & Robustness Analysis

Covers:
1. Standard Test Evaluation: Accuracy, Macro F1, Precision, Recall, per-genre breakdown.
2. Normalized Confusion Matrix Heatmap generation and saving.
3. Noisy & Real-World Robustness: Evaluates performance across an Additive White Gaussian Noise (AWGN)
   Signal-to-Noise Ratio (SNR) sweep: [Clean, 20dB, 10dB, 5dB, 0dB] and plots degradation curves.
4. Efficiency Profiling: Exact parameters, checkpoint file size (MB), FLOPs/MACs, inference latency, and Real-Time Factor (RTF).
"""

import os
import time
import argparse
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from dataset import get_dataloaders, GENRES, SAMPLES_PER_SEGMENT
from model import create_model, profile_model, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Music Genre Classification Model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to GTZAN dataset")
    default_ckpt = "./checkpoints/best_baseline.pth" if os.path.exists("./checkpoints/best_baseline.pth") else "./checkpoints/best_resaudionet.pth"
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Checkpoint path")
    parser.add_argument("--reports_dir", type=str, default="./reports", help="Directory to save evaluation plots")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def add_awgn_noise(specs: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Injects Additive White Gaussian Noise (AWGN) to Log-Mel spectrograms at a specific SNR (dB).
    Higher SNR = cleaner audio; 0dB = noise equal to signal power.
    """
    if snr_db == float("inf"):
        return specs

    signal_power = torch.mean(specs ** 2, dim=(-2, -1), keepdim=True)
    # SNR = 10 * log10(P_signal / P_noise) => P_noise = P_signal / (10^(SNR/10))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn_like(specs) * torch.sqrt(noise_power + 1e-8)
    return specs + noise


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    snr_db: float = float("inf")
) -> Tuple[List[int], List[int], float, float]:
    model.eval()
    all_preds = []
    all_labels = []

    for batch in test_loader:
        specs, labels = batch[0].to(device), batch[1].to(device)

        if snr_db != float("inf"):
            specs = add_awgn_noise(specs, snr_db)

        logits = model(specs)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        targets = labels.cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(targets)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return all_labels, all_preds, acc, macro_f1


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], classes: List[str], save_path: str):
    """
    Computes and plots a normalized confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )
    plt.title("Normalized Confusion Matrix - Music Genre Classification", fontsize=14, pad=15)
    plt.xlabel("Predicted Genre", fontsize=12)
    plt.ylabel("True Genre", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Confusion Matrix to '{save_path}'")


def plot_noise_robustness(snr_levels: List[float], accuracies: List[float], save_path: str):
    """
    Plots accuracy degradation across different SNR levels.
    """
    snr_labels = ["Clean (inf)" if s == float("inf") else f"{int(s)} dB" for s in snr_levels]

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(snr_levels)), [a * 100 for a in accuracies], marker="o", color="#2b5c8f", linewidth=2.5, markersize=8)
    plt.title("Noise Robustness: Model Accuracy vs. Signal-to-Noise Ratio (SNR)", fontsize=13, pad=12)
    plt.xlabel("Noise Level (AWGN SNR in dB)", fontsize=11)
    plt.ylabel("Classification Accuracy (%)", fontsize=11)
    plt.xticks(range(len(snr_levels)), snr_labels)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, 100)

    for i, acc in enumerate(accuracies):
        plt.annotate(f"{acc*100:.1f}%", (i, acc * 100 + 2.5), ha="center", fontsize=10, weight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Noise Robustness Curve to '{save_path}'")


def benchmark_efficiency(model: nn.Module, device: torch.device, input_size: Tuple[int, ...] = (1, 1, 128, 130)) -> Dict[str, Any]:
    """
    Measures latency, real-time factor, and parameter efficiency.
    """
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)

    # Latency test across 100 runs
    latencies = []
    runs = 100
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        latencies.append((time.perf_counter() - start) * 1000.0)  # ms

    mean_latency_ms = np.mean(latencies)
    std_latency_ms = np.std(latencies)

    # Real-Time Factor (RTF) for a 3.0s audio segment
    rtf = (mean_latency_ms / 1000.0) / 3.0

    return {
        "mean_latency_ms": round(float(mean_latency_ms), 3),
        "std_latency_ms": round(float(std_latency_ms), 3),
        "real_time_factor": round(float(rtf), 5),
        "is_real_time": rtf < 1.0
    }


def run_full_evaluation():
    args = parse_args()
    os.makedirs(args.reports_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}'. Run train.py first!")

    print(f"\n[Evaluation] Loading model from '{args.checkpoint}' onto {device}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint.get("model_name", "resaudionet")
    model = create_model(model_name, num_classes=len(GENRES)).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    # 1. Profile Model Efficiency
    profile = profile_model(model)
    eff = benchmark_efficiency(model, device)
    checkpoint_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)

    print("\n" + "=" * 65)
    print("                EFFICIENCY & COMPLEXITY METRICS")
    print("=" * 65)
    print(f"  Architecture:           {profile['model_name']}")
    print(f"  Trainable Parameters:   {profile['trainable_parameters']:,}")
    print(f"  Checkpoint Size:        {checkpoint_size_mb:.2f} MB (Constraint <= 12MB: {'PASS' if checkpoint_size_mb <= 12 else 'FAIL'})")
    print(f"  Complexity:             {profile['estimated_macs']}")
    print(f"  Inference Latency:      {eff['mean_latency_ms']} ms ± {eff['std_latency_ms']} ms (per 3s segment)")
    print(f"  Real-Time Factor (RTF): {eff['real_time_factor']:.5f} (RTF << 1.0 confirms real-time capability)")
    print("=" * 65 + "\n")

    # 2. Check for Test Data
    if not os.path.exists(args.data_dir):
        print(f"[Notice] Dataset '{args.data_dir}' not found locally.")
        print("To run the test set evaluation and noise robustness suite on actual GTZAN:")
        print("  python evaluate.py --data_dir ./data --checkpoint checkpoints/best_resaudionet.pth")
        return

    _, _, test_loader, stats = get_dataloaders(args.data_dir, batch_size=32)
    print(f"[Evaluation] Evaluating on held-out Test Set ({stats['test_tracks']} tracks, {stats['test_segments']} segments)...")

    # 3. Clean Test Set Evaluation
    y_true, y_pred, clean_acc, clean_f1 = evaluate_test_set(model, test_loader, device)

    print("\n" + "=" * 65)
    print(f"  CLEAN TEST SET METRICS: Accuracy = {clean_acc*100:.2f}%, Macro F1 = {clean_f1:.4f}")
    print("=" * 65)
    print("\nPer-Genre Classification Report:")
    print(classification_report(y_true, y_pred, target_names=GENRES, digits=4))

    # Save Confusion Matrix
    cm_path = os.path.join(args.reports_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, GENRES, cm_path)

    # 4. Noisy & Real-World Robustness Sweep
    print("\n[Evaluation] Conducting Real-World Noise Robustness Sweep (AWGN)...")
    snr_levels = [float("inf"), 20.0, 10.0, 5.0, 0.0]
    accuracies = []

    for snr in snr_levels:
        label = "Clean" if snr == float("inf") else f"{int(snr)}dB"
        _, _, snr_acc, _ = evaluate_test_set(model, test_loader, device, snr_db=snr)
        accuracies.append(snr_acc)
        print(f"  - Noise SNR = {label:<8}: Accuracy = {snr_acc*100:.2f}%")

    noise_plot_path = os.path.join(args.reports_dir, "noise_robustness.png")
    plot_noise_robustness(snr_levels, accuracies, noise_plot_path)

    print("\n[Evaluation] Full Evaluation Completed Successfully!")


if __name__ == "__main__":
    run_full_evaluation()
