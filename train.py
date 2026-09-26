"""
train.py - Training Pipeline for Music Genre Classification

Features:
1. Modular architecture selection (BaselineAudioCNN vs ResAudioNet).
2. CrossEntropyLoss with Label Smoothing (reduces overconfidence on ambiguous genres).
3. AdamW Optimizer with Cosine Annealing learning rate schedule.
4. Validation evaluation: Segment-level accuracy and Whole-song Voting accuracy.
5. TensorBoard logging (loss, accuracy, F1 score, learning rate).
6. Automatic checkpointing with strict size verification (<= 12MB).
7. Built-in smoke test mode for immediate validation without downloading full GTZAN.
"""

import os
import time
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score

from dataset import get_dataloaders, GENRES, SAMPLES_PER_SEGMENT, TARGET_SAMPLE_RATE
from model import create_model, profile_model, get_checkpoint_size_mb


def parse_args():
    parser = argparse.ArgumentParser(description="Train Music Genre Classification Model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to GTZAN dataset")
    parser.add_argument("--model", type=str, default="resaudionet", choices=["baseline", "resaudionet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./runs", help="TensorBoard log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick 2-epoch test on synthetic data")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        specs, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float]:
    """
    Evaluates both:
      1. Segment-level accuracy & macro F1
      2. Whole-song level accuracy (aggregating probabilities across segments by track_id)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    # For whole-song voting aggregation
    track_logits: Dict[int, list] = {}
    track_labels: Dict[int, int] = {}

    for batch in loader:
        specs, labels = batch[0].to(device), batch[1].to(device)
        track_ids = batch[2].tolist() if len(batch) > 2 else list(range(len(labels)))

        logits = model(specs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * specs.size(0)

        probs = torch.softmax(logits, dim=1).cpu()
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        targets = labels.cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(targets)

        # Group probabilities by track_id for song-level voting
        for i, tid in enumerate(track_ids):
            if tid not in track_logits:
                track_logits[tid] = []
                track_labels[tid] = targets[i]
            track_logits[tid].append(probs[i])

    total_samples = len(all_labels)
    eval_loss = total_loss / total_samples
    segment_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Song-level aggregation (mean probability pooling across all segments of a song)
    song_correct = 0
    total_songs = len(track_logits)
    for tid, prob_list in track_logits.items():
        stacked_probs = torch.stack(prob_list, dim=0)  # (n_segments, num_classes)
        mean_prob = torch.mean(stacked_probs, dim=0)   # (num_classes,)
        song_pred = torch.argmax(mean_prob).item()
        if song_pred == track_labels[tid]:
            song_correct += 1

    song_acc = song_correct / total_songs if total_songs > 0 else 0.0

    return eval_loss, segment_acc, song_acc, macro_f1


def run_training():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Environment] Using Device: {device}")
    if device.type == "cuda":
        print(f"[Environment] GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 1. Load Data
    if args.smoke_test or not os.path.exists(args.data_dir):
        if not args.smoke_test:
            print(f"[Warning] Dataset path '{args.data_dir}' not found. Falling back to synthetic smoke test mode.")
        print("[Data] Generating synthetic Log-Mel spectrograms for pipeline verification...")
        # 100 synthetic segments: (batch, 1, 128, 130)
        n_samples = 120
        syn_x = torch.randn(n_samples, 1, 128, 130)
        syn_y = torch.randint(0, len(GENRES), (n_samples,))
        syn_tracks = torch.tensor([i // 10 for i in range(n_samples)])
        syn_segs = torch.tensor([i % 10 for i in range(n_samples)])
        syn_dataset = TensorDataset(syn_x, syn_y, syn_tracks, syn_segs)
        train_loader = DataLoader(syn_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(syn_dataset, batch_size=args.batch_size, shuffle=False)
        args.epochs = min(args.epochs, 2)
    else:
        print(f"[Data] Loading GTZAN from '{args.data_dir}' with track-level splitting...")
        train_loader, val_loader, _, stats = get_dataloaders(args.data_dir, batch_size=args.batch_size)
        print(f"[Data] Loaded: {stats['train_tracks']} train tracks ({stats['train_segments']} segments), "
              f"{stats['val_tracks']} val tracks ({stats['val_segments']} segments)")

    # 2. Initialize Model
    print(f"\n[Model] Initializing architecture: '{args.model}'...")
    model = create_model(args.model, num_classes=len(GENRES)).to(device)
    profile = profile_model(model)
    print(f"[Model] Trainable Parameters: {profile['trainable_parameters']:,}")
    print(f"[Model] Checkpoint Size: {profile['checkpoint_size_mb']:.2f} MB")
    print(f"[Model] Size <= 12MB Constraint: {'PASSED [OK]' if profile['is_size_compliant_le_12mb'] else 'FAILED [X]'}")
    assert profile["is_size_compliant_le_12mb"], "Model checkpoint size exceeds 12MB limit!"

    # 3. Optimization Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. TensorBoard Writer
    run_name = f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))

    print(f"\n[Training] Starting {args.epochs} epochs (TensorBoard logs: {args.log_dir}/{run_name})...")
    print("-" * 80)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Seg Acc':<14} {'Val Song Acc':<14} {'Val F1':<10}")
    print("-" * 80)

    best_val_f1 = 0.0
    best_checkpoint_path = os.path.join(args.output_dir, f"best_{args.model}.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_seg_acc, val_song_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # TensorBoard Logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train_Segment", train_acc, epoch)
        writer.add_scalar("Accuracy/Val_Segment", val_seg_acc, epoch)
        writer.add_scalar("Accuracy/Val_Song", val_song_acc, epoch)
        writer.add_scalar("F1/Validation_Macro", val_f1, epoch)
        writer.add_scalar("Optimization/Learning_Rate", current_lr, epoch)

        print(f"{epoch:<8} {train_loss:<12.4f} {train_acc*100:<11.2f}% {val_loss:<12.4f} {val_seg_acc*100:<13.2f}% {val_song_acc*100:<13.2f}% {val_f1:<10.4f}")

        # Checkpoint if validation F1 improves
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_song_acc": val_song_acc,
                "genres": GENRES
            }, best_checkpoint_path)

    writer.close()

    # Verify saved checkpoint size on disk
    if os.path.exists(best_checkpoint_path):
        saved_size_mb = os.path.getsize(best_checkpoint_path) / (1024 * 1024)
        print("\n" + "=" * 60)
        print(f"[Checkpoint] Successfully saved: {best_checkpoint_path}")
        print(f"[Checkpoint] Final File Size: {saved_size_mb:.2f} MB")
        print(f"[Checkpoint] 12MB Constraint Check: {'PASSED [OK]' if saved_size_mb <= 12.0 else 'FAILED [X]'}")
        print(f"[Checkpoint] Best Val Macro F1: {best_val_f1:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    run_training()
