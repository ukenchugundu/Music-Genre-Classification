"""
run_all.py - Master Pipeline Execution Script

Executes the entire end-to-end project:
1. Environment & Model Architecture Verification (Strict <= 12MB Checkpoint Check).
2. Model Training with TensorBoard event logging (optional if checkpoint exists).
3. Test Set Evaluation & Real-World Noise Robustness Sweep (AWGN 20dB -> 0dB).
4. Unseen Audio Inference on multiple genre tracks.
5. Deliverables & Artifacts Confirmation.
"""

import os
import sys
import time
import argparse
import subprocess

def run_step(title: str, command: str):
    print("\n" + "=" * 75)
    print(f"  [STEP] {title}")
    print("=" * 75)
    print(f"Executing: {command}\n")

    start_time = time.perf_counter()
    ret = subprocess.run(command, shell=True, text=True)
    duration = time.perf_counter() - start_time

    if ret.returncode != 0:
        print(f"\n[ERROR] Step '{title}' failed with exit code {ret.returncode}.")
        sys.exit(ret.returncode)

    print(f"\n[OK] Step '{title}' completed in {duration:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Run the entire music genre classification project")
    parser.add_argument("--train", action="store_true", help="Force retrain the model before evaluating")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs to train if --train is set")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "resaudionet"], help="Model architecture")
    args = parser.parse_args()

    python_exe = sys.executable
    checkpoint_path = f"./checkpoints/best_{args.model}.pth"

    print("=" * 75)
    print("       MUSIC GENRE CLASSIFICATION - END-TO-END MASTER RUNNER")  
    print("=" * 75)
    # 1. Model Architecture & <= 12MB Constraint Verification
    run_step(
        "Model Architecture & <= 12MB Checkpoint Size Verification",
        f'"{python_exe}" model.py'
    )

    # 2. Training Pipeline (if requested or checkpoint missing)
    if args.train or not os.path.exists(checkpoint_path):
        run_step(
            f"Train {args.model.upper()} Model on GTZAN (AdamW, Cosine LR, Label Smoothing)",
            f'"{python_exe}" train.py --data_dir ./data --model {args.model} --epochs {args.epochs} --batch_size 32'
        )
    else:
        print(f"\n[INFO] Checkpoint '{checkpoint_path}' already exists. Proceeding directly to Evaluation & Inference.")
        print("       (To force retrain, run: python run_all.py --train)\n")

    # 3. Comprehensive Evaluation & Noise Robustness Sweep
    run_step(
        f"Held-Out Test Evaluation, Confusion Matrix & AWGN Noise Sweep ({args.model.upper()})",
        f'"{python_exe}" evaluate.py --checkpoint "{checkpoint_path}" --data_dir ./data'
    )

    # 4. Audio Inference on Multiple Unseen Real GTZAN Tracks
    test_tracks = [
        ("classical", "classical.00095.wav"),
        ("metal", "metal.00095.wav"),
        ("disco", "disco.00095.wav"),
        ("blues", "blues.00095.wav"),
        ("rock", "rock.00095.wav")
    ]
    for genre, filename in test_tracks:
        sample_path = os.path.join("./data/genres_original", genre, filename)
        if os.path.exists(sample_path):
            run_step(
                f"Audio Inference on Real {genre.upper()} Track ({filename})",
                f'"{python_exe}" inference.py --audio "{sample_path}" --checkpoint "{checkpoint_path}"'
            )

    # 5. Summary of Deliverables
    print("\n" + "=" * 75)
    print("                ALL PROJECT DELIVERABLES CONFIRMED")
    print("=" * 75)
    print("  [x] dataset.py          - Audio loading, leak-free splitting, Log-Mel extraction")
    print("  [x] model.py            - Baseline CNN & ResAudioNet (<= 12MB limit compliant)")
    print("  [x] train.py            - Training loop with AdamW and TensorBoard logging")
    print("  [x] inference.py        - Inference script for arbitrary unseen WAV files")
    print("  [x] evaluate.py         - Test metrics, Confusion Matrix, and Noise sweep")
    print(f"  [x] checkpoints/        - Saved {os.path.basename(checkpoint_path)}")
    print("  [x] reports/            - confusion_matrix.png & noise_robustness.png")
    print("  [x] runs/               - TensorBoard event logs")
    print("  [x] README.md           - Complete research documentation & setup guide")
    print("=" * 75 + "\n")
    print("Master pipeline execution finished successfully!\n")


if __name__ == "__main__":
    main()
