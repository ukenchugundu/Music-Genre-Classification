"""
inference.py - Inference Script for Unseen Audio Files (Music Genre Classification)

Capabilities:
1. Ingests any arbitrary WAV file regardless of duration, sampling rate, or channels.
2. Resamples to 22,050 Hz and converts stereo to mono.
3. Performs sliding-window segmentation (3.0s window with 50% overlap).
4. Extracts Log-Mel spectrograms and runs forward pass on saved checkpoint (<= 12MB).
5. Aggregates segment predictions using mean probability pooling for whole-track classification.
6. Returns top-k genre predictions, confidence scores, and segment consistency.
"""

import os
import json
import argparse
from typing import Dict, Any, List

import torch
import torchaudio
import torchaudio.transforms as T

from dataset import GENRES, TARGET_SAMPLE_RATE, SAMPLES_PER_SEGMENT, LogMelExtractor
from model import create_model


def load_audio_for_inference(file_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """
    Loads audio using soundfile (cross-platform), converts stereo to mono, and resamples to target_sr.
    Returns 1D waveform: (1, total_samples).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: '{file_path}'")

    import soundfile as sf
    try:
        data, sr = sf.read(file_path, dtype="float32")
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
        else:
            waveform = torch.from_numpy(data.T)            # (channels, samples)
    except Exception:
        waveform, sr = torchaudio.load(file_path)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Downmix stereo to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform


def slice_waveform(waveform: torch.Tensor, window_size: int, hop_size: int) -> torch.Tensor:
    """
    Slices 1D waveform into overlapping windows.
    Returns batch tensor: (num_segments, 1, window_size).
    """
    total_samples = waveform.shape[-1]

    # If audio is shorter than window_size, pad with reflection/zero
    if total_samples < window_size:
        pad_amount = window_size - total_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        total_samples = window_size

    # Sliding window
    segments = []
    start = 0
    while start + window_size <= total_samples:
        segment = waveform[:, start:start + window_size]
        segments.append(segment)
        start += hop_size

    # Ensure at least one window
    if not segments:
        segments.append(waveform[:, :window_size])

    return torch.stack(segments, dim=0)  # (N, 1, window_size)


class MusicGenreClassifier:
    """
    Inference Engine wrapping checkpoint loading, preprocessing, and prediction.
    """
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.feature_extractor = LogMelExtractor().to(self.device)
        self.genres = GENRES

        print(f"[Inference] Loading checkpoint from '{checkpoint_path}' onto {self.device}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_name = checkpoint.get("model_name", "resaudionet")
        if "genres" in checkpoint:
            self.genres = checkpoint["genres"]

        self.model = create_model(model_name, num_classes=len(self.genres)).to(self.device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_path: str, top_k: int = 3, hop_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Runs full inference on unseen WAV file.
        """
        waveform = load_audio_for_inference(audio_path)
        duration_sec = waveform.shape[-1] / TARGET_SAMPLE_RATE

        window_size = SAMPLES_PER_SEGMENT
        hop_size = int(window_size * hop_ratio)

        # Slice into overlapping segments
        segments = slice_waveform(waveform, window_size, hop_size).to(self.device)
        num_segments = segments.shape[0]

        # Extract Log-Mel spectrograms for all segments
        # segments: (N, 1, 66150) -> specs: (N, 1, 128, 130)
        specs = torch.stack([self.feature_extractor(seg) for seg in segments], dim=0)

        # Forward pass
        logits = self.model(specs)
        probs = torch.softmax(logits, dim=1)  # (N, num_classes)

        # Segment-level predictions
        seg_preds = torch.argmax(probs, dim=1).cpu().tolist()

        # Whole-song aggregation: Mean probability pooling
        mean_probs = torch.mean(probs, dim=0).cpu()  # (num_classes,)
        top_probs, top_indices = torch.topk(mean_probs, k=min(top_k, len(self.genres)))

        # Consistency: % of segments that agree with the top predicted genre
        top_class_idx = top_indices[0].item()
        consistency = sum(1 for p in seg_preds if p == top_class_idx) / num_segments

        top_predictions = [
            {
                "genre": self.genres[idx.item()],
                "confidence": round(prob.item() * 100, 2)
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

        full_distribution = {
            self.genres[i]: round(mean_probs[i].item() * 100, 2)
            for i in range(len(self.genres))
        }

        return {
            "audio_file": os.path.basename(audio_path),
            "duration_seconds": round(duration_sec, 2),
            "num_evaluated_segments": num_segments,
            "predicted_genre": top_predictions[0]["genre"],
            "confidence_percent": top_predictions[0]["confidence"],
            "segment_consistency_percent": round(consistency * 100, 2),
            "top_k_predictions": top_predictions,
            "full_genre_distribution": full_distribution
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Unseen Audio Files")
    parser.add_argument("--audio", type=str, required=True, help="Path to unseen WAV file")
    default_ckpt = "./checkpoints/best_baseline.pth" if os.path.exists("./checkpoints/best_baseline.pth") else "./checkpoints/best_resaudionet.pth"
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint file '{args.checkpoint}' not found.")
        print("Please train the model first with 'python train.py' or provide a valid checkpoint path.")
        exit(1)

    classifier = MusicGenreClassifier(checkpoint_path=args.checkpoint, device=args.device)
    result = classifier.predict(audio_path=args.audio, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 60)
        print(f"  AUDIO GENRE INFERENCE RESULT")
        print("=" * 60)
        print(f"  File: {result['audio_file']} ({result['duration_seconds']}s, {result['num_evaluated_segments']} segments)")
        print(f"  Predicted Genre:  {result['predicted_genre'].upper()}")
        print(f"  Confidence:       {result['confidence_percent']}%")
        print(f"  Consistency:      {result['segment_consistency_percent']}% of segments agreed")
        print("-" * 60)
        print("  Top Genre Probabilities:")
        for rank, item in enumerate(result['top_k_predictions'], 1):
            bar = "#" * int(item['confidence'] / 5)
            print(f"    {rank}. {item['genre']:<12} {item['confidence']:>6.2f}%  | {bar}")
        print("=" * 60 + "\n")
