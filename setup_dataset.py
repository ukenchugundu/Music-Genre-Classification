"""
setup_dataset.py - Dataset Setup and Downloader for GTZAN

Handles:
1. Automatic download of GTZAN dataset via torchaudio or direct mirror.
2. Generating a sample dataset (10 genres x 5 sample tracks = 50 tracks) for immediate local experimentation
   if the full 1.2 GB download is skipped or taking too long.
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf

from dataset import GENRES, TARGET_SAMPLE_RATE

def create_synthetic_gtzan_dataset(output_dir: str = "./data/genres_original", tracks_per_genre: int = 5, duration_sec: int = 30):
    """
    Generates synthetic audio tracks with distinct harmonic/percussive signatures for each genre
    so the entire pipeline (train, test, evaluate, confusion matrix) can be executed and visualized immediately.
    """
    os.makedirs(output_dir, exist_ok=True)
    sr = TARGET_SAMPLE_RATE
    total_samples = sr * duration_sec
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    # Base frequencies and rhythmic profiles per genre to give distinct features
    genre_profiles = {
        "blues": [196.0, 233.08, 261.63],      # G blues scale
        "classical": [261.63, 329.63, 392.0],  # Major triad, smooth strings
        "country": [293.66, 369.99, 440.0],    # D major, acoustic guitar profile
        "disco": [130.81, 261.63, 523.25],     # Four-on-the-floor beat simulation
        "hiphop": [65.41, 98.0, 130.81],       # Heavy sub-bass
        "jazz": [220.0, 261.63, 311.13, 392.0],# Minor 7th chord
        "metal": [82.41, 123.47, 164.81],      # Low E, heavy distortion simulation
        "pop": [329.63, 392.0, 493.88],        # Catchy melody
        "reggae": [174.61, 220.0, 261.63],     # Off-beat syncopation
        "rock": [110.0, 164.81, 220.0]         # Power chord A5
    }

    print(f"\n[Setup] Creating sample GTZAN dataset ({len(GENRES)} genres x {tracks_per_genre} tracks = {len(GENRES) * tracks_per_genre} tracks)...")
    for genre in GENRES:
        genre_folder = os.path.join(output_dir, genre)
        os.makedirs(genre_folder, exist_ok=True)
        freqs = genre_profiles.get(genre, [440.0])

        for track_idx in range(tracks_per_genre):
            filename = f"{genre}.{track_idx:05d}.wav"
            file_path = os.path.join(genre_folder, filename)

            # Synthesize harmonic content
            signal = np.zeros(total_samples, dtype=np.float32)
            for f in freqs:
                harmonic = np.sin(2 * np.pi * f * t)
                signal += harmonic / len(freqs)

            # Add rhythmic envelope
            beat_freq = 2.0 if genre in ["disco", "pop", "rock", "metal"] else 1.0
            rhythm = 0.7 + 0.3 * np.sin(2 * np.pi * beat_freq * t)
            signal = signal * rhythm

            # Slight noise texture
            signal += 0.05 * np.random.randn(total_samples).astype(np.float32)

            # Normalize to [-0.9, 0.9]
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = (signal / max_val) * 0.85

            sf.write(file_path, signal, sr)

    print(f"[Setup] Sample dataset successfully created in '{output_dir}'.\n")


def try_download_gtzan(root_dir: str = "./data"):
    """
    Attempts to download official GTZAN dataset via torchaudio.
    """
    print(f"[Setup] Attempting to download GTZAN dataset (~1.2 GB) into '{root_dir}'...")
    import torchaudio
    try:
        dataset = torchaudio.datasets.GTZAN(root=root_dir, download=True)
        print(f"[Setup] Successfully downloaded GTZAN dataset ({len(dataset)} items).")
        return True
    except Exception as e:
        print(f"[Warning] Official GTZAN download failed or timed out: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup GTZAN dataset")
    parser.add_argument("--download", action="store_true", help="Download official 1.2GB GTZAN from source")
    parser.add_argument("--sample", action="store_true", default=True, help="Create a sample dataset for immediate testing")
    parser.add_argument("--tracks_per_genre", type=int, default=5, help="Number of tracks per genre for sample dataset")
    args = parser.parse_args()

    data_dir = "./data/genres_original"
    if args.download:
        success = try_download_gtzan("./data")
        if not success:
            print("[Setup] Falling back to sample dataset generation...")
            create_synthetic_gtzan_dataset(data_dir, tracks_per_genre=args.tracks_per_genre)
    else:
        create_synthetic_gtzan_dataset(data_dir, tracks_per_genre=args.tracks_per_genre)


if __name__ == "__main__":
    main()
