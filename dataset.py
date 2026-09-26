"""
dataset.py - Audio Dataset Pipeline for Music Genre Classification (GTZAN)

Handles:
1. Audio loading and sanitization (gracefully handles known corrupt files like jazz.00054.wav).
2. Track-level train/val/test splitting to strictly prevent data leakage across 3s slices.
3. Audio segmentation: 30s tracks -> 3s segments (10 slices per track).
4. Feature extraction: On-the-fly Log-Mel Spectrograms using torchaudio.
5. Data augmentations: SpecAugment (Time & Frequency masking) and Gaussian noise.
"""

import os
import glob
import random
import warnings
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

GENRES: List[str] = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

GENRE_TO_IDX: Dict[str, int] = {genre: idx for idx, genre in enumerate(GENRES)}
IDX_TO_GENRE: Dict[int, str] = {idx: genre for idx, genre in enumerate(GENRES)}

# Standard GTZAN parameters
TARGET_SAMPLE_RATE: int = 22050
SEGMENT_DURATION_SEC: float = 3.0
TRACK_DURATION_SEC: float = 30.0
SAMPLES_PER_SEGMENT: int = int(TARGET_SAMPLE_RATE * SEGMENT_DURATION_SEC)  # 66,150 samples
SEGMENTS_PER_TRACK: int = int(TRACK_DURATION_SEC // SEGMENT_DURATION_SEC)  # 10 segments


class SpecAugment(nn.Module):
    """
    SpecAugment for Log-Mel Spectrograms:
    Applies frequency masking and time masking to prevent overfitting.
    """
    def __init__(self, freq_mask_param: int = 16, time_mask_param: int = 24):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec shape: (channels, n_mels, time)
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec


class LogMelExtractor(nn.Module):
    """
    Converts 1D waveform to a Log-Mel Spectrogram.
    Output shape for 3s segment (66,150 samples): (1, n_mels=128, time=130)
    """
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 20.0,
        f_max: float = 11025.0
    ):
        super().__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform shape: (channels, samples)
        mel = self.mel_spectrogram(waveform)
        log_mel = self.amplitude_to_db(mel)
        # Instance normalization (zero mean, unit variance per sample)
        norm_log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return norm_log_mel


def find_gtzan_audio_files(data_dir: str) -> Dict[str, List[str]]:
    """
    Recursively scans for GTZAN audio files.
    Supports directories like:
      - data_dir/genres_original/genre/*.wav
      - data_dir/genre/*.wav
      - data_dir/*.wav
    """
    genre_files: Dict[str, List[str]] = {g: [] for g in GENRES}

    # Possible subfolder search locations
    search_paths = [
        os.path.join(data_dir, "genres_original"),
        os.path.join(data_dir, "genres"),
        data_dir
    ]

    for base in search_paths:
        if not os.path.exists(base):
            continue
        for genre in GENRES:
            # Check genre subfolder
            pattern = os.path.join(base, genre, "*.wav")
            matches = glob.glob(pattern)
            if not matches:
                # Some datasets have hiphop as hip-hop
                alt_genre = "hip-hop" if genre == "hiphop" else genre
                pattern = os.path.join(base, alt_genre, "*.wav")
                matches = glob.glob(pattern)
            
            for f in matches:
                if f not in genre_files[genre]:
                    # Verify file is not corrupt (e.g., notorious jazz.00054.wav)
                    try:
                        import soundfile as sf
                        _ = sf.info(f)
                        genre_files[genre].append(f)
                    except Exception as e:
                        warnings.warn(f"[Data Sanitization] Skipped unreadable/corrupt audio: {os.path.basename(f)} ({e})")

    return genre_files


def create_track_level_splits(
    genre_files: Dict[str, List[str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]], List[Tuple[str, int, int]]]:
    """
    Performs track-level splitting before slicing into segments to prevent data leakage.
    Returns lists of (file_path, genre_idx, track_id).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"
    random.seed(seed)

    train_tracks: List[Tuple[str, int, int]] = []
    val_tracks: List[Tuple[str, int, int]] = []
    test_tracks: List[Tuple[str, int, int]] = []

    global_track_id = 0

    for genre, files in genre_files.items():
        sorted_files = sorted(files)  # Ensure deterministic ordering
        random.shuffle(sorted_files)

        n_total = len(sorted_files)
        if n_total >= 3:
            n_val = max(1, int(n_total * val_ratio))
            n_test = max(1, int(n_total * test_ratio))
            n_train = n_total - n_val - n_test
        else:
            n_train = n_total
            n_val = 0
            n_test = 0

        genre_idx = GENRE_TO_IDX[genre]

        for i, f in enumerate(sorted_files):
            item = (f, genre_idx, global_track_id)
            global_track_id += 1
            if i < n_train:
                train_tracks.append(item)
            elif i < n_train + n_val:
                val_tracks.append(item)
            else:
                test_tracks.append(item)

    return train_tracks, val_tracks, test_tracks


class GTZANSliceDataset(Dataset):
    """
    Dataset of 3-second segments derived from track-level splits.
    Each item returns:
      - spec: (1, 128, 130) Log-Mel spectrogram
      - label: genre index (0-9)
      - track_id: unique integer identifying the source song (for majority voting)
      - slice_idx: segment index (0-9) within the track
    """
    def __init__(
        self,
        track_list: List[Tuple[str, int, int]],
        is_train: bool = False,
        use_augmentation: bool = True
    ):
        self.is_train = is_train
        self.use_augmentation = use_augmentation and is_train
        self.feature_extractor = LogMelExtractor()
        self.spec_augment = SpecAugment() if self.use_augmentation else None

        # Build segment index mapping: (track_path, genre_idx, track_id, segment_idx)
        self.samples: List[Tuple[str, int, int, int]] = []
        for file_path, genre_idx, track_id in track_list:
            for seg_idx in range(SEGMENTS_PER_TRACK):
                self.samples.append((file_path, genre_idx, track_id, seg_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio_segment(self, file_path: str, seg_idx: int) -> torch.Tensor:
        """
        Loads the specific 3s audio window using soundfile (cross-platform).
        If file is corrupted, returns zeros gracefully.
        """
        start_frame = seg_idx * SAMPLES_PER_SEGMENT
        num_frames = SAMPLES_PER_SEGMENT

        try:
            import soundfile as sf
            data, sr = sf.read(file_path, start=start_frame, stop=start_frame + num_frames, dtype="float32")
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T)

            # Resample if sample rate deviates
            if sr != TARGET_SAMPLE_RATE:
                resampler = T.Resample(sr, TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)

            # Downmix to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Pad or trim to exact required samples
            if waveform.shape[-1] < SAMPLES_PER_SEGMENT:
                padding = SAMPLES_PER_SEGMENT - waveform.shape[-1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.shape[-1] > SAMPLES_PER_SEGMENT:
                waveform = waveform[:, :SAMPLES_PER_SEGMENT]

            return waveform

        except Exception as e:
            warnings.warn(f"Warning: Failed loading {file_path} (segment {seg_idx}): {e}")
            return torch.zeros((1, SAMPLES_PER_SEGMENT), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        file_path, genre_idx, track_id, seg_idx = self.samples[idx]

        waveform = self._load_audio_segment(file_path, seg_idx)

        # Waveform augmentation (slight noise & gain jitter during training)
        if self.use_augmentation:
            # Random gain (+/- 2 dB)
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain
            # Low-amplitude Gaussian noise
            if random.random() < 0.3:
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise

        # Extract Log-Mel Spectrogram -> Shape: (1, 128, 130)
        with torch.no_grad():
            spec = self.feature_extractor(waveform)

        # SpecAugment (Time & Frequency masking)
        if self.use_augmentation and self.spec_augment is not None:
            spec = self.spec_augment(spec)

        return spec, genre_idx, track_id, seg_idx


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Creates PyTorch DataLoaders for Train, Validation, and Test sets.
    """
    genre_files = find_gtzan_audio_files(data_dir)
    total_found = sum(len(v) for v in genre_files.values())

    if total_found == 0:
        raise FileNotFoundError(
            f"No GTZAN audio files found in '{data_dir}'. "
            "Please ensure the dataset is placed with genre subfolders."
        )

    train_tracks, val_tracks, test_tracks = create_track_level_splits(genre_files, seed=seed)

    train_dataset = GTZANSliceDataset(train_tracks, is_train=True, use_augmentation=True)
    val_dataset = GTZANSliceDataset(val_tracks, is_train=False, use_augmentation=False)
    test_dataset = GTZANSliceDataset(test_tracks, is_train=False, use_augmentation=False)

    has_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=has_cuda
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=has_cuda
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=has_cuda
    )

    stats = {
        "train_tracks": len(train_tracks),
        "val_tracks": len(val_tracks),
        "test_tracks": len(test_tracks),
        "train_segments": len(train_dataset),
        "val_segments": len(val_dataset),
        "test_segments": len(test_dataset)
    }

    return train_loader, val_loader, test_loader, stats


if __name__ == "__main__":
    print("GTZAN Dataset Pipeline Module")
    print(f"Supported Genres ({len(GENRES)}): {', '.join(GENRES)}")
    print(f"Segment length: {SEGMENT_DURATION_SEC}s ({SAMPLES_PER_SEGMENT} samples at {TARGET_SAMPLE_RATE}Hz)")
    
    # Test feature extraction on synthetic waveform
    extractor = LogMelExtractor()
    synthetic_wave = torch.randn(1, SAMPLES_PER_SEGMENT)
    spec = extractor(synthetic_wave)
    print(f"Synthetic Audio Waveform Shape: {synthetic_wave.shape}")
    print(f"Extracted Log-Mel Spectrogram Shape: {spec.shape} (Channels, Mels, Time Frames)")
    assert spec.shape[1] == 128 and spec.shape[2] == 130, "Unexpected spectrogram dimensions!"
    print("Feature extractor verification PASSED.")
