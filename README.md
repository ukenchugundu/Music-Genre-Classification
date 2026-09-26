# End-to-End Music Genre Classification Pipeline (AI/ML R&D)

An end-to-end deep learning system for **Music Genre Classification** on the **GTZAN** audio dataset, engineered with a research and development mindset.

## 1. Project Overview & Research Motivation

### Problem Statement

Given an audio recording, classify it into one of 10 musical genres:
**Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, and Rock**.

### Research Mindset & Design Philosophy

Rather than treating this as a black-box competition, this project explores:

1. **Audio Representation Trade-offs:** Analyzing why Log-Mel Spectrograms capture polyphonic timbre better than MFCCs or raw waveforms.
2. **Strict Leak-Free Splitting:** Preventing the prevalent research pitfall in GTZAN literature where sliced segments of the same song leak between train and test sets.
3. **Parameter Efficiency:** Designing architectures strictly adhering to the **<= 12 MB** checkpoint limit while maximizing generalization.
4. **Real-World & Noisy Evaluation:** Stress-testing the model under Additive White Gaussian Noise (AWGN) sweeps (from clean to 0 dB).
5. **Computational Profiling:** Measuring MACs/FLOPs, latency per segment, and Real-Time Factor (RTF).

---

## 2. Literature Survey & Architectural Trade-offs

| Representation / Model | Parameters / Size | Strengths | Trade-offs in Music Classification |
| --- | --- | --- | --- |
| **Raw Waveforms (1D CNN)** | High compute | Preserves phase | Extreme dimensionality (661,500 samples/track); struggles on small datasets. |
| **MFCCs + GMM / SVM** | Low compute | Compact feature space | Discards higher-order harmonic details and polyphony via DCT decorrelation. |
| **Log-Mel + Baseline 2D CNN** | **390,890 params (1.50 MB)** | Ultra-fast, lightweight | Limited receptive field; captures local textural patterns. |
| **Log-Mel + ResAudioNet (SE)** | **2,821,866 params (10.83 MB)** | Deep feature hierarchy, channel attention (SE), skip connections | Well within 12MB limit (10.83 MB), superior generalization across complex polyphonic genres. |
| **Audio Spectrogram Transformer (AST)** | ~80M+ params (>300 MB) | SOTA on massive data | Violates 12MB limit; severely overfits small datasets like GTZAN without multi-gigabyte pre-training. |

---

## 3. Project Directory Structure

```text
Meeami_Audio_AI_Submission/
├── dataset.py                # Audio loader, corruption filtering, leak-free splits, Log-Mel extraction, SpecAugment
├── model.py                  # BaselineAudioCNN & ResAudioNet (<= 12MB constraint profiler)
├── train.py                  # Training pipeline, AdamW, CosineAnnealing, label smoothing, TensorBoard
├── inference.py              # Standalone inference on unseen WAV files (sliding window probability pooling)
├── evaluate.py               # Held-out test evaluation, confusion matrix, AWGN noise robustness sweep, RTF
├── run_all.py                # Master 1-click end-to-end execution script
├── requirements.txt          # Minimal reproducible dependencies
├── README.md                 # Project documentation & research report
├── checkpoints/              # Model weights: best_baseline.pth (1.50 MB), best_resaudionet.pth (10.83 MB)
├── reports/                  # confusion_matrix.png & noise_robustness.png
└── runs/                     # TensorBoard event logs
```

---

## 4. Setup & Installation

### Step 1: Clone & Navigate

```bash
git clone https://github.com/ukenchugundu/Music-Genre-Classification.git
cd Music-Genre-Classification
```

### Step 2: Create & Activate Virtual Environment

Using Python 3.10, 3.11, or 3.12:

#### Windows (PowerShell)

```powershell
# 1. Create venv
py -3.12 -m venv .venv

# 2. Activate terminal
.\.venv\Scripts\Activate.ps1
```

*(If script execution is disabled on your system, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` then activate).*

#### Windows (Command Prompt / cmd.exe)

```cmd
.\.venv\Scripts\activate.bat
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Dataset Setup (GTZAN)

1. Download the dataset from [Kaggle GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data).
2. Place the audio files inside `./data/genres_original/` so the structure looks like:

   ```text
   data/genres_original/
   ├── blues/
   │   ├── blues.00000.wav
   │   └── ...
   ├── classical/
   └── ...
   ```

*(Note: The data pipeline automatically sanitizes the dataset by identifying and skipping the notorious corrupted file `jazz.00054.wav`, preserving 999 valid audio tracks).*

---

## 6. How to Run the Project (1-Click Master Runner)

To execute the entire project end-to-end (architecture check, test evaluation, noise sweep, and multi-genre audio inference):

```bash
python run_all.py
```

### Individual Training Options

To train the baseline model:

```bash
python train.py --model baseline --epochs 5 --batch_size 32 --data_dir ./data
```

To train the deeper `ResAudioNet` model:

```bash
python train.py --model resaudionet --epochs 5 --batch_size 32 --data_dir ./data
```

### Launching TensorBoard

Monitor training and validation loss, accuracy, and learning rate schedules:

```bash
tensorboard --logdir runs
```

*(In VS Code, press `Ctrl+Shift+P` and choose `Python: Launch TensorBoard`)*.

---

## 7. Model Evaluation & Robustness Testing

Run the full evaluation suite on the held-out test set:

```bash
python evaluate.py --checkpoint checkpoints/best_baseline.pth --data_dir ./data
```

This generates:

1. **Clean Test Classification Report:** Precision, Recall, and F1-score for each of the 10 genres.
2. **Normalized Confusion Matrix Heatmap:** Saved to `reports/confusion_matrix.png`.
3. **Real-World Noise Robustness Analysis:**
   * Injects Additive White Gaussian Noise (AWGN) at SNRs of Clean, 20 dB, 10 dB, 5 dB, and 0 dB.
   * Generates accuracy degradation curves saved to `reports/noise_robustness.png`.
4. **Efficiency & Profiling:** Measures parameter count, file size, FLOPs/MACs, and Real-Time Factor (RTF).

---

## 8. Running Inference on Unseen Audio

Classify any unseen WAV file (regardless of length, sample rate, or channels):

```bash
python inference.py --audio "test_sample.wav"
```

### Example Terminal Output

```text
============================================================
  AUDIO GENRE INFERENCE RESULT
============================================================
  File: test_sample.wav (5.0s, 2 segments)
  Predicted Genre:  JAZZ
  Confidence:       38.44%
  Consistency:      100.0% of segments agreed
------------------------------------------------------------
  Top Genre Probabilities:
    1. jazz          38.44%  | #######
    2. classical     30.69%  | ######
    3. blues         16.95%  | ###
============================================================
```

---

## 9. Efficiency Metrics & Size Constraint Compliance

| Metric | Target / Constraint | Measured (`BaselineAudioCNN`) | Measured (`ResAudioNet`) | Status |
| --- | --- | --- | --- | --- |
| **Model Disk Size** | **<= 12.0 MB** | **1.50 MB** | **10.83 MB** | **PASSED [OK]** |
| **Trainable Parameters** | Low footprint | 390,890 | 2,821,866 | Optimized |
| **Computational Complexity** | Edge/Real-time | ~0.78 M FLOPs | ~5.64 M FLOPs | Low Compute |
| **Inference Latency (CPU)** | Low latency | **28.9 ms** / 3s segment | **42.5 ms** / 3s segment | Real-time |
| **Real-Time Factor (RTF)** | < 1.0 | **0.0096** (~104x real-time) | **0.0141** (~70x real-time) | **High Speed** |

---

## 10. Experimental Results & Noise Robustness (GTZAN Dataset)

### Held-Out Test Set Performance (990 Segments)

* **Classical:** **97.00% Recall** (Precision = 84.35%, F1 = 0.9023)
* **Metal:** **70.00% Recall** (Precision = 74.47%, F1 = 0.7216)
* **Pop:** **75.00% Recall** (F1 = 0.4673)
* **Blues:** **51.00% Recall** (F1 = 0.4493)
* **Overall Clean Test Accuracy:** **50.71%** (Macro F1 = 0.4927)

### Additive White Gaussian Noise (AWGN) Sweep

| Noise Level (AWGN SNR) | Classification Accuracy | Degradation vs Clean |
| --- | --- | --- |
| **Clean (inf dB)** | **50.71%** | Baseline |
| **20 dB SNR** | **51.41%** | Resilient (+0.70%) |
| **10 dB SNR** | **37.47%** | -13.24% |
| **5 dB SNR** | **31.11%** | -19.60% |
| **0 dB SNR (Extreme Noise)** | **12.42%** | Noise matches signal power |

### Key Failure Analysis

1. **Genre Ambiguity:** The primary misclassifications occur between acoustically contiguous genres:
   * `Rock` vs. `Blues` (shared pentatonic scales, acoustic/electric guitar riffs).
   * `Country` vs. `Blues` (shared chord progressions and instrumentation).
2. **Noise Masking:** Under heavy noise (5 dB and 0 dB), rhythmic genres (`disco`, `hiphop`) degrade faster because percussive drum transients are masked by noise peaks, whereas continuous harmonic genres (`classical`, `metal`) maintain higher structural resilience.

---
