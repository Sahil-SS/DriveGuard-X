# DriveGuard-X: Road Condition Model (Audio-Only) - Training Pipeline

This document describes how the audio-only road-condition classifier (Model 3 of DriveGuard-X) is trained, validated and packaged, and records the results of the current training run.

> **Status: pipeline validated on synthetic audio only.**
> Every number in the Results section comes from procedurally generated sounds, used to verify that the code, splits, metrics and export work end to end. They say **nothing** about performance on real roads. Real-data results will replace them once real recordings are trained (see [Roadmap](#roadmap)).

---

## 1. Goal

Classify the road surface from microphone audio alone (INMP441 on an ESP32 RC car) into four classes:

| Class | Meaning |
|---|---|
| `smooth` | Even surface, low tyre/road noise |
| `moderately_rough` | Textured or mildly uneven surface |
| `rough` | Strongly uneven / broadband loud surface |
| `impact` | Short transient events (bumps, potholes, joints) |

The output (class and confidence) feeds the DriveGuard-X fusion and battery-protection logic.

---

## 2. Pipeline overview

```
raw audio -> resample 16 kHz mono -> 50 Hz high-pass -> peak normalise
          -> 1.0 s windows (0.5 s hop) -> 47 features/window
          -> recording-level split -> (train-only augmentation)
          -> RandomForest / XGBoost / SVM -> evaluation -> road_model.pkl
```

Training lives in `DriveGuardX_Road_Audio_Classifier_v2.ipynb` (Google Colab). Feature extraction lives in a separate module, `road_features.py`, which the notebook writes and which the FastAPI backend should import, so training and serving use identical code.

### Configuration

| Parameter | Value | Notes |
|---|---|---|
| Sample rate | 16 kHz | Typical INMP441 / ESP32 I2S setting |
| Window / hop | 1.0 s / 0.5 s | Shorter windows may suit impacts |
| High-pass filter | 4th-order Butterworth, 50 Hz | Removes DC and low rumble |
| Normalisation | per-recording peak | Config flag `NORMALIZE` |
| Augmentation | 2 copies per training window | Training data only |
| Random seed | 42 | |

---

## 3. Data

### Expected layout

```
road_audio/
  smooth/            asphalt_s1_spd0.5.wav
  moderately_rough/  ...
  rough/             ...
  impact/            ...
```

Filenames carry two optional tags:

- `s<session>`: recording session or day, used for leave-one-session-out evaluation
- `spd<speed>`: speed (m/s or PWM %), used for per-speed reporting

Each file is treated as one group when splitting.

### Dataset sources (real data)

No public dataset found so far has exactly these four classes, so the plan is to map public surface/terrain audio onto the three surface classes and record the `impact` class (and RC-domain calibration data) on the RC car.

| Source | Use | Caveat |
|---|---|---|
| [Freiburg Terrains audio](https://deepterrain.cs.uni-freiburg.de/) | Wheel-mounted microphone on a small robot; closest domain to the RC car | Free for non-commercial use; per-terrain labels in the audio download are unconfirmed, so files may need manual sorting |
| A3CarScene (Data in Brief, 2023) | Asphalt vs cobblestone, dry vs wet, as a smooth-vs-rough proxy | Full-size car, multichannel audio; download link not verified |
| Own RC recordings | `impact` class, speed and motor-noise calibration | Must be collected |

### Data used in the current run (synthetic)

12 generated recordings per class, 20 s each, split across 4 sessions and 3 speeds (0.3, 0.6, 1.0), giving 1,872 windows (468 per class).

---

## 4. Features (47 per window)

Computed by `road_features.extract_features()`:

| Group | Features |
|---|---|
| Level | `rms`, `crest_factor`, `kurtosis` |
| Spectral shape | zero-crossing rate, centroid (mean/std), bandwidth, rolloff, flatness, dominant frequency, spectral entropy |
| Band energy | 8 band energy fractions (0-100, 100-250, 250-500, 0.5-1k, 1-2k, 2-3.5k, 3.5-5.5k, 5.5-8k Hz) |
| Transient cues | short-time energy (20 ms frames): log-std and max/median ratio |
| MFCC | 13 coefficients, mean and std (26 values) |

Four feature sets are compared in the ablation: `spectral`, `mfcc`, `combined`, and `combined_no_loudness` (drops `rms`, `mfcc0_mean`, `mfcc0_std`).

---

## 5. Validation design

Audio windows overlap by 50%, so a random window-level split would put near-duplicate audio in both train and test and inflate scores. The pipeline avoids this:

1. **Recording-level split**: whole recordings are held out as the test set (`StratifiedGroupKFold`, about 20% test). An assertion checks that no recording appears on both sides.
2. **Group cross-validation**: model selection uses 4-fold `StratifiedGroupKFold` on the train/validation recordings only. The best model is chosen on CV macro-F1, **not** on test scores.
3. **Leave-one-session-out (LOSO)**: each recording session is held out in turn. A large gap between LOSO and the standard split indicates the model is learning session conditions (microphone gain, background noise) rather than the road.
4. **Augmentation on training data only**: gain scaling (0.5-1.5x), white or motor-hum noise (5-25 dB SNR, 70% of copies), and +/-0.25 s time shift. Augmented copies stay in the same fold as their source window, and the test set is never augmented.

---

## 6. Models

| Model | Settings |
|---|---|
| Random Forest | 300 trees, balanced class weights |
| XGBoost | 300 trees, depth 6, learning rate 0.1, subsample 0.8, colsample 0.8 |
| SVM (RBF) | StandardScaler, C=10, balanced class weights, probability outputs |

Metrics reported: accuracy, macro precision, macro recall, macro F1, per-class report, confusion matrix.

---

## 7. Results (synthetic data, pipeline validation only)

### 7.1 R1: Classifier comparison

| Model | CV macro-F1 (mean +/- std) | Test accuracy | Test macro-F1 |
|---|---|---|---|
| **Random Forest** | **0.983 +/- 0.017** | 1.000 | 1.000 |
| XGBoost | 0.983 +/- 0.018 | 1.000 | 1.000 |
| SVM (RBF) | 0.953 +/- 0.049 | 1.000 | 1.000 |

Random Forest was selected on CV score (nearly tied with XGBoost). All three score perfectly on test, which is expected on synthetic classes that differ strongly by construction. The test set (390 windows) has per-class support: impact 117, moderately_rough 78, rough 117, smooth 78.

### 7.2 Leave-one-session-out

| Features | Mean F1 | Std | Per-session F1 (sessions 0-3) |
|---|---|---|---|
| combined | 0.996 | 0.009 | 1.000, 1.000, 0.983, 1.000 |
| combined_no_loudness | 0.707 | 0.013 | 0.708, 0.724, 0.693, 0.702 |

### 7.3 R2: Feature ablation (test macro-F1)

| Model | combined | combined_no_loudness | mfcc | spectral |
|---|---|---|---|---|
| Random Forest | 1.000 | 0.736 | 1.000 | 1.000 |
| SVM (RBF) | 1.000 | 0.826 | 1.000 | 1.000 |
| XGBoost | 1.000 | 0.754 | 1.000 | 1.000 |

### 7.4 Loudness-shortcut check

Removing `rms` and `mfcc0` dropped Random Forest test F1 from **1.000 to 0.736** (drop 0.264), and the notebook flagged a loudness dependence. The feature importances agree: `rms` and `mfcc0_mean` are the two most important features. On this synthetic data that is by design, because classes differ mainly in level. The check is included because on real data, loudness also depends on microphone gain, mounting distance, vehicle speed and session, and a model that leans on it may fail when conditions change.

### 7.5 Per-speed performance (test)

| Speed | Windows | Accuracy | Macro-F1 |
|---|---|---|---|
| 0.3 | 156 | 1.000 | 1.000 |
| 0.6 | 39 | 1.000 | 1.000 |
| 1.0 | 195 | 1.000 | 1.000 |

Mean prediction confidence on correct predictions: 0.987 (no errors on test).

### 7.6 R3: Noise robustness (test windows, features re-extracted after mixing)

| SNR | White noise F1 | Motor-hum F1 |
|---|---|---|
| clean | 1.000 | 1.000 |
| 20 dB | 1.000 | 1.000 |
| 10 dB | 1.000 | 1.000 |
| 5 dB | 1.000 | 1.000 |
| 0 dB | 0.603 | 0.927 |

Performance holds down to 5 dB SNR and degrades at 0 dB, more sharply for white noise. Training used noise augmentation over 5-25 dB, so 0 dB lies outside the augmented range. The motor-hum noise is a crude synthetic harmonic model, not real motor recordings.

---

## 8. Interpretation

- **Pipeline works.** Loading, windowing, feature extraction, group splitting, augmentation, three classifiers, ablation, LOSO, noise tests and model export all run end to end.
- **Perfect scores are not evidence of a good model.** The synthetic classes are easy to separate. Real recordings will have overlapping classes and lower scores.
- **The diagnostics do their job.** The loudness check and LOSO exposed a shortcut that a plain accuracy table would hide. These same checks are the main tool for judging real-data results.

---

## 9. Limitations

- All reported results are synthetic and should not be cited as model performance.
- The motor-noise test uses a synthetic hum; real motor-only recordings are needed.
- Per-recording peak normalisation cannot be reproduced on a live stream (the recording peak is unknown). For deployment, either normalise with a fixed calibration peak (`live_normalize()`) or retrain with `NORMALIZE = "none"`.
- The time-shift augmentation (+/-0.25 s) is untested on real impact events, where a shift could move a short bump out of the window.
- Class definitions (what counts as moderately rough vs rough) still need to be fixed against real recordings.
- Not yet explored: window-size sweep, a separate short-window impact detector, probability calibration, and a CNN on log-mel spectrograms.

---

## 10. Outputs

| File | Content |
|---|---|
| `road_model.pkl` | Bundle: fitted model, label encoder, feature names, config |
| `road_features.py` | Shared preprocessing and feature extraction |
| `R1_classifier_comparison.csv` | Classifier comparison |
| `R2_feature_ablation.csv` | Feature-set ablation |
| `LOSO_combined.csv`, `LOSO_combined_no_loudness.csv` | Leave-one-session-out results |
| `per_speed.csv` | Per-speed test results |
| `R3_noise_robustness.csv` | Noise robustness |
| `confusion_matrix.png` | Held-out confusion matrix |

The current `road_model.pkl` was trained on synthetic data (`trained_on_synthetic: true` in its config) and must not be used in the live system.

---

## 11. Reproducing

1. Open `DriveGuardX_Road_Audio_Classifier_v2.ipynb` in Google Colab.
2. Run all cells (default `USE_SYNTHETIC = True`) to verify the pipeline.
3. For real data, upload audio in the layout from [Section 3](#3-data), set `USE_SYNTHETIC = False` and `DATA_ROOT`, and re-run.
4. Check the loudness-shortcut warning and the LOSO results; set `DROP_LOUDNESS_FOR_FINAL = True` if the model depends heavily on loudness.
5. Optionally set `RETRAIN_ON_ALL = True` for the final export.

---

## 12. Using the model in the backend

```python
import joblib
import road_features as rf

bundle = joblib.load("road_model.pkl")

# ESP32 sends 32-bit I2S words with 24-bit INMP441 samples
samples = rf.decode_inmp441(raw_int32_array)

# window: float32, 16 kHz, 1.0 s, preprocessed the same way as in training
label, probabilities = rf.predict_window(bundle, window)
```

`probabilities` maps each class to its predicted probability, for use as the confidence signal in the fusion logic.

---

## Roadmap

- [ ] Collect or assemble real labelled audio (public terrain data plus RC recordings, including the `impact` class)
- [ ] Record motor-only audio for realistic noise tests
- [ ] Re-run with real data and replace Section 7
- [ ] Decide final class definitions and whether to drop loudness features
- [ ] Window-size sweep and short-window impact detector
- [ ] Probability calibration for fusion
- [ ] Optional CNN baseline on log-mel spectrograms