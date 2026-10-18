# backend/app/preprocessing/road_features.py
import numpy as np
import pandas as pd
import scipy.signal as sps
import scipy.stats as st
import librosa

SR = 16000
N_FFT, HOP = 512, 256
BAND_EDGES = np.array([0, 100, 250, 500, 1000, 2000, 3500, 5500, 8000])
_HP = sps.butter(4, 50, btype="highpass", fs=SR, output="sos")

def decode_inmp441(raw_i32):
    """INMP441: 24-bit sample, left-justified in a 32-bit I2S word -> float in [-1, 1]"""
    return (np.asarray(raw_i32, dtype=np.int32) >> 8).astype(np.float32) / 8388608.0

def preprocess(y, sr=SR, normalize="none"):
    y = np.asarray(y, dtype=np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    
    # Auto-pad short test arrays to at least 1 second (16000 samples) to prevent empty STFT frames
    if len(y) < SR:
        y = np.pad(y, (0, SR - len(y)))
        
    y = sps.sosfilt(_HP, y - np.mean(y)).astype(np.float32)
    if normalize == "peak":
        y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)

def extract_features(win):
    f = {}
    S = np.abs(librosa.stft(win, n_fft=N_FFT, hop_length=HOP)) ** 2
    mag = np.sqrt(S)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    f["rms"] = float(np.sqrt(np.mean(win ** 2)))
    f["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(win, frame_length=N_FFT, hop_length=HOP)))
    cen = librosa.feature.spectral_centroid(S=mag, sr=SR)[0]
    f["centroid_mean"], f["centroid_std"] = float(cen.mean()), float(cen.std())
    f["bandwidth"] = float(librosa.feature.spectral_bandwidth(S=mag, sr=SR).mean())
    f["rolloff"] = float(librosa.feature.spectral_rolloff(S=mag, sr=SR).mean())
    f["flatness"] = float(librosa.feature.spectral_flatness(S=mag).mean())
    ps = S.mean(axis=1) + 1e-12
    f["dominant_freq"] = float(freqs[np.argmax(ps)])
    p = ps / ps.sum()
    f["spectral_entropy"] = float(-(p * np.log2(p)).sum() / np.log2(len(p)))
    for i in range(len(BAND_EDGES) - 1):
        m = (freqs >= BAND_EDGES[i]) & (freqs < BAND_EDGES[i + 1])
        f["band%d_frac" % i] = float(ps[m].sum() / ps.sum())
    fl = int(0.02 * SR)
    n = len(win) // fl
    e = (win[:n * fl].reshape(n, fl) ** 2).mean(axis=1) + 1e-12
    f["ste_std_log"] = float(np.std(np.log10(e)))
    f["ste_max_over_median"] = float(np.log10(e.max() / np.median(e)))
    f["crest_factor"] = float(np.max(np.abs(win)) / (f["rms"] + 1e-9))
    f["kurtosis"] = float(st.kurtosis(win))
    mf = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=SR, n_mels=40)),
                              n_mfcc=13)
    for i in range(13):
        f["mfcc%d_mean" % i] = float(mf[i].mean())
        f["mfcc%d_std" % i] = float(mf[i].std())
    # Silent or nearly silent windows can produce NaN/inf in kurtosis and
    # logarithmic spectral statistics. Models must never receive non-finite
    # values, so replace those edge-case outputs with a neutral zero.
    for name, value in f.items():
        if not np.isfinite(value):
            f[name] = 0.0
    return f

def predict_window(bundle, win):
    feats = extract_features(win)
    x = pd.DataFrame([feats])[bundle["feature_names"]]
    proba = bundle["model"].predict_proba(x)[0]
    names = list(bundle["label_encoder"].classes_)
    k = int(np.argmax(proba))
    return names[k], {n: float(q) for n, q in zip(names, proba)}
