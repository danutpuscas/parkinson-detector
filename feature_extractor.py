import parselmouth
import numpy as np
from scipy.stats import entropy

def safe_praat_call(obj, *args):
    try:
        return parselmouth.praat.call(obj, *args)
    except Exception:
        return np.nan

def compute_ppe(pitch_values, bins=30):
    pitch_values = pitch_values[pitch_values > 0]
    if len(pitch_values) == 0:
        return np.nan
    pitch_norm = (pitch_values - np.mean(pitch_values)) / np.std(pitch_values)
    hist, _ = np.histogram(pitch_norm, bins=bins, density=True)
    return entropy(hist + 1e-6)

def compute_rpde(signal, embedding_dim=10, delay=2, radius=0.2):
    N = len(signal)
    M = N - (embedding_dim - 1) * delay
    if M <= 0:
        return np.nan
    embedded = np.empty((M, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = signal[i * delay:i * delay + M]
    D = np.linalg.norm(embedded[:, None] - embedded[None, :], axis=-1)
    R = D < radius
    L = []
    for k in range(1, M):
        diag = np.diag(R, k=k)
        count = 0
        for val in diag:
            if val:
                count += 1
            else:
                if count >= 2:
                    L.append(count)
                count = 0
        if count >= 2:
            L.append(count)
    if not L:
        return np.nan
    hist, _ = np.histogram(L, bins=30, density=True)
    return entropy(hist + 1e-6)

def compute_dfa(signal):
    try:
        y = np.cumsum(signal - np.mean(signal))
        scales = np.floor(np.logspace(2, np.log10(len(y) / 4), num=10)).astype(int)
        flucts = []
        for scale in scales:
            if scale >= len(y):
                continue
            shape = (len(y) - scale + 1, scale)
            if shape[0] <= 1:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(y, scale)
            rms = np.sqrt(np.mean((windows - np.mean(windows, axis=1, keepdims=True))**2, axis=1))
            flucts.append(np.mean(rms))
        if len(flucts) < 2:
            return np.nan
        coeffs = np.polyfit(np.log(scales[:len(flucts)]), np.log(flucts), 1)
        return coeffs[0]
    except:
        return np.nan

def extract_features(file_path, age, sex):
    snd = parselmouth.Sound(file_path).convert_to_mono()
    signal = snd.values[0][:16000]
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pp = safe_praat_call(snd, "To PointProcess (periodic, cc)", 75, 500)
    harmonicity = safe_praat_call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    features = {
        'Jitter(%)': safe_praat_call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
        'Jitter:RAP': safe_praat_call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
        'Jitter:PPQ5': safe_praat_call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
        'Jitter:DDP': safe_praat_call(pp, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
        'Shimmer': safe_praat_call((snd, pp), "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        'Shimmer:APQ3': safe_praat_call((snd, pp), "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        'Shimmer:APQ5': safe_praat_call((snd, pp), "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        'Shimmer:APQ11': safe_praat_call((snd, pp), "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        'Shimmer:DDA': safe_praat_call((snd, pp), "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        'HNR': safe_praat_call(harmonicity, "Get mean", 0, 0),
        'RPDE': compute_rpde(signal),
        'DFA': compute_dfa(signal),
        'PPE': compute_ppe(pitch_values)
    }
    return features, snd, pitch
