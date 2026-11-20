"""Ekstraksi fitur MFCC yang dipakai saat training dan prediksi aplikasi."""
import numpy as np
import librosa

SR = 22050
SEG_SEC = 3.0
N_MFCC = 20


def seg_features(y, sr=SR):
    """Mengubah satu segmen audio menjadi 40 fitur: 20 MFCC mean + 20 MFCC std."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        y = np.zeros(int(sr * SEG_SEC), dtype=float)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1)))


def fit_length(y, n=None):
    """Memastikan panjang audio tepat 3 detik."""
    if n is None:
        n = int(SR * SEG_SEC)

    y = np.asarray(y)
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def file_features(path):
    """Mengekstraksi fitur dari satu file audio training IRMAS."""
    y, _ = librosa.load(path, sr=SR, mono=True, duration=SEG_SEC)
    return seg_features(fit_length(y))


def load_segments(path, max_seconds=60):
    """Memuat audio dan membaginya menjadi segmen 3 detik tanpa overlap."""
    y, _ = librosa.load(path, sr=SR, mono=True, duration=max_seconds)
    n = int(SR * SEG_SEC)

    if len(y) == 0:
        raise ValueError("File audio kosong atau tidak dapat dibaca.")

    segs = [y[i:i + n] for i in range(0, len(y), n)]
    # Buang sisa segmen yang lebih pendek dari 3 detik (padding nol merusak MFCC),
    # kecuali jika audio memang hanya satu segmen pendek.
    if len(segs) > 1 and len(segs[-1]) < n:
        segs = segs[:-1]
    segs = [fit_length(seg, n) for seg in segs]
    return y, segs
    n = int(SR * SEG_SEC)

    if len(y) == 0:
        raise ValueError("File audio kosong atau tidak dapat dibaca.")

    segs = [y[i:i + n] for i in range(0, len(y), n)]
    segs = [fit_length(seg, n) for seg in segs]
    return y, segs
