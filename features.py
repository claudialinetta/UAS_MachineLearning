
"""Preprocessing audio untuk model SVM dan CNN MusicType."""

import numpy as np
import librosa

# Konfigurasi umum
SR = 22050
SEG_SEC = 3.0

# Konfigurasi SVM
N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512

# Konfigurasi CNN
N_MELS = 96
TIME_STEPS = 128


def ringkas_fitur(data):
    data = np.asarray(data)
    return np.concatenate([
        data.mean(axis=1),
        data.std(axis=1)
    ])


def seg_features(y, sr=SR):
    """Menghasilkan 166 fitur untuk SVM."""
    y = fit_length(y, int(sr * SEG_SEC))

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    semua_fitur = [
        mfcc, delta, delta2, chroma, contrast,
        centroid, bandwidth, rolloff, zcr
    ]

    hasil = np.concatenate([
        ringkas_fitur(fitur) for fitur in semua_fitur
    ])

    return np.nan_to_num(
        hasil, nan=0.0, posinf=0.0, neginf=0.0
    )


def extract_mel(y, sr=SR):
    """Menghasilkan Mel Spectrogram untuk CNN."""
    y = fit_length(y, int(sr * SEG_SEC))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=sr // 2
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=80
    )

    mel_db = (mel_db + 80) / 80
    mel_db = mel_db[:, :TIME_STEPS]

    if mel_db.shape[1] < TIME_STEPS:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, TIME_STEPS - mel_db.shape[1]))
        )

    return mel_db.astype(np.float32)


def fit_length(y, n=None):
    """Memastikan audio memiliki panjang 3 detik."""
    if n is None:
        n = int(SR * SEG_SEC)

    y = np.asarray(y, dtype=np.float32)

    if len(y) >= n:
        return y[:n]

    return np.pad(y, (0, n - len(y)))


def load_segments(path, max_seconds=60):
    """Membaca audio dan membaginya menjadi segmen 3 detik."""
    y, _ = librosa.load(
        path, sr=SR, mono=True,
        duration=max_seconds
    )

    if len(y) == 0:
        raise ValueError("Audio kosong atau tidak dapat dibaca.")

    n = int(SR * SEG_SEC)
    segments = [y[i:i+n] for i in range(0, len(y), n)]

    if len(segments) > 1 and len(segments[-1]) < n:
        segments = segments[:-1]

    segments = [fit_length(seg, n) for seg in segments]

    return y, segments
