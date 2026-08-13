import librosa
import numpy as np
import torch


TARGET_SR = 16000
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160


def audio_to_logmel(audio_path):
    waveform, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
    )

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    tensor = torch.tensor(
        log_mel,
        dtype=torch.float32,
    )

    return tensor