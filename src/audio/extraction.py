from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


TARGET_SR = 16000
TARGET_DURATION = 1.0


def load_audio(audio_path):
    waveform, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
    )

    return waveform


def standardize_clip(
    waveform,
    target_sr=TARGET_SR,
    target_duration=TARGET_DURATION,
):
    target_samples = int(
        target_sr * target_duration
    )

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    # Remove leading/trailing silence.
    intervals = librosa.effects.split(
        waveform,
        top_db=30,
    )

    if len(intervals) > 0:
        start = intervals[0][0]
        end = intervals[-1][1]
        waveform = waveform[start:end]

    # Normalize safely.
    peak = np.max(np.abs(waveform))

    if peak > 0:
        waveform = waveform / peak

    # Trim if too long.
    if len(waveform) > target_samples:
        waveform = waveform[:target_samples]

    # Zero-pad if too short.
    elif len(waveform) < target_samples:
        padding = target_samples - len(waveform)

        waveform = np.pad(
            waveform,
            (0, padding),
            mode="constant",
        )

    return waveform.astype(np.float32)


def extract_clip(
    audio_path,
    start_time,
    end_time,
    output_path,
):
    waveform = load_audio(audio_path)

    start_sample = int(
        start_time * TARGET_SR
    )

    end_sample = int(
        end_time * TARGET_SR
    )

    clip = waveform[
        start_sample:end_sample
    ]

    clip = standardize_clip(clip)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        output_path,
        clip,
        TARGET_SR,
        subtype="PCM_16",
    )

    return output_path