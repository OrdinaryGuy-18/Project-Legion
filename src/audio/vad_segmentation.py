from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from silero_vad import load_silero_vad, get_speech_timestamps


TARGET_SR = 16000


def load_audio(audio_path: str):
    """Load WAV audio using SoundFile."""

    waveform, sr = sf.read(
        audio_path,
        dtype="float32",
    )

    # Stereo → mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # This pipeline expects 16 kHz audio.
    if sr != TARGET_SR:
        raise ValueError(
            f"Expected {TARGET_SR} Hz audio, "
            f"but received {sr} Hz."
        )

    waveform = torch.from_numpy(
        np.asarray(waveform, dtype=np.float32)
    )

    return waveform, sr


def detect_speech(audio_path: str):
    """Detect speech regions using Silero VAD."""

    waveform, sr = load_audio(audio_path)

    model = load_silero_vad()

    speech_timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sr,
        min_speech_duration_ms=250,
        min_silence_duration_ms=250,
    )

    return waveform, sr, speech_timestamps

def analyze_internal_silence(
    waveform,
    sr,
    segment,
    top_db=25,
):
    """
    Analyze internal silence inside one VAD region.

    Used only for diagnosing whether a VAD region
    may contain multiple rapid utterances.
    """

    start_sample = segment["start"]
    end_sample = segment["end"]

    audio = waveform[start_sample:end_sample]

    intervals = librosa.effects.split(
        audio.numpy(),
        top_db=top_db,
        frame_length=512,
        hop_length=128,
    )

    local_segments = []

    for start, end in intervals:

        local_segments.append(
            {
                "start": start / sr,
                "end": end / sr,
                "duration": (end - start) / sr,
            }
        )

    return local_segments

def main():
    audio_path = Path("data/raw/nitin_test.wav")

    waveform, sr, timestamps = detect_speech(
        str(audio_path)
    )

    print(f"Sample rate: {sr}")
    print(
        f"Duration: "
        f"{len(waveform) / sr:.2f}s"
    )

    print(
        f"Speech regions: "
        f"{len(timestamps)}"
    )

    print("\nDetected speech regions")
    print("=" * 70)

    for i, segment in enumerate(
        timestamps,
        start=1,
    ):
        start = segment["start"] / sr
        end = segment["end"] / sr
        duration = end - start

        print(
            f"{i:03d}: "
            f"{start:.2f}s → "
            f"{end:.2f}s "
            f"({duration:.2f}s)"
        )

    print("\nLong regions (> 1.20s)")
    print("=" * 70)

    for i, segment in enumerate(timestamps, start=1):
        start = segment["start"] / sr
        end = segment["end"] / sr
        duration = end - start

        if duration > 1.20:
            print(
                f"{i:03d}: "
                f"{start:.2f}s → "
                f"{end:.2f}s "
                f"({duration:.2f}s)"
           )


    print("\nShort regions (< 0.50s)")
    print("=" * 70)

    for i, segment in enumerate(timestamps, start=1):
        start = segment["start"] / sr
        end = segment["end"] / sr
        duration = end - start

        if duration < 0.50:
            print(
                f"{i:03d}: "
                f"{start:.2f}s → "
                f"{end:.2f}s "
                f"({duration:.2f}s)"
            )
    print("\nInternal analysis of suspicious regions")
    print("=" * 70)

    for index in [28, 103, 109]:

        segment = timestamps[index - 1]

        print(
            f"\nVAD region {index}: "
            f"{segment['start'] / sr:.2f}s → "
            f"{segment['end'] / sr:.2f}s"
        )

        local_segments = analyze_internal_silence(
            waveform,
            sr,
            segment,
            top_db=25,
        )

        for j, local in enumerate(
            local_segments,
            start=1,
        ):
            print(
                f"  {j}: "
                f"{local['start']:.3f}s → "
                f"{local['end']:.3f}s "
                f"({local['duration']:.3f}s)"
            )

if __name__ == "__main__":
    main()