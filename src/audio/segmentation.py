from pathlib import Path

import librosa
import numpy as np
import matplotlib.pyplot as plt

TARGET_SR = 16000


def load_audio(audio_path: str):
    """
    Load an already-standardized WAV file.
    """

    waveform, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
    )

    return waveform, sr


def detect_candidate_segments(
    waveform: np.ndarray,
    sr: int,
    top_db: int = 30,
    min_duration: float = 0.25,
):
    """
    Detect candidate speech/activity segments.

    This is only a first segmentation hypothesis.
    We will inspect the results before generating WAV files.
    """

    intervals = librosa.effects.split(
        waveform,
        top_db=top_db,
        frame_length=1024,
        hop_length=256,
    )

    segments = []

    for start_sample, end_sample in intervals:
        start_time = start_sample / sr
        end_time = end_sample / sr
        duration = end_time - start_time

        if duration >= min_duration:
            segments.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "duration": duration,
                }
            )

    return segments

def analyze_segment_gaps(segments):
    """Analyze gaps between detected candidate segments."""

    gaps = []

    for i in range(len(segments) - 1):
        current = segments[i]
        next_segment = segments[i + 1]

        gap = next_segment["start"] - current["end"]
        gaps.append(gap)

    gaps = np.array(gaps)

    print("\nGap analysis")
    print("=" * 60)

    print(f"Number of gaps: {len(gaps)}")

    if len(gaps) == 0:
        return

    print(f"Minimum gap: {gaps.min():.3f}s")
    print(f"Maximum gap: {gaps.max():.3f}s")
    print(f"Average gap: {gaps.mean():.3f}s")
    print(f"Median gap: {np.median(gaps):.3f}s")

    for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        count = np.sum(gaps <= threshold)

        print(
            f"Gap <= {threshold:.2f}s: "
            f"{count}"
        )

    print("\nSmallest 20 gaps:")

    sorted_gaps = np.sort(gaps)

    for i, gap in enumerate(sorted_gaps[:20], start=1):
        print(f"{i:02d}: {gap:.3f}s")

def plot_segments(waveform, sr, segments, output_path=None):
    """Plot waveform with detected candidate boundaries."""

    time = np.arange(len(waveform)) / sr

    plt.figure(figsize=(18, 6))
    plt.plot(time, waveform)

    for index, segment in enumerate(segments, start=1):
        start = segment["start"]
        end = segment["end"]

        plt.axvline(start, linestyle="--")
        plt.axvline(end, linestyle="--")

        plt.text(
            start,
            0.9,
            str(index),
            transform=plt.gca().get_xaxis_transform(),
            rotation=90,
            fontsize=7,
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Nitin Recording — Candidate Segmentation")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)

    plt.show()

def plot_segment_window(
    waveform,
    sr,
    segments,
    start_time,
    end_time,
):
    """Plot a selected time window with candidate boundaries."""

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    time = np.arange(start_sample, end_sample) / sr
    audio = waveform[start_sample:end_sample]

    plt.figure(figsize=(18, 5))
    plt.plot(time, audio)

    for index, segment in enumerate(segments, start=1):

        if segment["end"] < start_time:
            continue

        if segment["start"] > end_time:
            continue

        plt.axvline(
            segment["start"],
            linestyle="--",
        )

        plt.axvline(
            segment["end"],
            linestyle="--",
        )

        plt.text(
            segment["start"],
            0.9,
            str(index),
            transform=plt.gca().get_xaxis_transform(),
            rotation=90,
            fontsize=8,
        )

    plt.xlim(start_time, end_time)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(
        f"Nitin Segmentation: "
        f"{start_time:.1f}s–{end_time:.1f}s"
    )

    plt.tight_layout()
    plt.show()

def main():
    audio_path = Path("data/raw/nitin_test.wav")

    waveform, sr = load_audio(str(audio_path))

    segments = detect_candidate_segments(
        waveform,
        sr,
        top_db=25,
        min_duration=0.20,
    )

    print(f"Candidate segments: {len(segments)}")

    print("\nSuspicious candidates")
    print("=" * 80)

    for i, segment in enumerate(segments):
        duration = segment["duration"]

        if duration < 0.50 or duration > 1.50:

            previous_gap = None
            next_gap = None

            if i > 0:
                previous_gap = (
                    segment["start"]
                    - segments[i - 1]["end"]
                )

            if i < len(segments) - 1:
                next_gap = (
                    segments[i + 1]["start"]
                    - segment["end"]
                )

            print(
                f"{i + 1:03d} | "
                f"{segment['start']:.2f}s → "
                f"{segment['end']:.2f}s | "
                f"duration={duration:.2f}s | "
                f"prev_gap="
                f"{previous_gap:.2f}s "
                if previous_gap is not None
                else
                f"{i + 1:03d} | "
                f"{segment['start']:.2f}s → "
                f"{segment['end']:.2f}s | "
                f"duration={duration:.2f}s | "
                f"prev_gap=None ",
                end=""
            )

            print(
                f"| next_gap="
                f"{next_gap:.2f}s"
                if next_gap is not None
                else
                "| next_gap=None"
            )


if __name__ == "__main__":
    main()