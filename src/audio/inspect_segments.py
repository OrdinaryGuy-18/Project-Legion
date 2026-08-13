from pathlib import Path

import soundfile as sf

from vad_segmentation import load_audio, detect_speech


COMMANDS = [
    "MOVE_FORWARD",
    "MOVE_BACK",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
]

REPETITIONS = 20


def expected_command(command_number):
    """
    Return expected command and repetition number
    for a 1-based command number.
    """

    command_index = (command_number - 1) // REPETITIONS
    repetition = ((command_number - 1) % REPETITIONS) + 1

    return COMMANDS[command_index], repetition


def main():
    audio_path = Path("data/raw/nitin_test.wav")

    waveform, sr, timestamps = detect_speech(
        str(audio_path)
    )

    print(f"Total VAD regions: {len(timestamps)}")
    print()

    blocks = [
        ("MOVE_FORWARD", 0, 30),
        ("MOVE_BACK", 30, 60),
        ("TURN_LEFT", 60, 90),
        ("TURN_RIGHT", 90, 120),
        ("STOP", 120, 154),
    ]

    print("VAD REGIONS BY RECORDING TIME")
    print("=" * 90)

    for name, start_time, end_time in blocks:

        print(
            f"\n{name}: "
            f"{start_time:.1f}s → {end_time:.1f}s"
        )

        matching = []

        for index, segment in enumerate(
            timestamps,
            start=1,
        ):
            start = segment["start"] / sr
            end = segment["end"] / sr

            if end >= start_time and start <= end_time:
                matching.append(
                    (
                        index,
                        start,
                        end,
                        end - start,
                    )
                )

        print(
            f"Regions in this time range: "
            f"{len(matching)}"
        )

        for index, start, end, duration in matching:
            print(
                f"  {index:03d} | "
                f"{start:7.2f}s → "
                f"{end:7.2f}s | "
                f"{duration:5.2f}s"
            )

        print("\nPotential block transition areas")
        print("=" * 90)

        transition_windows = [
            ("MOVE_FORWARD → MOVE_BACK", 28.0, 33.0),
            ("MOVE_BACK → TURN_LEFT", 57.0, 62.0),
            ("TURN_LEFT → TURN_RIGHT", 83.0, 88.0),
            ("TURN_RIGHT → STOP", 135.0, 140.0),
        ]

        for name, window_start, window_end in transition_windows:

            print(f"\n{name}")
            print("-" * 90)

            for index, segment in enumerate(timestamps, start=1):

                start = segment["start"] / sr
                end = segment["end"] / sr

                if end >= window_start and start <= window_end:

                    print(
                        f"{index:03d} | "
                        f"{start:7.2f}s → "
                        f"{end:7.2f}s | "
                        f"{end - start:5.2f}s"
                    )

if __name__ == "__main__":
    main()