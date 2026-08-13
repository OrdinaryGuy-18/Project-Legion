from extraction import extract_clip


input_audio = (
    "data/raw/controlled/"
    "PRAVEEN_MOVE_FORWARD_01.wav"
)

output_audio = (
    "data/processed/move_forward/"
    "PRAVEEN_move_forward_01.wav"
)


extract_clip(
    audio_path=input_audio,
    start_time=0.0,
    end_time=1.5,
    output_path=output_audio,
)


print("Standardized WAV created:")
print(output_audio)