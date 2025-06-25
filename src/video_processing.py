import os

# Enforce .cache directory for PyTorch inside the project directory
os.environ["TORCH_HOME"] = "./model_weights/torch"
os.environ["HF_HOME"] = "./model_weights/huggingface"
os.environ["PYANNOTE_CACHE"] = "./model_weights/torch/pyannote"

import warnings

import pandas as pd
import soundfile as sf
import torch
from pyannote.audio import Pipeline

from src.audio_processing import analyze_tone_intensity, extract_audio
from src.sentiment_analysis import analyze_sentiment
from src.transcription import transcribe_audio

warnings.filterwarnings("ignore")  # Set environment variable for PyTorch model cache


# Initialize the pipeline with cache_dir to ensure local caching
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    # use_auth_token="", # we don't need this since we running locally
)

# Send pipeline to GPU if available
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def process_video(
    video_path,
    output_csv_path,
    perform_diarization=True,
    perform_sentiment=True,
    perform_tone=True,
):
    """Process a single video: extract audio, diarize, transcribe, analyze sentiment and tone, save to CSV."""
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)

    diarization = pipeline(audio_path, num_speakers=6) if perform_diarization else None
    audio, sr = sf.read(audio_path)
    output_dir = "speaker_segments"
    os.makedirs(output_dir, exist_ok=True)
    table_data = []

    for turn, _, speaker in (
        diarization.itertracks(yield_label=True) if diarization else []
    ):
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        segment = audio[start_sample:end_sample]
        output_file = os.path.join(
            output_dir, f"{speaker}_{turn.start:.1f}_{turn.end:.1f}.wav"
        )
        sf.write(output_file, segment, sr)
        transcription = transcribe_audio(output_file)["text"]
        sentiment_score = (
            analyze_sentiment(transcription) if perform_sentiment else "N/A"
        )
        tone_intensity = (
            analyze_tone_intensity(output_file, 0, len(segment) / sr)
            if perform_tone
            else "N/A"
        )
        table_data.append(
            {
                "speaker_id": speaker,
                "start_time": f"{turn.start:.1f}s",
                "end_time": f"{turn.end:.1f}s",
                "transcribed_content": transcription,
                "sentiment_score": sentiment_score,
                "tone_intensity": tone_intensity,
            }
        )
        print(
            f"Saved and transcribed: {output_file} (start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker})"
        )

    df = pd.DataFrame(table_data)
    df.to_csv(output_csv_path, index=False)
    os.remove(audio_path)
    return df


def process_all_videos_from_path(
    input_dir,
    output_dir,
    perform_diarization=True,
    perform_sentiment=True,
    perform_tone=True,
):
    """Process all video files in the input directory, saving results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    for video_file in os.listdir(input_dir):
        if video_file.lower().endswith(".mp4"):
            video_path = os.path.join(input_dir, video_file)
            output_csv_path = os.path.join(
                output_dir, video_file.replace(".mp4", ".csv")
            )
            try:
                process_video(
                    video_path,
                    output_csv_path,
                    perform_diarization,
                    perform_sentiment,
                    perform_tone,
                )
                print(f"Processed: {video_file}")
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")

                print(f"Processed: {video_file}")
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
