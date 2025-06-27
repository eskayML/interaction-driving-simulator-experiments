# ===================================== ENFORCE CACHE DIRECTORY======================================
import os
os.environ["TORCH_HOME"] = "./model_weights/torch"
os.environ["HF_HOME"] = "./model_weights/huggingface"
os.environ["PYANNOTE_CACHE"] = "./model_weights/torch/pyannote"
# ====================================================================================================

import warnings
import whisper
import torch

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper, returning segments with timestamps.
    Args:
        audio_path (str): Path to the audio file.
    """
    device = "cuda" if  torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base.en", device=device, download_root="./model_weights/whisper")
    result = model.transcribe(audio_path, word_timestamps=True)
    return result
