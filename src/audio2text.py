import torch
import whisper
from whisper.utils import get_writer
import os
from src.log import logger


def audio2text(
    audio_path,
    model_path="whisper_models/small.pt",
    prompt=None,
    output_format="all",
    output_dir="temp",
):
    logger.info(f"Transcribing {audio_path}")
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model(model_path).to(device)
    # Transcribe audio
    if prompt:
        result = model.transcribe(audio_path, initial_prompt=prompt)
    else:
        result = model.transcribe(audio_path)

    os.makedirs(output_dir, exist_ok=True)
    writer = get_writer(output_format, output_dir)
    writer(result, audio_path)
    logger.info(f"Transcription saved to {output_dir}")
    return result["text"]
