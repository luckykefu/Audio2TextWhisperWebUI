import torch
import whisper
from whisper.utils import get_writer
import os
from src.log import get_logger

logger = get_logger(__name__)


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


if __name__ == "__main__":
    audio_path = r"d:\Music\阿七御姐充电提示音_爱给网_aigei_com.mp3"

    audio2text(
        audio_path,
        model_path=r"whisper_models\small.pt",
        prompt=None,
        output_format="all",
        output_dir="temp",
    )