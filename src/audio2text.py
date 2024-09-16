


def audio2text(
    audio_path,
    model_path="whisper_models/small.pt",
    prompt=None,
    output_format="all",
    output_dir="temp",
):
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
    return result["text"]