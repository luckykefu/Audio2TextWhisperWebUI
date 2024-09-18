import os
import gradio as gr
from .audio2text import audio2text

whisper_models_dir = os.path.join(os.getcwd(), "whisper_models")
output_dir = os.path.join(os.getcwd(), "output")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(whisper_models_dir, exist_ok=True)

whisper_models = os.listdir(whisper_models_dir)
if whisper_models:
    whisper_models = [m for m in whisper_models if m.endswith(".pt")]
    print(whisper_models)
else:
    whisper_models = [f"No model in {whisper_models_dir}"]


def demo_whisper():
    with gr.Row():
        audio_file_path5 = gr.Audio(label="Upload Audio File", type="filepath")
        model_path = gr.Dropdown(
            choices=whisper_models,
            label="Choose Model",
            value=whisper_models[0],
        )
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="Chinese", lines=2)
            output_format = gr.Dropdown(
                choices=["txt", "vtt", "srt", "tsv", "json", "all"],
                label="Output Format",
                value="all",
            )
            output_dir1 = gr.Textbox(label="Output Folder", value=output_dir)
    with gr.Row():
        audio_recognition_btn = gr.Button("Recognize")

    gr.Markdown("### Files are in the output folder")
    with gr.Row():

        audio_recognition_output = gr.Textbox(
            label="Recognition Result", lines=5, value=""
        )

    audio_recognition_btn.click(
        fn=audio2text,
        inputs=[
            audio_file_path5,
            model_path,
            prompt,
            output_format,
            output_dir1,
        ],
        outputs=[audio_recognition_output],
    )
