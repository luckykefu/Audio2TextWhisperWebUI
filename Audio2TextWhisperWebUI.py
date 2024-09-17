import argparse
import os
from src.audio2text import audio2text
from src.log import logger
import gradio as gr

# 获取当前文件的目录

dir_path = os.path.dirname(os.path.realpath(__file__))
temp_dir = os.path.join(dir_path, "temp")

with gr.Blocks() as demo:
    with gr.TabItem("Audio"):
        with gr.TabItem("Audio2Text Whisper"):
            with gr.Row():
                audio_file_path5 = gr.Audio(label="上传音频文件", type="filepath")
                model_path = gr.Textbox(
                    label="模型路径", value="whisper_models/small.pt"
                )
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", value="中文", lines=2)
                    output_format = gr.Dropdown(
                        choices=["txt", "vtt", "srt", "tsv", "json", "all"],
                        label="输出格式",
                        value="all",
                    )
                    output_dir1 = gr.Textbox(label="输出文件夹", value=temp_dir)
            with gr.Row():
                audio_recognition_btn = gr.Button("识别")

            gr.Markdown("### 文件在output文件夹下")
            with gr.Row():

                audio_recognition_output = gr.Textbox(
                    label="识别结果", lines=5, value=""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "--server_name", type=str, default="localhost", help="server name"
    )
    parser.add_argument("--server_port", type=int, default=8080, help="server port")
    parser.add_argument("--root_path", type=str, default=None, help="root path")
    args = parser.parse_args()

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        root_path=args.root_path,
        show_api=False,
    )
