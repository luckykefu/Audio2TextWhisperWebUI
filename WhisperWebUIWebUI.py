# WhisperWebUIWebUI.py
# --coding:utf-8--
# Time:2024-09-17 21:15:05
# Author:Luckykefu
# Email:3124568493@qq.com
# Description:

import gradio as gr
import argparse
from src.log import get_logger
import os
from src.audio2text import audio2text

logger = get_logger(__name__)

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 更改当前工作目录
os.chdir(script_dir)

# 创建临时文件夹
output_dir = os.path.join(script_dir, "output")
whisper_models_dir = os.path.join(script_dir, "whisper_models")
if os.path.exists(whisper_models_dir):
    whisper_models = os.listdir(whisper_models_dir)
    whisper_models_path = [
        os.path.join(whisper_models_dir, model) for model in whisper_models
    ]


def main():

    # Define the interface
    with gr.Blocks() as demo:
        with gr.TabItem("Audio"):
            with gr.TabItem("WhisperWebUI"):
                with gr.Row():
                    audio_file_path5 = gr.Audio(label="上传音频文件", type="filepath")
                    model_path = gr.Dropdown(
                        choices=whisper_models_path,
                        label="模型路径",
                        value=whisper_models_path[0],
                    )
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", value="中文", lines=2)
                        output_format = gr.Dropdown(
                            choices=["txt", "vtt", "srt", "tsv", "json", "all"],
                            label="输出格式",
                            value="all",
                        )
                        output_dir1 = gr.Textbox(label="输出文件夹", value=output_dir)
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

    # Launch the interface
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


if __name__ == "__main__":
    main()
