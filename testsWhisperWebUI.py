# testsWhisperWebUI.py
# --coding:utf-8--
# Time:2024-09-17 21:15:05
# Author:Luckykefu
# Email:3124568493@qq.com
# Description:

# 获取脚本所在目录的绝对路径
import os


script_dir = os.path.dirname(os.path.abspath(__file__))

# 更改当前工作目录
os.chdir(script_dir)

########################################
# TODO：audio to text
from src.audio2text import audio2text

if __name__ == "__main__":
    audio_path = r"d:\Music\阿七御姐充电提示音_爱给网_aigei_com.mp3"

    audio2text(
        audio_path,
        model_path=r"whisper_models\small.pt",
        prompt=None,
        output_format="all",
        output_dir="temp",
    )

#####################################################
