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
# from src.audio2text import audio2text

# if __name__ == "__main__":
#     audio_path = r"d:\Music\阿七御姐充电提示音_爱给网_aigei_com.mp3"

#     audio2text(
#         audio_path,
#         model_path=r"whisper_models\small.pt",
#         prompt=None,
#         output_format="all",
#         output_dir="temp",
#     )

#####################################################
whisper_models_dir = os.path.join(os.getcwd(), "whisper_models")
output_dir = os.path.join(os.getcwd(), "output")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(whisper_models_dir, exist_ok=True)

whisper_models = os.listdir(whisper_models_dir)
if whisper_models:
    whisper_models = [m for m in whisper_models if m.endswith(".pt")]
    print(whisper_models)
else:
    whisper_models = ["Nomodel"]
