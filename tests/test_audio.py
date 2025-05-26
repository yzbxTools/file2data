"""
use steamlit to record audio, then process it, convert it to text, finally use tts to play the text
"""

import streamlit as st
import sounddevice as sd
import wavio
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os

def record_audio(duration=5, sample_rate=44100, channels=1):
    """录制音频"""
    st.write(f"开始录音，持续 {duration} 秒...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()

    st.write("录音完成！")
    return recording, sample_rate

def save_audio(recording, sample_rate, filename="temp_recording.wav"):
    """保存音频文件"""
    # wav.write(filename, sample_rate, recording)
    wavio.write(filename, recording, sample_rate, sampwidth=2)
    return filename

def speech_to_text(audio_file):
    """将语音转换为文字"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language='zh-CN')
            return text
        except sr.UnknownValueError:
            return "无法识别音频"
        except sr.RequestError:
            return "无法连接到语音识别服务"

def text_to_speech(text, lang='zh-cn'):
    """将文字转换为语音"""
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

def main():
    st.title("语音处理应用")
    st.write("这个应用可以录制音频，将其转换为文字，然后再将文字转换回语音。")

    # 录音部分
    if st.button("开始录音"):
        recording, sample_rate = record_audio()
        audio_file = save_audio(recording, sample_rate)

        # 显示录音
        st.audio(audio_file)

        # 语音转文字
        text = speech_to_text(audio_file)
        st.write("识别结果：", text)

        # 文字转语音
        if text and text != "无法识别音频" and text != "无法连接到语音识别服务":
            tts_file = text_to_speech(text)
            st.audio(tts_file)

            # 清理临时文件
            os.unlink(audio_file)
            os.unlink(tts_file)

if __name__ == "__main__":
    main()