# https://pypi.org/project/streamlit-mic-recorder/
# https://github.com/stefanrmmr/streamlit-audio-recorder
# https://console.groq.com/docs/speech-text
# https://www.koyeb.com/tutorials/using-groq-to-build-a-real-time-language-translation-app

import whisper
import streamlit as st
from groq import Groq
from audio_recorder_streamlit import audio_recorder
from streamlit_mic_recorder import mic_recorder
from st_audiorec import st_audiorec
import dotenv
import io

dotenv.load_dotenv()
path = "D:/2. GP/1. Pipeline/Version4/output.wav"


def speech_to_text_whisper():

    model = whisper.load_model("base")
    result = model.transcribe(path)
    text = result["text"]
    language = result["language"]
    return text, language


def recorder1():
    audio_bytes = audio_recorder()

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        with open(path, mode='wb') as f:
            f.write(audio_bytes)
        st.success("Audio recorded successfully!")
    else:
        st.warning("Please record some audio.")


def recorder2():

    audio = mic_recorder(
        start_prompt="⏺️",
        stop_prompt="⏹️",
        key='recorder'
    )

    if audio is not None:
        audio_bio = io.BytesIO(audio['bytes'])
        audio_bio.name = 'output.wav'
        return audio_bio


def recorder3():
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        with open(path, mode='wb') as f:
            f.write(wav_audio_data)


def speech_to_text_groq():

    audio = recorder2()

    try:

        client = Groq(api_key="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")
        transcription = client.audio.transcriptions.create(
            file=audio,
            model="whisper-large-v3",
            prompt="Specify context or spelling",
            response_format="json",
            language="ar",
            temperature=0.0
        )

        return transcription.text

    except Exception as e:
        str(e)