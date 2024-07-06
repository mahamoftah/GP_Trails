# https://pypi.org/project/streamlit-TTS/
from streamlit_TTS import text_to_speech


def text_to_speech_streamlit(text, lang):
    text_to_speech(text=text, language=lang)
