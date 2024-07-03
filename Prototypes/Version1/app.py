import os
import streamlit as st
from text_to_text import text_to_text_gpt
from text_to_speech import text_to_speech_GTTS
from speech_to_text import speech_to_text
from playsound import playsound
import simpleaudio as sa
import tempfile


def main():
    st.title("ITI Avatar")

    if st.button("Start Recording"):
        question, language = speech_to_text()
        answer = text_to_text_gpt(question)
        audio_file = text_to_speech_GTTS(answer, language)
        st.audio(audio_file, format="audio/wav")


if __name__ == "__main__":
    main()
