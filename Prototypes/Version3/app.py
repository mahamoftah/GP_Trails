import streamlit as st
from text_to_text import text_to_text_ollam_api
from text_to_speech import text_to_speech_GTTS
from speech_to_text import speech_to_text
from preprocessing import preprocessing


def main():
    st.title("ITI Avatar")
    model = "gemma2"

    if st.button("Start Recording"):
        question, language = speech_to_text()
        st.write(question)

        quest = [{
            "role": "user",
            "content": question,
        }]

        if model == "llama3" and language == "ar":
            quest = [{
                "role": "system",
                "content": "Answer in Arabic",
            }, {
                "role": "user",
                "content": question,
            }]

        answer = text_to_text_ollam_api(quest, model)
        st.write(answer)
        answer = preprocessing(answer)

        audio_file = text_to_speech_GTTS(answer, language)
        st.audio(audio_file, format="audio/wav")


if __name__ == "__main__":
    main()
