import streamlit as st
from text_to_text import text_to_text_ollam_api
from text_to_text import text_to_text_ollam_langchain
from text_to_speech import text_to_speech_GTTS
from speech_to_text import speech_to_text_groq
from preprocessing import preprocessing


def main():
    st.title("ITI Avatar")

    question = speech_to_text_groq()
    # text, language = speech_to_text()
    while question is not None:
        st.write(question)
        # st.write(language)
        model = "gemma2"
    #

        quest = [{
            "role": "user",
            "content": question,
        }]

        language = "ar"

        if model == "llama3" and language == "ar":
            quest = [{
                "role": "system",
                "content": "Answer in Arabic",
            }, {
                "role": "user",
                "content": question,
            }]

        answer = text_to_text_ollam_langchain(quest, model)
        st.write(answer)
        answer = preprocessing(answer)

        audio_file = text_to_speech_GTTS(answer, language)
        st.audio(audio_file, format="audio/wav")

        break


if __name__ == "__main__":
    main()
