# https://pypi.org/project/streamlit-mic-recorder/
from streamlit_mic_recorder import speech_to_text


def speech_to_text_streamlit(lang):

    text = speech_to_text(
        language=lang,
        start_prompt='🔇',
        stop_prompt="🔈",
        just_once=False,
        use_container_width=False,
        callback=None,
        args=(),
        kwargs={},
        key=None
    )

    return text

