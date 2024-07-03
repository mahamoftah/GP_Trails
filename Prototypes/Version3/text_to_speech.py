import torch
from TTS.api import TTS
from gtts import gTTS


def text_to_speech_TTS(text, lang="en"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    audio_file = "output.wav"
    tts.tts_to_file(
        text=text,
        file_path=audio_file,
        speaker="Ana Florence",
        language=lang,
        split_sentences=True
    )

    return audio_file


def text_to_speech_GTTS(text, lang):
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_file = "output.wav"
    tts.save(audio_file)
    return audio_file
