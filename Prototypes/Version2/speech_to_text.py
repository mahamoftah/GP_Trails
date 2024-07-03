import sounddevice as sd
import whisper

AUDIO_FILE_PATH = "output.wav"


def record_audio(duration=7, fs=44100):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete...")
    return audio, fs


def save_audio(audio, fs, filename):
    import wave
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())


def speech_to_text():
    audio, fs = record_audio()
    save_audio(audio, fs, AUDIO_FILE_PATH)

    model = whisper.load_model("base")
    result = model.transcribe(AUDIO_FILE_PATH)
    text = result["text"]
    language = result["language"]
    return text, language
