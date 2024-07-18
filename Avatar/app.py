import time
import re
import io
import os, sys, contextlib
import pandas as pd
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from scipy.io.wavfile import read, write
import audio2face_pb2
import audio2face_pb2_grpc
import grpc
from audio2face_streaming_utils import push_audio_track
from typing import Union, Type
import google.generativeai as genai


asr = sr.Recognizer()
a2f_url = 'localhost:50051'
sample_rate = 22050
a2f_avatar_instance = '/World/audio2face/PlayerStreaming'
GOOGLE_API_KEY = "AIzaSyBaz13UTSLEsag18c_rHQ9yFUbX4sx3YYM"


@contextlib.contextmanager
def ignoreStderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


def speech_to_text(audio: sr.AudioData, lang: str) -> tuple[bool, Union[str, Type[Exception]]]:
    """
    Convert speech audio to text using Google Web Speech API.
    
    Parameters:
        audio (sr.AudioData): Speech audio data.
        
    Returns:
        Tuple[bool, Union[str, Type[Exception]]]: A tuple containing a boolean indicating if the recognition
                                                 was successful (True) or not (False), and the recognized text
                                                 or the class of the exception if an error occurred.
    """
    global asr
    try:
        # Use Google Web Speech API to recognize speech from audio data
        return True, asr.recognize_google(audio, language=lang)   # "en-US"
    except Exception as e:
        # If an error occurs during speech recognition, return False and the type of the exception
        return False, e.__class__


def get_tts_data(text: str, lang: str) -> bytes:
    """
    Generate Text-to-Speech (TTS) audio in mp3 format.
    
    Parameters:
        text (str): The text to be converted to speech.
        
    Returns:
        bytes: TTS audio in mp3 format.
    """
    # Create a BytesIO object to hold the TTS audio data in mp3 format
    tts_result = io.BytesIO()
    # Generate TTS audio using gTTS library with the specified text and language (en-US)
    tts = gTTS(text=text, lang=lang, slow=False)
    # Write the TTS audio data to the BytesIO object
    tts.write_to_fp(tts_result)
    tts_result.seek(0)
    # Read and return the TTS audio data as bytes
    return tts_result.read()

def tts_to_wav(tts_byte: bytes, framerate: int = 22050) -> np.ndarray:
    """
    Convert TTS audio from mp3 format to WAV format and set the desired frame rate and channels.
    
    Parameters:
        tts_byte (bytes): TTS audio in mp3 format.
        framerate (int, optional): Desired frame rate for the WAV audio. Defaults to 22050.
        
    Returns:
        numpy.ndarray: TTS audio in WAV format as a numpy array of float32 values.
    """
    # Convert the TTS audio bytes in mp3 format to a pydub AudioSegment object
    seg = AudioSegment.from_mp3(io.BytesIO(tts_byte))
    # Set the frame rate and number of channels for the audio
    seg = seg.set_frame_rate(framerate)
    seg = seg.set_channels(1)
    # Create a BytesIO object to hold the WAV audio data
    wavIO = io.BytesIO()
    # Export the AudioSegment as WAV audio to the BytesIO object
    seg.export(wavIO, format="wav")
    # Read the WAV audio data from the BytesIO object using scipy.io.wavfile.read()
    rate, wav = read(io.BytesIO(wavIO.getvalue()))
    return wav

def wav_to_numpy_float32(wav_byte: bytes) -> np.ndarray:
    """
    Convert WAV audio from bytes to a numpy array of float32 values.
    
    Parameters:
        wav_byte (bytes): WAV audio data.
        
    Returns:
        numpy.ndarray: WAV audio as a numpy array of float32 values.
    """
    # Convert the WAV audio bytes to a numpy array of float32 values
    return wav_byte.astype(np.float32, order='C') / 32768.0

def get_tts_numpy_audio(text: str, lang: str) -> np.ndarray:
    """
    Generate Text-to-Speech (TTS) audio in WAV format and convert it to a numpy array of float32 values.
    
    Parameters:
        text (str): The text to be converted to speech.
        
    Returns:
        numpy.ndarray: TTS audio as a numpy array of float32 values.
    """
    # Generate TTS audio in mp3 format from the given text
    mp3_byte = get_tts_data(text, lang)
    # Convert the TTS audio in mp3 format to WAV format and a numpy array of float32 values
    wav_byte = tts_to_wav(mp3_byte)
    return wav_to_numpy_float32(wav_byte)


def make_avatar_speaks(text: str, lang: str) -> None:
    """
    Make the avatar speak the given text by pushing the audio track to the NVIDIA A2F instance.
    
    Parameters:
        text (str): The text to be spoken by the avatar.
        
    Returns:
        None
    """
    global a2f_url
    global sample_rate
    global a2f_avatar_instance
    # Get the TTS audio in WAV format as a numpy array of float32 values
    tts_audio = get_tts_numpy_audio(text, lang)
    # Push the TTS audio to the NVIDIA A2F instance for the avatar to speak
    push_audio_track(a2f_url, tts_audio, sample_rate, a2f_avatar_instance)
    return


def configure_api(api_key, proxy_url=None):
    if proxy_url:
        os.environ['https_proxy'] = proxy_url if proxy_url else None
        os.environ['http_proxy'] = proxy_url if proxy_url else None
    genai.configure(api_key=api_key)


def get_response(question):
    model_path = 'gemini-1.5-flash'
    configure_api(GOOGLE_API_KEY, None)
    model = genai.GenerativeModel(model_path)
    response = model.generate_content(question, stream=True)
    for res in response:
        if hasattr(res, 'text') and res.text:
            yield res.text


def main():
    # lang = input("Enter the language: ")
    # print(lang.lower()[:2])

    lang = "en"

    with ignoreStderr():
        with sr.Microphone() as source:
            asr.adjust_for_ambient_noise(source, duration=5)

            make_avatar_speaks("Tell me your selected language please, english or arabic", "en")
            print('Select a language')
            audio=asr.listen(source)
            is_valid_input, _input = speech_to_text(audio, "en")
            if is_valid_input:
                lang = _input.lower().strip()[:2]
                print("Selected Language: ", _input)
                print("Language:", lang)

            while True:
                print('Say something')
                audio=asr.listen(source)
                is_valid_input, _input = speech_to_text(audio, lang)
                if is_valid_input:
                    print("User : ", _input)
                    text = ""
                    for sentence in get_response(_input):
                        pattern = re.compile(r'[*#,]')
                        text += pattern.sub('', sentence)
                    print("Avatar : ", text)
                    make_avatar_speaks(text, lang)
                else:
                    if _input is sr.RequestError:
                        print("No response from Google Speech Recognition service: {0}".format(_input))


if __name__ == "__main__":
    main()

