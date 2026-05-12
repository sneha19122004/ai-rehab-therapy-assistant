import pyttsx3

def initialize_tts():
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 0.9)
    print("Text-to-Speech Engine initialized successfully.")
    return tts_engine