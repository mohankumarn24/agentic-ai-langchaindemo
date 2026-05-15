import os
import whisper

def get_transcription(audio_file):

    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result['text']

audio_file = "sample_english.m4a"

transcription = get_transcription(audio_file)

print(transcription)
