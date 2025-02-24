import json
import queue
import sounddevice as sd
import vosk
import pyttsx3
import os
from datetime import datetime

# Load Vosk Model
model_path = r"D:\vosk-model-en-in-0.5"
model = vosk.Model(model_path)  # Initialize the model

# Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speaking rate (words per minute)

# Callback function to process audio
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

# Initialize queue for audio processing
q = queue.Queue()

def recognize_speech():
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                           channels=1, callback=callback):
        print("Listening...")
        rec = vosk.KaldiRecognizer(model, 16000)  # Use the initialized model
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                return result.get("text", "")

def respond(text):
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def log_command(command):
    with open("command_log.txt", "a") as f:
        f.write(f"{datetime.now()}: {command}\n")

def assistant():
    try:
        while True:
            command = recognize_speech()
            if command:
                print(f"You: {command}")
                log_command(command)

                if "hello" in command.lower():
                    respond("Hello! How can I assist you?")
                elif "your name" in command.lower():
                    respond("I am your AI assistant.")
                elif "exit" in command.lower():
                    respond("Goodbye!")
                    break
                else:
                    respond("I'm not sure how to respond to that. Could you please repeat?")
                    respond(f"Did you say: {command}")
            else:
                respond("I didn't catch that. Could you please repeat?")
    except Exception as e:
        error_message = f"Error: {e}"
        print(error_message)
        respond("An error occurred. Please check the logs.")
        with open("error_log.txt", "a") as f:
            f.write(f"{datetime.now()}: {error_message}\n")

# Start Assistant
assistant()
