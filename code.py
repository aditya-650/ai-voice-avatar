import webbrowser
import os
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr
import base64
import time
import numpy as np
from dotenv import load_dotenv
from groq import Groq

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DID_API_KEY = os.getenv("DID_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

if not ELEVEN_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found in .env")

if not DID_API_KEY:
    raise ValueError("DID_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# 1. Record Voice
# -----------------------------
def record_audio(filename="input.wav", duration=7, fs=16000):
    print("🎤 Speak now...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    recording_int16 = np.int16(recording * 32767)
    wav.write(filename, fs, recording_int16)

    print("Recording saved.")

# -----------------------------
# 2. Speech to Text
# -----------------------------
def speech_to_text(filename="input.wav"):
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = r.record(source)

    text = r.recognize_google(audio)
    print("You said:", text)
    return text

# -----------------------------
# 3. Groq LLM Response
# -----------------------------
def get_llm_response(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    reply = response.choices[0].message.content
    print("AI:", reply)
    return reply

# -----------------------------
# 4. Text to Speech (ElevenLabs)
# -----------------------------
def text_to_speech(text, voice_id, output_file="response.mp3"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/zUsfQHa4yhg9PiwpxSwK"

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.7
        }
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code != 200:
        raise Exception(f"ElevenLabs error: {response.text}")

    with open(output_file, "wb") as f:
        f.write(response.content)

    print("Voice generated.")
    return output_file

# -----------------------------
# 5. Generate Talking Video (D-ID)
# -----------------------------
def create_video(text_to_speak, image_url):
    url = "https://api.d-id.com/talks"

    auth_string = DID_API_KEY + ":"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/json"
    }

    payload = {
        "source_url": image_url,
        "script": {
            "type": "text",
            "input": text_to_speak
        }
    }

    response = requests.post(url, json=payload, headers=headers)

    print("D-ID Status:", response.status_code)
    print("D-ID Response:", response.text)

    if response.status_code != 201:
        raise Exception(f"D-ID error: {response.text}")

    talk_id = response.json()["id"]

    print("Generating video...")

    while True:
        status_response = requests.get(
            f"https://api.d-id.com/talks/{talk_id}",
            headers=headers
        )

        status = status_response.json()

        if status["status"] == "done":
            video_url = status["result_url"]
            print("Video ready:", video_url)

            html_content = f"""
<html>
<head><title>AI Avatar</title></head>
<body style="background:black; display:flex; justify-content:center; align-items:center; height:100vh;">
    <video width="720" controls autoplay>
        <source src="{video_url}" type="video/mp4">
    </video>
</body>
</html>
"""

            with open("player.html", "w", encoding="utf-8") as f:
                f.write(html_content)

            webbrowser.open("player.html")
            break

        elif status["status"] == "error":
            raise Exception("D-ID video generation failed.")

        time.sleep(3)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    try:
        print("Step 1: Recording...")
        record_audio()

        print("Step 2: Speech to text...")
        user_text = speech_to_text()

        print("Step 3: LLM response...")
        ai_reply = get_llm_response(user_text)

        print("Step 4: Generating voice...")
        voice_id = "zUsfQHa4yhg9PiwpxSwK"
        audio_file = text_to_speech(ai_reply, voice_id)

        print("Step 5: Creating video...")
        image_url = "https://i.ibb.co/wFRxcdjS/beautiful-girl.png"
        create_video(ai_reply, image_url)

    except Exception as e:
        print("ERROR:", e)
