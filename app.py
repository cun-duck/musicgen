import os
import requests
import streamlit as st
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from io import BytesIO
from pydub import AudioSegment

# Ambil API Token dari Streamlit Secrets
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]

# API URL untuk model Hugging Face
API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Fungsi untuk meminta audio dari Hugging Face API
def query_huggingface(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

# Fungsi untuk mengonversi wav ke mp3
def convert_wav_to_mp3(wav_bytes):
    audio = AudioSegment.from_wav(BytesIO(wav_bytes))
    mp3_bytes = BytesIO()
    audio.export(mp3_bytes, format="mp3")
    mp3_bytes.seek(0)
    return mp3_bytes

# Fungsi untuk menghasilkan musik menggunakan MusicGen
def generate_music(description, duration):
    model = MusicGen.get_pretrained("small")
    model.set_generation_params(duration=duration)
    wav = model.generate([description])  # Generate musik
    mp3_bytes = convert_wav_to_mp3(wav[0].cpu().numpy())  # Convert ke MP3
    return mp3_bytes

# UI Streamlit
st.title("Music Generation with MusicGen")
st.write("Generate music based on your description!")

# Input pengguna
description = st.text_input("Enter a description for the music (e.g. 'happy rock', 'energetic EDM')", "happy rock")
duration = st.slider("Select duration of the music (in seconds)", 1, 30, 8)

if st.button("Generate Music"):
    if description:
        # Generate music
        mp3_bytes = generate_music(description, duration)
        st.audio(mp3_bytes, format="audio/mp3")

        # Simpan file mp3 ke disk jika diperlukan
        with open("generated_music.mp3", "wb") as f:
            f.write(mp3_bytes.read())
        
        # Hapus file setelah 10 detik agar tidak memenuhi penyimpanan
        os.remove("generated_music.mp3")
