import streamlit as st
import subprocess
import whisperx
import torch
import os
import json
import gdown
import re
import requests

from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel
import vertexai

# ----------------------
# Configuration
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_size = "base"
batch_size = 16
temp_path = "temp_audio.wav"

# WhisperX model
model = whisperx.load_model(model_size, device, compute_type="float32")

# Gemini model setup
credentials = service_account.Credentials.from_service_account_file(
    'rag-key.json',
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
vertexai.init(project="rag-test-jungroo", location="us-central1", credentials=credentials)
qa_model = GenerativeModel("gemini-1.5-flash-002")

# ----------------------
# Helper Functions
# ----------------------

def download_youtube(url, output_path):
    subprocess.run(["yt-dlp", "-f", "bestaudio", "-o", output_path, url], check=True)

def download_gdrive(url, output_path):
    file_id_match = re.search(r"id=([^&]+)", url) or re.search(r"/d/([^/]+)", url)
    if not file_id_match:
        raise ValueError("Invalid Google Drive URL.")
    file_id = file_id_match.group(1)
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

    # Check if valid audio
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise ValueError("Downloaded file is not valid or too small.")

def generate_mcqs_for_segment(segment_text, timestamp):
    prompt = f"""
You are an educational question generator.

Given the following transcript segment from a lecture:

\"\"\"{segment_text}\"\"\"

1. Identify if this contains any key concepts or explanations.
2. If yes, generate **multiple choice question** (MCQ) on all the key concepts to test a student's understanding.
3. For each MCQ, provide:
    - Question
    - Four options (A–D)
    - Correct answer (just the option letter)
    - A difficulty label: Easy, Medium, or Hard

Return in this format:
---
Question: <text>
Choices:
A. <choice1>
B. <choice2>
C. <choice3>
D. <choice4>
Answer: <correct letter>
Difficulty: <Easy/Medium/Hard>
---

Only generate questions if there is a meaningful concept present.
"""
    try:
        response = qa_model.generate_content(prompt)
        mcqs = []
        blocks = response.text.strip().split("---")
        for block in blocks:
            if "Question:" in block and "Answer:" in block and "Difficulty:" in block:
                mcqs.append(block.strip())
        return {
            "timestamp": timestamp,
            "mcqs": mcqs
        } if mcqs else None
    except Exception as e:
        print(f"Gemini error: {e}")
        return None

def label_segment_type(segment_text):
    prompt = f"""
You're an AI specialized in analyzing lecture transcripts.

Classify the following transcript segment into one of:
- Key Concept (main idea or core takeaway)
- Definition (explains what something means)
- Example (illustrates a concept)
- Other (filler, greeting, off-topic, etc.)

Transcript segment:
\"\"\"{segment_text}\"\"\"

Respond with just one of the following labels:
Key Concept, Definition, Example, Other
"""
    try:
        response = qa_model.generate_content(prompt)
        label = response.text.strip()
        return label if label in ["Key Concept", "Definition", "Example"] else "Other"
    except Exception as e:
        print(f"Labeling error: {e}")
        return "Other"

def generate_single_reasonable_mcq(segment_text, timestamp):
    prompt = f"""
You are a helpful quiz maker.

Given the following transcript segment, try to create **only one plausible MCQ** based on the context or comprehension, **if possible**. If there is nothing to ask, respond with "None".

Segment:
\"\"\"{segment_text}\"\"\"

Format:
---
Question: <text>
Choices:
A. <choice1>
B. <choice2>
C. <choice3>
D. <choice4>
Answer: <correct letter>
Difficulty: <Easy/Medium/Hard>
---
"""

    try:
        response = qa_model.generate_content(prompt)
        text = response.text.strip()
        if "Question:" in text and "Answer:" in text:
            return {
                "timestamp": timestamp,
                "mcqs": [text]
            }
    except Exception as e:
        print(f"Fallback MCQ generation error: {e}")
    return None

def transcribe_file(file_path, delete_after=False):
    result = model.transcribe(file_path, batch_size=batch_size)
    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], align_model, metadata, file_path, device)

    segments = []
    questions = []

    for seg in result_aligned["segments"]:
        text = seg["text"].strip()
        label = label_segment_type(text)

        segment_data = {
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
            "label": label
        }
        segments.append(segment_data)

        if label != "Other":
            qset = generate_mcqs_for_segment(text, seg["end"])
            if qset:
                questions.append(qset)
        else:
            qset = generate_single_reasonable_mcq(text, seg["end"])
            if qset:
                questions.append(qset)

    if delete_after:
        os.remove(file_path)

    return segments, questions


# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="🧠 Speech-to-MCQ Generator", layout="wide")
st.title("🎧 Speech-to-MCQ Generator")
st.write("Upload or link a video/audio file. This app transcribes it, extracts key ideas, and generates MCQs with difficulty levels.")

option = st.radio("Choose input source:", ["YouTube URL", "Google Drive URL", "Upload file"])

if option == "YouTube URL":
    youtube_url = st.text_input("Enter YouTube URL")
    if st.button("Download and Transcribe") and youtube_url:
        try:
            download_youtube(youtube_url, temp_path)
            segments, questions = transcribe_file(temp_path, delete_after=True)
            st.success("Transcription and MCQ generation complete!")

            with st.expander("📝 Transcript with Tags"):
                for seg in segments:
                    st.markdown(f"**[{seg['start']:.2f} - {seg['end']:.2f}] ({seg['label']})** {seg['text']}")

            st.subheader("❓ Generated MCQs:")
            for qset in questions:
                st.markdown(f"**🕒 At {qset['timestamp']:.2f}s**")
                for mcq in qset['mcqs']:
                    st.code(mcq)


        except Exception as e:
            st.error(f"❌ Error: {e}")

elif option == "Google Drive URL":
    gdrive_url = st.text_input("Enter Google Drive URL")
    if st.button("Download and Transcribe") and gdrive_url:
        try:
            download_gdrive(gdrive_url, temp_path)
            segments, questions = transcribe_file(temp_path, delete_after=True)
            st.success("Transcription and MCQ generation complete!")

            with st.expander("📝 Transcript with Tags"):
                for seg in segments:
                    st.markdown(f"**[{seg['start']:.2f} - {seg['end']:.2f}] ({seg['label']})** {seg['text']}")

            st.subheader("❓ Generated MCQs:")
            for qset in questions:
                st.markdown(f"**🕒 At {qset['timestamp']:.2f}s**")
                for mcq in qset['mcqs']:
                    st.code(mcq)

        except Exception as e:
            st.error(f"❌ Error: {e}")

elif option == "Upload file":
    uploaded_file = st.file_uploader("Upload audio/video file", type=["wav", "mp3", "mp4", "m4a"])
    if uploaded_file and st.button("Transcribe Uploaded File"):
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        try:
            segments, questions = transcribe_file(temp_path, delete_after=False)
            st.success("Transcription and MCQ generation complete!")

            with st.expander("📝 Transcript with Tags"):
                for seg in segments:
                    st.markdown(f"**[{seg['start']:.2f} - {seg['end']:.2f}] ({seg['label']})** {seg['text']}")


            st.subheader("❓ Generated MCQs:")
            for qset in questions:
                st.markdown(f"**🕒 At {qset['timestamp']:.2f}s**")
                for mcq in qset['mcqs']:
                    st.code(mcq)

        except Exception as e:
            st.error(f"❌ Error: {e}")
