# ==========================
# Streamlit + LangChain GenAI Demo
# Use Cases: Text, Audio, Image, Video
# Models: OpenAI, Gemini, Ollama, Mistral, HuggingFace Chat, DeepSeek (HF)
# ==========================

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from openai import OpenAI
import google.generativeai as genai
import subprocess
import tempfile
import os
from PIL import Image
from moviepy import VideoFileClip
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# API Keys
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY"))
# genai.configure(api_key=os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY"))
# hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

openai_client = st.secrets.get("OPENAI_API_KEY")
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY"))
hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
hf_model_name = "mistralai/Mistral-7B-Instruct-v0.1"  
hf_provider = "auto"  # Use a valid provider string like 'huggingface'

hf_client = InferenceClient(model=hf_model_name,token=hf_token) if hf_token else None

# UI Setup
st.set_page_config(page_title="GenAI Assistant", layout="wide", page_icon="🚀")

# Professional CSS Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.main-header {
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
    padding: 3rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
}
.stButton > button {
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
    color: white;
    border: none;
    border-radius: 20px;
    padding: 0.7rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.95);
    border-radius: 12px;
    border: 2px solid transparent;
    color: #333 !important;
}
.stTextArea > div > div > textarea:focus {
    border: 2px solid #4ecdc4;
    box-shadow: 0 0 15px rgba(78, 205, 196, 0.3);
}
.response-box {
    background: linear-gradient(135deg, #e8f5e8, #e3f2fd);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 4px solid #4ecdc4;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3.5rem; margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        🚀 GenAI Assistant
    </h1>
    <p style="font-size: 1.3rem; margin: 1rem 0; color: rgba(255,255,255,0.9);">
        Professional Multi-Modal AI Platform
    </p>
    <div style="display: flex; justify-content: center; gap: 2.5rem; margin-top: 1.5rem; flex-wrap: wrap;">
        <div style="text-align: center; color: white;">
            <div style="font-size: 2.5rem;">💬</div>
            <div style="font-weight: 500;">Text AI</div>
        </div>
        <div style="text-align: center; color: white;">
            <div style="font-size: 2.5rem;">🎵</div>
            <div style="font-weight: 500;">Audio AI</div>
        </div>
        <div style="text-align: center; color: white;">
            <div style="font-size: 2.5rem;">🖼️</div>
            <div style="font-weight: 500;">Vision AI</div>
        </div>
        <div style="text-align: center; color: white;">
            <div style="font-size: 2.5rem;">🎬</div>
            <div style="font-weight: 500;">Video AI</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Professional Sidebar
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b, #4ecdc4); padding: 1.5rem; border-radius: 15px; text-align: center; margin-bottom: 1.5rem;">
        <h3 style="color: white; margin: 0; font-weight: 600;">🎯 AI Models</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0.3rem 0 0 0; font-size: 0.9rem;">Select Your AI Engine</p>
    </div>
    """, unsafe_allow_html=True)
    
    llm_choice = st.selectbox("AI Model", [
    "🤖 OpenAI GPT", "✨ Gemini", "🦙 Ollama (LLaMA)", "🌟 Mistral (Ollama)",
    "🤗 HuggingFace Chat", "🧠 DeepSeek (HF)",
    "🎵 Audio Transcript", "👁️ Image Vision", "🎬 Video Summary", "🎨 Text to Image"
])
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.95); padding: 1.2rem; border-radius: 12px; margin-top: 1.5rem;">
        <h5 style="color: #333; margin-bottom: 0.8rem;">✨ Features</h5>
        <div style="font-size: 0.85rem; color: #666; line-height: 1.4;">
            ✓ Multiple AI Models<br>
            ✓ Real-time Processing<br>
            ✓ Multi-Modal Support<br>
            ✓ Secure & Private
        </div>
    </div>
    """, unsafe_allow_html=True)

model_map = {
    "🤖 OpenAI GPT": "OpenAI GPT",
    "✨ Gemini": "Gemini",
    "🦙 Ollama (LLaMA)": "Ollama (LLaMA)",
    "🌟 Mistral (Ollama)": "Mistral (Ollama)",
    "🤗 HuggingFace Chat": "HuggingFace Chat",
    "🧠 DeepSeek (HF)": "DeepSeek (HF)",
    "🎵 Audio Transcript": "Audio Transcript",
    "👁️ Image Vision": "Image Vision",
    "🎬 Video Summary": "Video Summary",
    "🎨 Text to Image": "Text to Image"
}

clean_choice = model_map[llm_choice]
# Input Section
st.markdown("""
<div style="background: rgba(255,255,255,0.95); padding: 1rem; border-radius: 12px; margin: 1rem 0; text-align: center;">
    <h5 style="color: #333; margin-bottom: 0.5rem;">💭 Your Prompt</h5>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "Enter your prompt:", 
    height=100, 
    placeholder="Ask me anything!"
)

# TEXT HANDLER
def handle_text(prompt):
    if clean_choice == "OpenAI GPT":
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        response = llm([HumanMessage(content=prompt)])
        return response.content

    elif clean_choice == "Gemini":
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text

    elif clean_choice == "Ollama (LLaMA)":
        try:
            result = subprocess.run(
                ["ollama", "run", "llama3.2"],
                input=prompt, capture_output=True, text=True, check=True, timeout=30
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error running Ollama LLaMA: {e}"

    elif clean_choice == "Mistral (Ollama)":
        try:
            result = subprocess.run(
                ["ollama", "run", "mistral"],
                input=prompt, capture_output=True, text=True, check=True, timeout=30
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error running Ollama Mistral: {e}"

    elif clean_choice == "HuggingFace Chat":
        if not hf_client:
            return "Error: Hugging Face API token not set."
        try:
            response = hf_client.text_generation(prompt, max_new_tokens=100)
            if isinstance(response, list) and len(response) > 0:
                return response[0].get("generated_text", "")
            else:
                return str(response)
        except Exception as e:
            return f"HuggingFace Error: {str(e)}"

    elif clean_choice == "DeepSeek (HF)":
        if not hf_client:
            return "Error: Hugging Face API token not set."
        try:
            response = hf_client.text_generation(prompt, max_new_tokens=150)
            if isinstance(response, list) and len(response) > 0:
                return response[0].get("generated_text", "")
            else:
                return str(response)
        except Exception as e:
            return f"DeepSeek Error: {str(e)}"

# AUDIO HANDLER
def handle_audio(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file.read())
        path = tmp.name
    with open(path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

# IMAGE HANDLER
def handle_image(file):
    image = Image.open(file)
    model = genai.GenerativeModel("gemini-1.5-flash")
    result = model.generate_content(["Describe this image:", image])
    return result.text

# VIDEO HANDLER
def handle_video(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file.read())
        video_path = tmp.name
    audio_path = video_path.replace(".mp4", ".mp3")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    with open(audio_path, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        ).text
    summary = handle_text("Summarize this video: " + transcript)
    return transcript, summary

# TEXT TO IMAGE HANDLER
def handle_text_to_image(prompt):
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        return f"Error generating image: {e}"

# RUN LOGIC
if clean_choice in ["OpenAI GPT", "Gemini", "Ollama (LLaMA)", "Mistral (Ollama)", "HuggingFace Chat", "DeepSeek (HF)"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Response", use_container_width=True) and user_input:
            with st.spinner("🤖 Processing your request..."):
                response = handle_text(user_input)
            
            st.markdown("""
            <div class="response-box">
                <h5 style="color: #333; margin-bottom: 1rem;">
                    🎯 AI Response
                </h5>
            </div>
            """, unsafe_allow_html=True)
            st.write(response)

elif clean_choice == "Audio Transcript":
    file = st.file_uploader("Upload audio file", type=["mp3", "wav"])
    if file and st.button("Transcribe & Answer"):
        with st.spinner("Transcribing audio..."):
            text = handle_audio(file)
        st.markdown("**Transcript:**")
        st.info(text)
        st.markdown("**Answer:**")
        response = handle_text(text)
        st.write(response)

elif clean_choice == "Image Vision":
    img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if img and st.button("Describe Image"):
        st.image(img, use_column_width=True)
        with st.spinner("Analyzing image..."):
            desc = handle_image(img)
        st.write(desc)

elif clean_choice == "Text to Image":
    if st.button("🎨 Generate Image") and user_input:
        with st.spinner("🎨 Creating your image..."):
            image_url = handle_text_to_image(user_input)
        if image_url.startswith("http"):
            st.image(image_url, caption="Generated Image")
        else:
            st.error(image_url)

elif clean_choice == "Video Summary":
    vid = st.file_uploader("Upload video", type=["mp4", "mov"])
    if vid and st.button("Summarize Video"):
        with st.spinner("Processing video..."):
            transcript, summary = handle_video(vid)
        st.markdown("**Transcript:**")
        st.info(transcript)
        st.markdown("**Summary:**")
        st.write(summary)

# Professional Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 3rem; background: rgba(255,255,255,0.1); border-radius: 15px;">
    <h5 style="color: white; margin-bottom: 0.5rem;">🎆 Powered by Advanced AI</h5>
    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-bottom: 1rem;">Professional AI Solutions for Modern Businesses</p>
    <div style="display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap;">
        <span style="color: rgba(255,255,255,0.9); font-size: 0.8rem;">🔒 Secure</span>
        <span style="color: rgba(255,255,255,0.9); font-size: 0.8rem;">⚡ Fast</span>
        <span style="color: rgba(255,255,255,0.9); font-size: 0.8rem;">🌍 Reliable</span>
    </div>
</div>
""", unsafe_allow_html=True)

