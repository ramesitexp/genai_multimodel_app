# ==========================
# Streamlit + LangChain GenAI Demo - STYLED VERSION
# Use Cases: Text, Audio, Image, Video
# Models: OpenAI, Gemini, Ollama, Mistral, HuggingFace Chat, DeepSeek (HF)
# ==========================

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import openai
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
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY"))
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
hf_client = InferenceClient(hf_token) if hf_token else None

# UI Setup
st.set_page_config(page_title="GenAI Assistant", layout="wide", page_icon="🚀")

# Custom CSS for attractive design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
}
.main-header {
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
    padding: 3rem 2rem;
    border-radius: 25px;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
}
.stButton > button {
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.9);
    border-radius: 15px;
    border: 2px solid transparent;
}
.stTextArea > div > div > textarea:focus {
    border: 2px solid #4ecdc4;
    box-shadow: 0 0 20px rgba(78, 205, 196, 0.3);
}
.response-box {
    background: linear-gradient(135deg, #e8f5e8, #e3f2fd);
    padding: 2rem;
    border-radius: 20px;
    border-left: 5px solid #4ecdc4;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 4rem; margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        🚀 GenAI Assistant
    </h1>
    <p style="font-size: 1.5rem; margin: 1rem 0; color: rgba(255,255,255,0.9);">
        Next-Generation Multi-Modal AI Platform
    </p>
    <div style="display: flex; justify-content: center; gap: 3rem; margin-top: 2rem; flex-wrap: wrap;">
        <div style="text-align: center; color: white;">
            <div style="font-size: 3rem;">💬</div>
            <div style="font-weight: 600;">Smart Chat</div>
        </div>
        <div style="text-align: center; color: white;">
            <div style="font-size: 3rem;">🎵</div>
            <div style="font-weight: 600;">Audio AI</div>
        </div>
        <div style="text-align: center; color: white;">
            <div style="font-size: 3rem;">🖼️</div>
            <div style="font-weight: 600;">Vision AI</div>
        </div>
        <div style="text-align: center; color: white;">
            <div style="font-size: 3rem;">🎬</div>
            <div style="font-weight: 600;">Video AI</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b, #4ecdc4); padding: 2rem 1rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0; font-weight: 700;">🎯 AI Control Center</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Select Your AI Model</p>
    </div>
    """, unsafe_allow_html=True)
    
    llm_choice = st.selectbox("Choose AI Mode", [
        "🤖 OpenAI GPT", "✨ Gemini", "🦙 Ollama (LLaMA)", "🌟 Mistral (Ollama)",
        "🤗 HuggingFace Chat", "🧠 DeepSeek (HF)",
        "🎵 Audio Transcript", "👁️ Image Vision", "🎬 Video Summary"
    ])
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.95); padding: 1.5rem; border-radius: 15px; margin-top: 2rem;">
        <h4 style="color: #333; margin-bottom: 1rem;">🌟 Premium Features</h4>
        <div style="text-align: left; font-size: 0.9rem; color: #666;">
            ✅ Multiple AI Models<br>
            ✅ Real-time Processing<br>
            ✅ Multi-Modal Support<br>
            ✅ Enterprise Security<br>
            ✅ 24/7 Availability
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
    "🎬 Video Summary": "Video Summary"
}

clean_choice = model_map[llm_choice]

# Input Section
st.markdown("""
<div style="background: rgba(255,255,255,0.95); padding: 1.5rem; border-radius: 20px; margin: 2rem 0; text-align: center;">
    <h3 style="color: #333; margin-bottom: 1rem;">💭 Your AI Prompt</h3>
    <p style="color: #666; margin-bottom: 1rem;">Enter your question or request below</p>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "What can I help you with today?", 
    height=120, 
    placeholder="Ask me anything! I can process text, analyze images, transcribe audio, and summarize videos..."
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
        response = hf_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content

    elif clean_choice == "DeepSeek (HF)":
        if not hf_client:
            return "Error: Hugging Face API token not set."
        response = hf_client.text_generation(
            prompt=prompt,
            model="deepseek-ai/deepseek-llm-7b-chat",
            max_new_tokens=400,
            temperature=0.7
        )
        return response.generated_text

# AUDIO HANDLER
def handle_audio(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file.read())
        path = tmp.name
    return openai.Audio.transcribe("whisper-1", open(path, "rb"))["text"]

# IMAGE HANDLER
def handle_image(file):
    image = Image.open(file)
    model = genai.GenerativeModel("gemini-pro-vision")
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
        transcript = openai.Audio.transcribe("whisper-1", f)["text"]
    summary = handle_text("Summarize this video: " + transcript)
    return transcript, summary

# RUN LOGIC
if clean_choice in ["OpenAI GPT", "Gemini", "Ollama (LLaMA)", "Mistral (Ollama)", "HuggingFace Chat", "DeepSeek (HF)"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate AI Response", use_container_width=True) and user_input:
            with st.spinner("🤖 AI is processing your request..."):
                response = handle_text(user_input)
            
            st.markdown("""
            <div class="response-box">
                <h3 style="color: #333; margin-bottom: 1rem;">
                    🎯 AI Response
                </h3>
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

elif clean_choice == "Video Summary":
    vid = st.file_uploader("Upload video", type=["mp4", "mov"])
    if vid and st.button("Summarize Video"):
        with st.spinner("Processing video..."):
            transcript, summary = handle_video(vid)
        st.markdown("**Transcript:**")
        st.info(transcript)
        st.markdown("**Summary:**")
        st.write(summary)

# Footer
st.markdown("""
<div style="text-align: center; padding: 3rem 2rem; margin-top: 4rem; background: rgba(255,255,255,0.1); border-radius: 25px;">
    <h3 style="color: white; margin-bottom: 1rem;">🌟 Powered by Advanced AI</h3>
    <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; margin-bottom: 2rem;">Experience the future of artificial intelligence</p>
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">🔒 Secure & Private</div>
        <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">⚡ Lightning Fast</div>
        <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">🌍 Global Access</div>
    </div>
</div>
""", unsafe_allow_html=True)