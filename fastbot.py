import csv
import re
import multiprocessing
from multiprocessing import Pool
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import os
import time
import json
import speech_recognition as sr
import threading

# IBM Watson Text to Speech API settings
IBM_API_KEY = st.secrets["IBM_API_KEY"]
IBM_URL = st.secrets["IBM_URL"]

def text_to_speech(text):
    headers = {
        'Authorization': f'Bearer {IBM_API_KEY}',
        'Content-Type': 'application/json'
    }
    params = {
        'voice': 'en-US_AllisonV3Voice',
        'accept': 'audio/mp3'
    }
    data = {
        'text': text
    }
    response = requests.post(IBM_URL, headers=headers, params=params, json=data)
    with open('output.mp3', 'wb') as file:
        file.write(response.content)
    with open("output.mp3", "rb") as file:
        st.download_button("Download audio", file, file_name="audio.mp3")

# Function to recognize speech
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I didn't catch that."
        except sr.RequestError:
            return "Error requesting results from Google Speech Recognition service."

# --- MUST BE FIRST: Streamlit page config ---
st.set_page_config(
    page_title="🎓 College Info Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject the viewport meta tag
st.markdown(
    """
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    """,
    unsafe_allow_html=True,
)

# --- Dark Mode setup ---
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Sidebar - Settings
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.session_state["dark_mode"] = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"], key="dark_mode_toggle") 
    voice_mode = st.toggle("🗣️ Voice Mode", value=False, key="voice_mode_toggle")

# Inject dynamic CSS based on mode
st.markdown(f"""
<style>
/* Main App Container Background */
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(to right, {('#0f2027, #203a43, #2c5364') if st.session_state["dark_mode"] else '#e0f7fa, #e1bee7'});
    padding-top: 2rem;
}}
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
CSV_FILE = 'cleaned_dataset.csv'
TXT_FILE = 'institution_descriptions.txt'
OPENROUTER_API_KEYS = [
    st.secrets["OPENROUTER_API_KEY_1"],
    st.secrets["OPENROUTER_API_KEY_2"],
    
    # Add more keys as needed
]
MODEL = 'google/gemini-2.0-flash-exp:free'

if "api_key_index" not in st.session_state:
    st.session_state["api_key_index"] = 0

# --- Utility Functions ---

def clean_field_name(field_name):
    field_name = field_name.replace('_', ' ').replace('\n', ' ').strip().capitalize()
    field_name = re.sub(' +', ' ', field_name)
    return field_name

def process_row(row):
    description = ""
    institution_name = row.get('Institution_Name', '').strip()
    if institution_name:
        description += f"{institution_name}."
    else:
        description += "Institution Name: Not Available."

    for field_name, field_value in row.items():
        if not field_value:
            continue
        field_value = field_value.strip()
        if field_value.lower() in ['n', 'no', 'not available']:
            continue
        if field_name != 'Institution_Name':
            clean_name = clean_field_name(field_name)
            description += f" {clean_name}: {field_value}."

    return description.strip()

def generate_metadata_from_csv(csv_filepath, output_txt_path, num_workers=None):
    if os.path.exists(output_txt_path):
        return

    with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
        reader = list(csv.DictReader(csvfile))

    with Pool(processes=num_workers or multiprocessing.cpu_count()) as pool:
        paragraphs = pool.map(process_row, reader)

    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        for paragraph in paragraphs:
            outfile.write(paragraph + '\n' + '-' * 40 + '\n')

@st.cache_resource
def load_data_and_embeddings():
    with open(TXT_FILE, 'r', encoding='utf-8') as file:
        texts = file.read().split('----------------------------------------')
    texts = [text.strip() for text in texts if text.strip()]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return model, texts, index

def retrieve_relevant_context(query, top_k):
    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)
    context = "\n\n".join([texts[i] for i in indices[0]])
    return context

def ask_openrouter(context, question):
    prompt = f"""You are a friendly and helpful college information assistant. Answer based on CONTEXT. If unsure, say 'I couldn't find that specific information.'

    CONTEXT:
    {context}

    USER QUESTION:
    {question}

    Answer:"""

    current_api_key_index = st.session_state["api_key_index"]
    current_api_key = OPENROUTER_API_KEYS[current_api_key_index]

    headers = {
        "Authorization": f"Bearer {current_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if 'choices' in data:
            return data['choices'][0]['message']['content']
        else:
            if "rate limit" in str(data).lower() or "quota" in str(data).lower():
                st.session_state["api_key_index"] = (current_api_key_index + 1) % len(OPENROUTER_API_KEYS)
                return ask_openrouter(context, question)  
            else:
                return f"❌ OpenRouter error: {data}"

    except Exception as e:
        if "rate limit" in str(e).lower() or "quota" in str(e).lower():
            st.session_state["api_key_index"] = (current_api_key_index + 1) % len(OPENROUTER_API_KEYS)
            return ask_openrouter(context, question)  
        else:
            return f"❌ Error contacting OpenRouter: {e}"

# --- Memory persistence ---
MEMORY_FILE = "chat_memory.json"

def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state["messages"], f)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            st.session_state["messages"] = json.load(f)

# --- MAIN LOGIC START ---

generate_metadata_from_csv(CSV_FILE, TXT_FILE)

model, texts, index = load_data_and_embeddings()
TOP_K = len(texts)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    load_memory()

st.title("🎓 College Info Assistant")
st.markdown("The ultimate college companion")
st.markdown("##### Ask anything about colleges — accurate, fast, and friendly!")

if voice_mode:
    def listen_and_respond():
        user_query = recognize_speech()
        st.session_state["messages"].append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(f"<div class='chat-bubble'>{user_query}</div>", unsafe_allow_html=True)

        context = retrieve_relevant_context(user_query, TOP_K)
        raw_answer = ask_openrouter(context, user_query)

        st.session_state["messages"].append({"role": "assistant", "content": raw_answer})

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            final_answer = ""
            for i in range(len(raw_answer)):
                final_answer = raw_answer[:i+1]
                answer_placeholder.markdown(f"<div class='chat-bubble'>{final_answer}</div>", unsafe_allow_html=True)
                time.sleep(0.01)
        
        st.write("Listen to the response:")
        text_to_speech(raw_answer)
        save_memory()

    threading.Thread(target=listen_and_respond).start()

# Display Messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='chat-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

# User Input
if not voice_mode:
    user_query = st.chat_input("Type your question here...")

    if user_query:
        st.session_state["messages"].append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(f"<div class='chat-bubble'>{user_query}</div>", unsafe_allow_html=True)

        context = retrieve_relevant_context(user_query, TOP_K)
        raw_answer = ask_openrouter(context, user_query)

        st.session_state["messages"].append({"role": "assistant", "content": raw_answer})

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            final_answer = ""
            for i in range(len(raw_answer)):
                final_answer = raw_answer[:i+1]
                answer_placeholder.markdown(f"<div class='chat-bubble'>{final_answer}</div>", unsafe_allow_html=True)
                time.sleep(0.01)
        
        st.write("Listen to the response:")
        text_to_speech(raw_answer)
        save_memory()
