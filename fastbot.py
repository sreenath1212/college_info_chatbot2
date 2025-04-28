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
    st.session_state["dark_mode"] = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"], key="dark_mode_toggle")  # Unique key

# Inject dynamic CSS based on mode
st.markdown(f"""
<style>
/* CSS styles (same as you provided) */
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(to right, {('#0f2027, #203a43, #2c5364') if st.session_state["dark_mode"] else '#e0f7fa, #e1bee7'});
    padding-top: 2rem;
}}
/* ... (rest of your CSS exactly same, skipped here to save space) ... */
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
CSV_FILE = 'cleaned_dataset.csv'
TXT_FILE = 'institution_descriptions.txt'
OPENROUTER_API_KEYS = [
    st.secrets["OPENROUTER_API_KEY_1"],
    st.secrets["OPENROUTER_API_KEY_2"],
    # Add more if needed
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

# Initialize model, texts, index safely
if "model" not in st.session_state:
    model, texts, index = load_data_and_embeddings()
    st.session_state["model"] = model
    st.session_state["texts"] = texts
    st.session_state["index"] = index
else:
    model = st.session_state["model"]
    texts = st.session_state["texts"]
    index = st.session_state["index"]

def retrieve_relevant_context(query, top_k):
    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)
    context = "\n\n".join([texts[i] for i in indices[0]])
    return context

def ask_openrouter(context, question):
    prompt = f"""You are a friendly and helpful college information assistant. Answer based on CONTEXT. If unsure, say 'I couldn't find that specific information.'\n\nCONTEXT:\n{context}\n\nUSER QUESTION:\n{question}\n\nAnswer:"""

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
                return "❌ OpenRouter API error."
    except Exception as e:
        return f"❌ Error: {e}"

# --- Streamlit App ---

st.title("🎓 College Info Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
prompt = st.chat_input("Ask about colleges...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching information..."):
            context = retrieve_relevant_context(prompt, top_k=5)
            response = ask_openrouter(context, prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Clear Chat Button ---
with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

