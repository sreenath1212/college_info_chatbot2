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
    initial_sidebar_state="collapsed"
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
    st.session_state["dark_mode"] = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"], key="dark_mode_toggle") # Added a unique key

# Inject dynamic CSS based on mode
st.markdown(f"""
<style>

/* Root colors */
:root {{
  --color-bg-light: #ffffff;
  --color-bg-dark: #0e1117;
  --color-text-light: #111827;
  --color-text-dark: #f3f4f6;
  --color-secondary-light: #6b7280;
  --color-secondary-dark: #9ca3af;
  --color-accent: #3b82f6;
  --color-border-light: #e5e7eb;
  --color-border-dark: #374151;
  --color-card-light: #f9fafb;
  --color-card-dark: #1f2937;
}}

body {{
  background-color: { "#0e1117" if st.session_state["dark_mode"] else "#ffffff" };
  color: { "#f3f4f6" if st.session_state["dark_mode"] else "#111827" };
  font-family: 'Inter', 'Segoe UI', sans-serif;
}}

/* Center header */
h1 {{
    text-align: center;
    font-size: 2.5rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    color: { "#f9fafb" if st.session_state["dark_mode"] else "#1e293b" };
}}

p {{
    text-align: center;
    font-size: 1.1rem;
    color: { "#9ca3af" if st.session_state["dark_mode"] else "#6b7280" };
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: { "#1f2937" if st.session_state["dark_mode"] else "#f9fafb" };
    border-right: 1px solid { "#374151" if st.session_state["dark_mode"] else "#e5e7eb" };
    padding: 1rem;
}}

/* Chat bubbles */
.chat-bubble {{
  background-color: { "#1f2937" if st.session_state["dark_mode"] else "#f3f4f6" };
  border: 1px solid { "#374151" if st.session_state["dark_mode"] else "#e5e7eb" };
  border-radius: 1rem;
  padding: 1rem 1.5rem;
  margin: 1rem 0;
  font-size: 1rem;
  color: { "#f9fafb" if st.session_state["dark_mode"] else "#1e293b" };
  line-height: 1.6;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}}

/* Buttons */
button[kind="secondary"] {{
    background-color: var(--color-accent);
    color: #ffffff;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    border: none;
    border-radius: 0.6rem;
    transition: all 0.3s ease;
}}

button[kind="secondary"]:hover {{
    background-color: #2563eb;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(59, 130, 246, 0.3);
}}

/* Chat input box */
.css-15zrgzn.eqr7zpz3 {{
    background-color: { "#1f2937" if st.session_state["dark_mode"] else "#f9fafb" };
    border: 1px solid { "#374151" if st.session_state["dark_mode"] else "#e5e7eb" };
    color: { "#f3f4f6" if st.session_state["dark_mode"] else "#111827" };
    border-radius: 0.5rem;
    padding: 0.75rem;
}}

/* Text Input */
input[type="text"] {{
  background-color: inherit;
  color: inherit;
  border: none;
  outline: none;
  font-size: 1rem;
}}

/* Scrollbar */
::-webkit-scrollbar {{
  width: 8px;
}}
::-webkit-scrollbar-thumb {{
  background: { "#374151" if st.session_state["dark_mode"] else "#d1d5db" };
  border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
  background: { "#3b82f6" };
}}

/* Loader spinner */
.css-1v0mbdj.e1tzin5v2 {{
  color: { "#3b82f6" };
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

st.markdown(
    """
    <div style="position: fixed; top: 3.5rem; width: 100%; background-color: #0e1117; padding: 10px 0; z-index: 1000; text-align: center;">
        <h1 style="margin: 0; font-size: 28px; color: white;">🎓 College Info Assistant</h1>
        <p style="margin: 0; font-size: 16px; color: #cccccc;">Ask anything about colleges — accurate, fast, and friendly!</p>
        <hr style="margin-top:10px; border:1px solid #333;">
    </div>
    <br><br><br><br><br><br>
    """,
    unsafe_allow_html=True,
)



# Sidebar: Chat History
with st.sidebar:
    st.header("🕑 Chat History")
    if st.session_state["messages"]:
        for idx, msg in enumerate(st.session_state["messages"]):
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content'][:30]}...")
    else:
        st.markdown("*No chats yet.*")

    if st.button("🧹 Clear Chat"):
        st.session_state["messages"] = []
        save_memory()
        st.rerun()

    if st.button("📥 Download Chat"):
        if st.session_state["messages"]:
            chat_text = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state["messages"]])
            st.download_button("Download as TXT", data=chat_text, file_name="chat_history.txt", mime="text/plain")

# Display Messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='chat-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

# User Input
user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(f"<div class='chat-bubble'>{user_query}</div>", unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        context = retrieve_relevant_context(user_query, TOP_K)
        raw_answer = ask_openrouter(context, user_query)

    final_answer = ""
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        for i in range(len(raw_answer)):
            final_answer = raw_answer[:i+1]
            answer_placeholder.markdown(f"<div class='chat-bubble'>{final_answer}</div>", unsafe_allow_html=True)
            time.sleep(0.01)

    st.session_state["messages"].append({"role": "assistant", "content": raw_answer})
    save_memory()
