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
    initial_sidebar_state="auto"
)

# Force Sidebar to Collapse (for Streamlit Cloud too)
st.markdown(
    """
    <script>
        window.addEventListener('DOMContentLoaded', (event) => {
            const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.style.transform = 'translateX(-100%)'; // hide sidebar
            }
        });
    </script>
    """,
    unsafe_allow_html=True
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
/* Main App Container Background */
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(to right, {('#0f2027, #203a43, #2c5364') if st.session_state["dark_mode"] else '#e0f7fa, #e1bee7'});
    padding-top: 2rem;
}}

/* Top and Bottom Bars */
header[data-testid="stHeader"], footer {{ /* Target both top and bottom bars */
    background-color: #4B0082 !important; /* Dark Purple/Indigo */
    color: #FFFFFF !important; /* White text for contrast */
}}

footer {{
    position: fixed;
    bottom: 0;
    width: 100%;
    z-index: 1000; /* Ensure it's above other content */
    padding: 10px;
    text-align: center;
}}

/* Sidebar Background and Font */
[data-testid="stSidebar"] {{
    background-color: {('#3a4354' if st.session_state["dark_mode"] else '#e9d8fd')}; /* Changed Sidebar Background */
    color: {'#edf2f7' if st.session_state["dark_mode"] else '#1a202c'};
}}

/* Sidebar Elements (Dark Mode Toggle, etc.) */
.stSidebarContent svg {{
    color: {'#edf2f7' if st.session_state["dark_mode"] else '#1a202c'} !important;
}}

/* Sidebar Buttons */
button[kind="secondary"] {{
    background-color: {'#2d3748' if st.session_state["dark_mode"] else '#e2e8f0'};
    color: {'#edf2f7' if st.session_state["dark_mode"] else '#1a202c'};
    border: 1px solid #cbd5e0;
    border-radius: 10px;
    margin: 10px 0px;
    width: 100%;
}}
button[kind="secondary"]:hover {{
    background-color: {'#4a5568' if st.session_state["dark_mode"] else '#cbd5e0'};
    color: {'#e2e8f0' if st.session_state["dark_mode"] else '#1a202c'};
}}

/* Chat Message Styling */
.stChatMessage {{
    display: flex;
    width: 100%;
    margin: 1rem 0;
    background: none;
}}
.stChatMessage.user {{
    justify-content: flex-end;
}}
.stChatMessage.assistant {{
    justify-content: flex-start;
}}
.chat-bubble {{
    max-width: 70%;
    padding: 1rem 1.2rem;
    border-radius: 1.5rem;
    background: {('#2d3748' if st.session_state["dark_mode"] else '#ffffff')};
    color: {('#edf2f7' if st.session_state["dark_mode"] else '#1a202c')};
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    font-size: 1rem;
}}
.stChatMessage.user .chat-bubble {{
    background: {('#4fd1c5' if st.session_state["dark_mode"] else '#c6f6d5')};
    color: #1a202c;
    border-bottom-right-radius: 0.3rem;
}}
.stChatMessage.assistant .chat-bubble {{
    background: {('#805ad5' if st.session_state["dark_mode"] else '#e9d8fd')};
    color: #1a202c;
    border-bottom-left-radius: 0.3rem;
}}
.chat-bubble:hover {{
    transform: scale(1.02);
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
}}

/* Chat Input Box */
[data-testid="stChatInput"] textarea {{
    background: {('#2d3748' if st.session_state["dark_mode"] else '#ffffff')};
    border: 2px solid #4B0082; /* Dark Purple/Indigo border */
    border-radius: 2rem;
    padding: 1rem;
    color: {('#e2e8f0' if st.session_state["dark_mode"] else '#333333')};
    font-size: 1.1rem;
    transition: 0.3s ease;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: {'#63b3ed' if st.session_state["dark_mode"] else '#7c3aed'};
    outline: none;
}}

/* Sidebar Chat History Items */
section[data-testid="stSidebar"] > div > div > div:nth-child(3) {{
    margin-top: 20px;
}}
section[data-testid="stSidebar"] .element-container:nth-child(3) div {{
    background-color: {('#2d3748' if st.session_state["dark_mode"] else '#edf2f7')};
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: 0.3s ease;
    font-size: 0.9rem;
}}
section[data-testid="stSidebar"] .element-container:nth-child(3) div:hover {{
    background-color: {('#4a5568' if st.session_state["dark_mode"] else '#d1d5db')};
    cursor: pointer;
}}

/* Headings and Texts */
h1, h2, h3, h4, h5, h6 {{
    color: {'#f8fafc' if st.session_state["dark_mode"] else '#1f2937'};
}}
p, li, span, div {{
    color: {'#e2e8f0' if st.session_state["dark_mode"] else '#333333'};
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
st.markdown("##### Ask anything about colleges — accurate, fast, and friendly!")

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
