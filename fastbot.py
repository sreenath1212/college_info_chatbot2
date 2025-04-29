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

# --- Centered Title and Subtitle ---

st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .centered-subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #555;
        margin-bottom: 1em;
    }
    </style>

    <div class="centered-title">🎓 College Info Assistant</div>
    <div class="centered-subtitle">
        An Intelligent Chatbot for College Search Powered by FAISS, Sentence Transformers, and OpenRouter LLMs
        /n ALL MESSEGES ARE CURRENTLY MONITORED BY ADMIN FOR IMPROVING CHATBOT RESPONSE
    </div>
    <hr>
""", unsafe_allow_html=True)


# (Continue your main application logic from here)

# --- Configuration ---
CSV_FILE = 'standardized_finatdata.csv'
TXT_FILE = 'institution_descriptions.txt'
OPENROUTER_API_KEYS = [
    st.secrets["OPENROUTER_API_KEY_1"],
    st.secrets["OPENROUTER_API_KEY_2"],
    st.secrets["OPENROUTER_API_KEY_3"],
    st.secrets["OPENROUTER_API_KEY_4"],


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
    prompt = f"""You are a friendly and knowledgeable college assistant.
Help the user by providing clear, helpful, and polite answers.
If schools and IHRD centers are involved, mention them separately to the user.
If exact information is unavailable, respond politely without mentioning internal data or context.
Use external data only when the user specifically asks about a route map.
Route map is an exception: you may freely provide any available information regarding it.
At the end of the conversation not the first time messages like "hi", ask the user whether they need help with anything else.

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

if not st.session_state["messages"]:
    welcome_message = "👋 Hello! How can I help you today? I can assist you with any college information you need."
    st.session_state["messages"].append({"role": "assistant", "content": welcome_message})
    save_memory()



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
