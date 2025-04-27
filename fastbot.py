# Install these packages if not already:
# pip install streamlit pandas sentence-transformers faiss-cpu requests

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

# --------------------------------
# MUST BE FIRST Streamlit command
st.set_page_config(
    page_title="🎓 College Info Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --------------------------------

# Inject Custom CSS
st.markdown("""
<style>
    /* Background Gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #e0f7fa, #e1bee7);
    }

    /* Flex for aligning chat bubbles */
    .stChatMessage {
        display: flex;
        width: 100%;
        margin: 0.5rem 0;
    }

    .stChatMessage.user {
        justify-content: flex-end;
    }

    .stChatMessage.assistant {
        justify-content: flex-start;
    }

    /* Chat bubble style */
    .chat-bubble {
        max-width: 70%;
        padding: 1rem;
        border-radius: 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        background: #ffffff !important;
        color: #000000 !important;
        position: relative;
        transition: 0.2s;
        font-size: 1rem;
    }

    .stChatMessage.user .chat-bubble {
        background: #dcedc8 !important;
        border-bottom-right-radius: 0.2rem;
    }

    .stChatMessage.assistant .chat-bubble {
        background: #f8bbd0 !important;
        border-bottom-left-radius: 0.2rem;
    }

    .chat-bubble:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* Chat input box style */
    [data-testid="stChatInput"] {
        background: #ffffff;
        border: 2px solid #f8bbd0;
        border-radius: 2rem;
        padding: 0.5rem 1rem;
        color: #333333;
        font-size: 1rem;
    }

    /* Make all headings properly visible */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937; /* Darker shade for perfect visibility */
        font-weight: 700; /* Bold */
        margin-bottom: 0.5rem;
    }

    /* Make normal markdown text also more readable */
    p, li, span, div {
        color: #333333;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------
# CONFIG
CSV_FILE = 'cleaned_dataset.csv'
TXT_FILE = 'institution_descriptions.txt'
OPENROUTER_API_KEY = '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'
MODEL = 'google/gemini-2.0-flash-exp:free'
# --------------------------------

# --------------------------------
# 1. BATCH GENERATE METADATA
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

# --------------------------------
# 2. LOAD DATA & EMBED
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

# --------------------------------
# 3. RETRIEVE RELEVANT CONTEXT
def retrieve_relevant_context(query, top_k):
    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)
    context = "\n\n".join([texts[i] for i in indices[0]])
    return context

# --------------------------------
# 4. ASK OPENROUTER
def ask_openrouter(context, question):
    prompt = f"""You are a friendly and helpful college information assistant. Answer based on CONTEXT. If unsure, say 'I couldn't find that specific information.' 

CONTEXT:
{context}

USER QUESTION:
{question}

Answer:"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
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
            return f"❌ OpenRouter error: {data}"

    except Exception as e:
        return f"❌ Error contacting OpenRouter: {e}"

# --------------------------------
# MAIN START
generate_metadata_from_csv(CSV_FILE, TXT_FILE)

model, texts, index = load_data_and_embeddings()
TOP_K = len(texts)

# --------------------------------
# STREAMLIT CHATBOT INTERFACE
st.title("🎓 College Info Assistant")
st.markdown("##### Ask anything about colleges — accurate, fast, and friendly!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

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
        st.experimental_rerun()

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
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(f"<div class='chat-bubble'>{user_query}</div>", unsafe_allow_html=True)

    # Show loading animation
    with st.spinner("Thinking..."):
        context = retrieve_relevant_context(user_query, TOP_K)
        raw_answer = ask_openrouter(context, user_query)

    # Typing effect simulation
    final_answer = ""
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        for i in range(len(raw_answer)):
            final_answer = raw_answer[:i+1]
            answer_placeholder.markdown(f"<div class='chat-bubble'>{final_answer}</div>", unsafe_allow_html=True)
            time.sleep(0.01)

    # Add bot answer
    st.session_state["messages"].append({"role": "assistant", "content": raw_answer})
