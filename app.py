import os
os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "true"

import streamlit as st
import pdfplumber
import re
import faiss
import numpy as np
import google.generativeai as genai
import time

# ----------------------------
# Configure API
# ----------------------------
api_key = "AIzaSyDsDLtX7M3tRD1GZiNHmNCIVPGKFcjpD34"
genai.configure(api_key=api_key)

# ----------------------------
# PDF Processing Functions
# ----------------------------
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page \d+', '', text)
    return text.strip()

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def create_embeddings(chunks, batch_size=5, retry_attempts=3, delay=2):
    """Create embeddings in batches to avoid DeadlineExceeded"""
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        for attempt in range(retry_attempts):
            try:
                response = genai.embed_content(model="models/text-embedding-004", content=batch)
                all_embeddings.extend(response['embedding'])
                break
            except Exception as e:
                if attempt < retry_attempts - 1:
                    time.sleep(delay * (2 ** attempt))
                    continue
                else:
                    st.error(f"Embedding failed for batch {i//batch_size + 1}: {e}")
                    raise e
    return np.array(all_embeddings, dtype="float32")

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_chunks(query, index, chunks, top_k=3):
    q_embed = genai.embed_content(model="models/text-embedding-004", content=query)['embedding']
    q_embed = np.array(q_embed, dtype="float32").reshape(1, -1)
    distances, indices = index.search(q_embed, top_k)
    return [chunks[i] for i in indices[0]]

def ask_gemini(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""
    You are a helpful assistant. Answer the question based ONLY on the book content below.

    Context:
    {context}

    Question: {query}
    """
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="📘 RAG Chatbot", layout="wide")
st.title("📘 RAG Chatbot using Gemini + FAISS")

# ----------------------------
# Load PDF and build knowledge base in backend
# ----------------------------
@st.cache_resource
def load_knowledge_base(pdf_path="Software-Engineering-9th-Edition-by-Ian-Sommerville.pdf"):
    raw_text = extract_pdf_text(pdf_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)
    return chunks, index

# Preload PDF from repo (change path if needed)
chunks, index = load_knowledge_base()

# ----------------------------
# Chat Interface Only
# ----------------------------
query = st.text_input("💬 Ask a question from the PDF:")

if query:
    with st.spinner("Thinking..."):
        results = search_chunks(query, index, chunks, top_k=3)
        answer = ask_gemini(query, results)

    st.subheader("🤖 Answer:")
    st.write(answer)

    with st.expander("📚 Retrieved Context"):
        for i, chunk in enumerate(results, 1):
            st.markdown(f"**Chunk {i}:** {chunk}")

