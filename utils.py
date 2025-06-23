import PyPDF2
import pandas as pd
import nbformat
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Gemini API key setup
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# -------------------- Chunking Functions --------------------

def chunk_pdf(file_path, chunk_size=500):
    chunks = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Avoid None pages
                text += page_text

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

def chunk_csv(file_path):
    df = pd.read_csv(file_path)
    chunks = []
    for index, row in df.iterrows():
        chunks.append(str(row.to_dict()))
    return chunks

def chunk_notebook(file_path):
    chunks = []
    with open(file_path) as f:
        nb = nbformat.read(f, as_version=4)
        for cell in nb.cells:
            if cell.cell_type in ['markdown', 'code']:
                chunks.append(cell.source)
    return chunks

# -------------------- Embedding and Retrieval --------------------

def get_embedding(text):
    model = genai.GenerativeModel('embedding-001')
    try:
        response = model.embed_content([{"text": text}], task_type="retrieval_document")
        return response['embedding']
    except Exception as e:
        print(f"Embedding Error: {e}")
        return np.zeros((768,))  # Safe fallback

def find_best_chunk(chunks, user_question):
    question_embedding = get_embedding(user_question)
    max_score = -1
    best_chunk = ""

    for chunk in chunks:
        chunk_embedding = get_embedding(chunk)
        score = cosine_similarity([question_embedding], [chunk_embedding])[0][0]

        if score > max_score:
            max_score = score
            best_chunk = chunk

    return best_chunk

# -------------------- Gemini Conversational Answer --------------------

def ask_gemini(user_question, context_chunk):
    prompt = f"""
You are a helpful AI assistant.

You will receive a context from a document. Answer the user's question in a friendly, conversational tone.

Instructions:
- Summarize when needed.
- Provide step-by-step explanations.
- Compare items where possible.
- Create tables or bullet points when it improves clarity.
- NEVER copy or repeat the context directly.
- If you don’t have enough information, say: "I don't have enough information to answer that."

Context:
{context_chunk}

Question:
{user_question}

Answer:
"""

    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content(prompt)
        if response.text:
            return response.text.strip()
        else:
            return "I don’t have enough information to answer that."
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "I don’t have enough information to answer that."
