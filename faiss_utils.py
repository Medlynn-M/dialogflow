import faiss
import numpy as np
from google.generativeai import GenerativeModel

# Load embedding model
embedding_model = GenerativeModel('embedding-001')

def get_embedding(text):
    try:
        response = embedding_model.embed_content([{"text": text}], task_type="retrieval_document")
        return np.array(response['embedding'], dtype='float32')
    except Exception as e:
        print(f"Embedding Error: {e}")
        return np.zeros((768,), dtype='float32')

def build_faiss_index(chunks):
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        embeddings.append(emb)

    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    return index, chunks, embeddings_np

def search_faiss_index(index, chunks, embeddings_np, user_question, top_k=1):
    user_embedding = get_embedding(user_question).reshape(1, -1)
    distances, indices = index.search(user_embedding, top_k)
    best_chunk = chunks[indices[0][0]]
    return best_chunk
