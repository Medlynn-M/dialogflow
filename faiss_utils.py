import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    try:
        embedding = embedding_model.encode(text)
        return np.array(embedding, dtype='float32')
    except Exception as e:
        print(f"Embedding Error: {e}")
        return np.zeros((384,), dtype='float32')  # 384 is embedding size for MiniLM

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
