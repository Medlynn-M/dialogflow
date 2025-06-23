from flask import Flask, request, jsonify
from utils import chunk_pdf, chunk_csv, chunk_notebook, ask_gemini
from faiss_utils import build_faiss_index, search_faiss_index
import traceback
import os

app = Flask(__name__)

# Load chunks
pdf_chunks = chunk_pdf('report.pdf')
csv_chunks = chunk_csv('dataset.csv')
notebook_chunks = chunk_notebook('code.ipynb')

all_chunks = pdf_chunks + csv_chunks + notebook_chunks

# Build FAISS index
faiss_index, chunk_list, embeddings_np = build_faiss_index(all_chunks)

@app.route('/', methods=['GET'])
def home():
    return 'Service is running!', 200

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        req = request.get_json()
        user_question = req.get('queryResult').get('queryText')

        # Use FAISS to find best chunk
        best_chunk = search_faiss_index(faiss_index, chunk_list, embeddings_np, user_question)

        if not best_chunk:
            gemini_answer = "I don’t have enough information to answer that."
        else:
            gemini_answer = ask_gemini(user_question, best_chunk)

        return jsonify({'fulfillmentText': gemini_answer})

    except Exception as e:
        print(f"Webhook Error: {e}")
        traceback.print_exc()
        return jsonify({'fulfillmentText': 'An internal error occurred while processing your request.'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
