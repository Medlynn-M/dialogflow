from flask import Flask, request, jsonify
from utils import chunk_pdf, chunk_csv, chunk_notebook, find_best_chunk, ask_gemini
import traceback
import os

app = Flask(__name__)

# -------------------- Load Files --------------------

# Load and chunk files at server start
pdf_chunks = chunk_pdf('report.pdf')
csv_chunks = chunk_csv('dataset.csv')
notebook_chunks = chunk_notebook('code.ipynb')

# Combine all chunks
all_chunks = pdf_chunks + csv_chunks + notebook_chunks

# -------------------- Webhook Endpoint --------------------

@app.route('/', methods=['GET'])
def home():
    return 'Service is running!', 200

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        req = request.get_json()
        user_question = req.get('queryResult').get('queryText')

        # Find best matching chunk
        best_chunk = find_best_chunk(all_chunks, user_question)

        if not best_chunk:
            gemini_answer = "I don’t have enough information to answer that."
        else:
            gemini_answer = ask_gemini(user_question, best_chunk)

        return jsonify({'fulfillmentText': gemini_answer})

    except Exception as e:
        print(f"Webhook Error: {e}")
        traceback.print_exc()
        return jsonify({'fulfillmentText': 'An internal error occurred while processing your request.'})

# -------------------- Run App --------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
