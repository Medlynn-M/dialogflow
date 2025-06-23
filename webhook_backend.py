from flask import Flask, request, jsonify
import os
from utils import chunk_pdf, chunk_csv, chunk_notebook, find_best_chunk, ask_gemini

app = Flask(__name__)

# Load and chunk files
pdf_chunks = chunk_pdf('report.pdf')
csv_chunks = chunk_csv('dataset.csv')
notebook_chunks = chunk_notebook('code.ipynb')
all_chunks = pdf_chunks + csv_chunks + notebook_chunks

@app.route('/', methods=['GET'])
def home():
    return 'Service is running!', 200

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    user_question = req.get('queryResult').get('queryText')

    best_chunk = find_best_chunk(all_chunks, user_question)
    gemini_answer = ask_gemini(user_question, best_chunk)

    return jsonify({'fulfillmentText': gemini_answer})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
