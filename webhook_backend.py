from flask import Flask, request, jsonify
import pandas as pd
import PyPDF2
import nbformat
import google.generativeai as genai
import os

# 🔑 Gemini API Setup (using environment variable)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return 'Service is running!', 200

# ✅ Load CSV file
csv_df = pd.read_csv('dataset.csv')
csv_summary = f"The dataset contains {csv_df.shape[0]} rows and {csv_df.shape[1]} columns."

# ✅ Load PDF file
def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

pdf_text = read_pdf('report.pdf')

# ✅ Load Jupyter Notebook
def read_notebook(file_path):
    with open(file_path) as f:
        nb = nbformat.read(f, as_version=4)
    cells = []
    for cell in nb.cells:
        if cell.cell_type in ['markdown', 'code']:
            cells.append(cell.source)
    return "\n\n".join(cells)

notebook_text = read_notebook('code.ipynb')

# ✅ Combined Knowledge Base (Limiting to avoid memory overload)
knowledge_base = f"""
You are an expert AI assistant. Answer based ONLY on the following knowledge base.

📄 PDF Content: {pdf_text[:3000]}

📊 Dataset Summary: {csv_summary}

📓 Notebook Content: {notebook_text[:3000]}
"""

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    user_query = req.get('queryResult').get('queryText')

    try:
        # Gemini API Call
        gemini_response = model.generate_content([
            knowledge_base,
            f"User question: {user_query}"
        ])
        reply = gemini_response.text
    except Exception as e:
        reply = "Sorry, I encountered an error while processing your request."

    return jsonify({'fulfillmentText': reply})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
