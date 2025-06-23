from flask import Flask, request, jsonify
import pandas as pd
import PyPDF2
import nbformat

app = Flask(__name__)

# Load CSV file
csv_df = pd.read_csv('dataset.csv')  # Your file

# Load PDF file
def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

pdf_text = read_pdf('report.pdf')  # Your file

# Load Jupyter Notebook
def read_notebook(file_path):
    with open(file_path) as f:
        nb = nbformat.read(f, as_version=4)
    cells = []
    for cell in nb.cells:
        if cell.cell_type in ['markdown', 'code']:
            cells.append(cell.source)
    return "\n\n".join(cells[:5])  # Only showing first 5 cells

notebook_summary = read_notebook('code.ipynb')  # Your file

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    query = req.get('queryResult').get('queryText').lower()

    if 'pdf' in query:
        response_text = pdf_text[:1000]  # Limit response size for Dialogflow
    elif 'csv' in query or 'dataset' in query:
        response_text = f"The CSV file contains {csv_df.shape[0]} rows and {csv_df.shape[1]} columns."
    elif 'kaggle' in query:
        response_text = "Here is the Kaggle dataset link: [Paste your Kaggle link here]"
    elif 'notebook' in query:
        response_text = f"Here’s a quick summary from the notebook: {notebook_summary}"
    else:
        response_text = "Sorry, I didn’t understand that."

    return jsonify({'fulfillmentText': response_text})

if __name__ == '__main__':
    app.run(port=5000)
