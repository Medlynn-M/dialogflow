from flask import Flask, request, jsonify
import os
import google.generativeai as genai

# Load Gemini API key from env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.0-flash")

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    req = request.get_json()
    user_query = req["queryResult"]["queryText"]

    try:
        response = model.generate_content(user_query)
        reply = response.text
    except Exception as e:
        reply = f"Error from Gemini: {str(e)}"

    return jsonify({
        "fulfillmentText": reply
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
