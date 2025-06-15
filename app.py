from flask import Flask, request, jsonify
import google.generativeai as genai
import os

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load Gemini 2.0 Flash model
model = genai.GenerativeModel("models/gemini-2.0-flash")

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        req = request.get_json()
        user_input = req["queryResult"]["queryText"]

        response = model.generate_content(user_input)
        bot_reply = response.text

        return jsonify({
            "fulfillmentText": bot_reply
        })
    
    except Exception as e:
        return jsonify({
            "fulfillmentText": f"⚠️ Gemini error: {str(e)}"
        })

# Correct Render binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides this
    app.run(host="0.0.0.0", port=port)
