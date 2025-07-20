from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from utils import classify_sentiment, generate_prompt
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Flask setup
app = Flask(__name__)
CORS(app)

@app.route('/generate_response', methods=['POST'])
def generate_response():
    data = request.json
    review = data.get("review", "")
    sentiment = classify_sentiment(review)
    prompt = generate_prompt(review, sentiment)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        reply = response.choices[0].message.content
        return jsonify({
            "review": review,
            "sentiment": sentiment,
            "response": reply
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000)
