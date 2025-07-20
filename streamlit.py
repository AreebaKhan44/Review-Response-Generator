import streamlit as st
import pandas as pd
from textblob import TextBlob
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# === Sentiment classifier ===
def classify_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

# === Prompt generator ===
def generate_prompt(review, sentiment):
    if sentiment == "positive":
        return f"Write a warm, personalized thank you response to this review:\n\n'{review}'"
    elif sentiment == "neutral":
        return f"Write a professional response acknowledging this review and encouraging feedback:\n\n'{review}'"
    else:
        return f"Write a professional apology and resolution response to this negative review:\n\n'{review}'"

# === Streamlit UI ===
st.set_page_config(page_title="Review Response Generator", layout="centered")
st.title("📝 AI Review Response Generator")
st.markdown("Automatically generate professional responses for your reviews using GPT-4o.")

input_method = st.radio("Choose input method", ["Manual Input", "Upload CSV"])

def generate_response(review):
    sentiment = classify_sentiment(review)
    prompt = generate_prompt(review, sentiment)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    return sentiment, response.choices[0].message.content

if input_method == "Manual Input":
    review_text = st.text_area("Enter review text:")
    if st.button("Generate Response"):
        with st.spinner("Generating..."):
            sentiment, reply = generate_response(review_text)
            st.markdown(f"**Sentiment:** {sentiment.capitalize()}")
            st.text_area("Generated Response", value=reply, height=200)
else:
    uploaded_file = st.file_uploader("Upload CSV with 'review' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "review" not in df.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            responses = []
            with st.spinner("Processing reviews..."):
                for review in df["review"]:
                    sentiment, reply = generate_response(review)
                    responses.append({
                        "Review": review,
                        "Sentiment": sentiment,
                        "Response": reply
                    })
            result_df = pd.DataFrame(responses)
            st.dataframe(result_df)
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Responses as CSV", csv, "review_responses.csv", "text/csv")
