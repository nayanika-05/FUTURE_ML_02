import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
st.set_page_config(
    page_title="Support Ticket Classification",
    page_icon="🎫",
    layout="centered"
)
category_model = pickle.load(open("notebook/category_model.pkl", "rb"))
priority_model = pickle.load(open("notebook/priority_model.pkl", "rb"))
tfidf = pickle.load(open("notebook/tfidf.pkl", "rb"))

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

st.title("🎫 Support Ticket Classification System")
st.write(
    "This NLP-powered system automatically classifies support tickets and predicts their priority level."
)
ticket = st.text_area("Enter Support Ticket")

if st.button("Predict"):

    cleaned = clean_text(ticket)

    vector = tfidf.transform([cleaned])

    category = category_model.predict(vector)[0]
    priority = priority_model.predict(vector)[0]

    st.success("Prediction Complete!")

    st.write("### Category")
    st.write(category)

    st.write("### Priority")
    st.write(priority)