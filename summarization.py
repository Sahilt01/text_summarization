# -*- coding: utf-8 -*-
import streamlit as st
import pdfplumber
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from heapq import nlargest
from transformers import pipeline
import re
import os

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------
# Text Extraction
# -------------------------------
def extract_text(file_path, file_type):
    text = ""
    if file_type == "pdf":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif file_type == "docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_text(text):
    sentences = sent_tokenize(text)
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in word_tokenize(text)
        if word.isalnum() and word.lower() not in stop_words
    ]
    return sentences, words

# -------------------------------
# Frequency-based Extractive Summary
# -------------------------------
def calculate_word_freq(words):
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    max_freq = max(word_freq.values(), default=1)
    for word in word_freq:
        word_freq[word] /= max_freq
    return word_freq

def calculate_sentence_scores(sentences, word_freq):
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
        sentence_scores[sentence] /= max(len(word_tokenize(sentence)), 1)  # normalize by length
    return sentence_scores

def generate_extractive_summary(text, num_sentences):
    sentences, words = preprocess_text(text)
    word_freq = calculate_word_freq(words)
    sentence_scores = calculate_sentence_scores(sentences, word_freq)
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return " ".join(summary_sentences)

# -------------------------------
# Abstractive Summary (BART)
# -------------------------------
@st.cache_resource
def load_abstractive_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def generate_abstractive_summary(text, max_length=130, min_length=30):
    summarizer = load_abstractive_model()
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]

# -------------------------------
# Keyword Extraction
# -------------------------------
def extract_keywords(text, top_n=10):
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in word_tokenize(text)
        if word.isalnum() and word.lower() not in stop_words
    ]
    freq = calculate_word_freq(words)
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_keywords[:top_n]]

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.title("AI-Powered Document Summarizer 📝")
    st.write("Supports **Extractive** & **Abstractive** summarization with keyword extraction.")

    uploaded_file = st.file_uploader("Upload your file", type=["pdf", "txt", "docx"])
    mode = st.selectbox("Select summarization mode", ["Extractive", "Abstractive"])
    num_sentences = st.slider("Number of sentences (Extractive only)", 1, 15, 5)

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1]
        temp_path = f"temp.{file_type}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        text = extract_text(temp_path, file_type)

        if mode == "Extractive":
            summary = generate_extractive_summary(text, num_sentences)
        else:
            summary = generate_abstractive_summary(text)

        st.subheader("Summary")
        st.write(summary)

        st.subheader("Keywords")
        st.write(", ".join(extract_keywords(text)))

        st.download_button("Download Summary", summary, file_name="summary.txt")

        os.remove(temp_path)

if __name__ == "__main__":
    main()
