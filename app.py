pip install pdfplumber

import streamlit as st
import pdfplumber
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc 
              if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def rank_resumes(resume_texts, job_desc):
    corpus = [job_desc] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
    return scores

st.title("AI-Powered Resume Screening Tool")

job_desc = st.text_area("Paste Job Description")

uploaded_files = st.file_uploader("Upload Resumes (PDFs)", accept_multiple_files=True, type="pdf")

if st.button("Rank Candidates"):
    if job_desc and uploaded_files:
        job_desc_clean = preprocess(job_desc)
        resume_texts, resume_names = [], []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resume_texts.append(preprocess(text))
            resume_names.append(file.name)

        scores = rank_resumes(resume_texts, job_desc_clean)
        results = pd.DataFrame({"Resume": resume_names, "Score": scores})
        st.dataframe(results.sort_values(by="Score", ascending=False))

