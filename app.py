import os
import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader

# Set up the Hugging Face pipeline
def load_hugging_face_pipeline():
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", use_auth_token=os.getenv("HUGGING_FACE_API_KEY"))
        return summarizer
    except Exception as e:
        st.error(f"Error loading Hugging Face pipeline: {e}")
        return None

# Initialize pipeline
summarizer = load_hugging_face_pipeline()

# Streamlit App Header
st.title("AI Resume Reviewer")
st.write("Upload your resume and job description to get insights and a match percentage!")

# Upload Files
resume_file = st.file_uploader("Upload your resume (PDF or Text format only)", type=["pdf", "txt"])
job_description = st.text_area("Paste the job description here:")

submit1 = st.button("Analyze Resume")
submit3 = st.button("Get Match Percentage")

input_prompt1 = (
    "You are an experienced Technical HR Manager. "
    "Review the provided resume against the job description. "
    "Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements."
)

input_prompt3 = (
    "You are a skilled ATS scanner. "
    "Evaluate the resume against the provided job description. "
    "Provide a match percentage, list missing keywords, and give final thoughts."
)

def read_file(file):
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        return ""

if submit1 and resume_file and job_description:
    st.subheader("Analysis Results")
    resume_text = read_file(resume_file)
    if summarizer:
        try:
            summary = summarizer(f"{input_prompt1}\n\nResume:\n{resume_text}\n\nJob Description:\n{job_description}", max_length=500, min_length=50, do_sample=False)
            st.write(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"Error during analysis: {e}")

if submit3 and resume_file and job_description:
    st.subheader("Match Percentage")
    resume_text = read_file(resume_file)
    if summarizer:
        try:
            match_result = summarizer(f"{input_prompt3}\n\nResume:\n{resume_text}\n\nJob Description:\n{job_description}", max_length=500, min_length=50, do_sample=False)
            st.write(match_result[0]['summary_text'])
        except Exception as e:
            st.error(f"Error during analysis: {e}")
