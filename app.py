import streamlit as st
from transformers import pipeline
import os

# Set the page configuration
st.set_page_config(page_title="AI Resume Reviewer", layout="centered")

# Load the Hugging Face API Key from environment variables
api_key = os.getenv("HUGGING_FACE_API_KEY")

# Check if the API key is available
if not api_key:
    st.error("HUGGING_FACE_API_KEY not found. Please set it in the environment variables.")
    st.stop()

# Initialize the Hugging Face pipeline
try:
    model_name = "bert-base-uncased"  # Replace with the desired Hugging Face model
    classifier = pipeline("text-classification", model=model_name)
except Exception as e:
    st.error(f"Error initializing Hugging Face pipeline: {e}")
    st.stop()

# Streamlit UI
st.title("AI Resume Reviewer")
st.write("Upload your resume and job description to get insights and feedback.")

# File upload section
resume_file = st.file_uploader("Upload your Resume (PDF format)", type=["pdf"])
job_description = st.text_area("Paste the Job Description here:")

# Submit buttons
submit1 = st.button("Analyze Resume")
submit3 = st.button("Get Match Percentage")

# Input prompts for analysis
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

# Process uploaded files
if submit1:
    if resume_file is None or not job_description:
        st.error("Please upload a resume and provide a job description.")
    else:
        # Process the resume (example: extract text from PDF)
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(resume_file)
            resume_text = " ".join([page.extract_text() for page in reader.pages])
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            st.stop()

        # Analyze using Hugging Face model
        try:
            analysis = classifier(resume_text + "\n" + job_description)
            st.write("**Analysis Results:**")
            st.json(analysis)
        except Exception as e:
            st.error(f"Error during analysis: {e}")

if submit3:
    if resume_file is None or not job_description:
        st.error("Please upload a resume and provide a job description.")
    else:
        # Process the resume (example: extract text from PDF)
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(resume_file)
            resume_text = " ".join([page.extract_text() for page in reader.pages])
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            st.stop()

        # Perform ATS-like match percentage evaluation
        try:
            ats_analysis = classifier(resume_text + "\n" + job_description)
            st.write("**Match Percentage Results:**")
            st.json(ats_analysis)
        except Exception as e:
            st.error(f"Error during match percentage analysis: {e}")
