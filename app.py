import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader

# Load the Hugging Face pipeline
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="bert-base-uncased")

model = load_model()

# Helper function to read a PDF
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Input Prompts
input_prompt1 = (
    """
 You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
  Please share your professional evaluation on whether the candidate's profile aligns with the role. 
 Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""
)

input_prompt3 = (
    """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
"""
)

# Streamlit Application
st.title("AI Resume and Job Description Matcher")
st.header("Upload your Resume and Job Description to Get Started!")

# File uploader for resume
resume_file = st.file_uploader("Upload your Resume (PDF format only)", type=["pdf"])
job_description = st.text_area("Paste the Job Description Below:")

submit1 = st.button("Analyze Resume")
submit3 = st.button("Get Match Percentage")

if submit1 and resume_file is not None:
    try:
        # Read the uploaded resume
        resume_text = read_pdf(resume_file)
        if not resume_text.strip():
            st.error("The uploaded PDF contains no text. Please check the file.")
        else:
            st.success("Resume successfully uploaded and read!")
            
            # HR manager analysis logic
            analysis = f"{input_prompt1}\n\nResume:\n{resume_text}\n\nJob Description:\n{job_description}"
            st.subheader("HR Manager Analysis")
            st.text_area("Analysis Output:", value=analysis, height=300)
    
    except Exception as e:
        st.error(f"Error processing the resume file: {e}")

if submit3 and resume_file is not None:
    try:
        # Read the uploaded resume
        resume_text = read_pdf(resume_file)
        if not resume_text.strip():
            st.error("The uploaded PDF contains no text. Please check the file.")
        else:
            st.success("Resume successfully uploaded and read!")
            
            # ATS matching logic
            analysis = f"{input_prompt3}\n\nResume:\n{resume_text}\n\nJob Description:\n{job_description}"
            match_result = model(analysis[:512])  # Model compatibility limit
            st.subheader("Match Percentage and Missing Keywords")
            st.text(f"Match Score: {match_result[0]['score'] * 100:.2f}%")
            st.text(f"Predicted Label: {match_result[0]['label']}")
    
    except Exception as e:
        st.error(f"Error analyzing match percentage: {e}")

