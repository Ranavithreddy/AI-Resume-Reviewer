import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from PyPDF2 import PdfReader

# Initialize Hugging Face model and tokenizer
model_name = "bert-base-uncased"  # Replace with your model
max_length = 512  # Maximum token length for the model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define prompts
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

# Streamlit UI
st.title("AI Resume Reviewer")
st.write("Upload your resume and provide a job description to analyze.")

# File upload and job description input
resume_file = st.file_uploader("Upload your resume (PDF format only):", type="pdf")
job_description = st.text_area("Paste the job description here:")

# Buttons
submit1 = st.button("Analyze Resume")
submit3 = st.button("Get Match Percentage")

if submit1:
    if resume_file is None or not job_description:
        st.error("Please upload a resume and provide a job description.")
    else:
        try:
            # Extract text from PDF
            reader = PdfReader(resume_file)
            resume_text = " ".join([page.extract_text() for page in reader.pages])

            # Combine and truncate inputs
            resume_tokens = tokenizer.tokenize(resume_text, truncation=True, max_length=max_length // 2)
            job_tokens = tokenizer.tokenize(job_description, truncation=True, max_length=max_length // 2)
            combined_text = tokenizer.convert_tokens_to_string(resume_tokens + job_tokens)

            # Perform analysis
            result = classifier(combined_text)

            st.write("**Analysis Results:**")
            st.json(result)
        except Exception as e:
            st.error(f"Error during resume analysis: {e}")

if submit3:
    if resume_file is None or not job_description:
        st.error("Please upload a resume and provide a job description.")
    else:
        try:
            # Extract text from PDF
            reader = PdfReader(resume_file)
            resume_text = " ".join([page.extract_text() for page in reader.pages])

            # Combine and truncate inputs
            resume_tokens = tokenizer.tokenize(resume_text, truncation=True, max_length=max_length // 2)
            job_tokens = tokenizer.tokenize(job_description, truncation=True, max_length=max_length // 2)
            combined_text = tokenizer.convert_tokens_to_string(resume_tokens + job_tokens)

            # Perform match percentage analysis
            ats_analysis = classifier(combined_text)

            st.write("**Match Percentage Results:**")
            st.json(ats_analysis)
        except Exception as e:
            st.error(f"Error during match percentage analysis: {e}")
