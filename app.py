import streamlit as st
import os
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# Check if the API key exists
if not HUGGING_FACE_API_KEY:
    st.error("HUGGING_FACE_API_KEY not found. Please set it in the environment variables.")
    st.stop()

# Set up the model pipeline
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2", use_auth_token=HUGGING_FACE_API_KEY)

model = load_model()

# Streamlit app interface
st.title("AI Resume Reviewer")

# Input for job description
job_description = st.text_area("Enter the Job Description:")

# Upload resume PDF
uploaded_file = st.file_uploader("Upload your resume (PDF):", type=["pdf"])

if uploaded_file:
    st.success("Resume uploaded successfully!")

# Submit button
if st.button("Analyze Resume"):
    if not uploaded_file:
        st.error("Please upload your resume before submitting.")
    elif not job_description:
        st.error("Please enter a job description before submitting.")
    else:
        try:
            # Generate response using the model
            response = model(f"Analyze the following resume against the job description: {job_description}")
            st.subheader("Analysis Result")
            st.write(response[0]['generated_text'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
