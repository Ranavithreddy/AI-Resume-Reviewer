from dotenv import load_dotenv
import streamlit as st
import os
from transformers import pipeline

# Load environment variables
load_dotenv()

# Streamlit app setup
st.set_page_config(page_title="Resume Reviewer")
st.title("AI Resume Reviewer")

# Hugging Face API key
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
if not HUGGING_FACE_API_KEY:
    st.error("HUGGING_FACE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

# Load the Hugging Face model
st.write("Loading Hugging Face model...")
try:
    generator = pipeline(
        "text-generation",
        model="gpt2",
        tokenizer="gpt2",
        trust_remote_code=True,
        use_auth_token=HUGGING_FACE_API_KEY
    )
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Prompts
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

# Inputs
job_description = st.text_area("Enter the job description:")
uploaded_file = st.file_uploader("Upload your resume (Text format only):", type=["txt"])
analysis_type = st.selectbox(
    "Select the type of analysis:",
    ("HR Manager Review", "ATS Scanner Evaluation"),
)

if st.button("Analyze Resume"):
    if not job_description or not uploaded_file:
        st.error("Please provide a job description and upload a resume.")
    else:
        # Read resume content
        resume_content = uploaded_file.read().decode("utf-8")
        prompt = input_prompt1 if analysis_type == "HR Manager Review" else input_prompt3

        # Generate analysis
        st.write("Analyzing resume...")
        try:
            input_text = f"{prompt}\n\nJob Description: {job_description}\nResume: {resume_content}\nAnalysis:"
            result = generator(input_text, max_length=500, num_return_sequences=1)
            st.subheader("Analysis Result")
            st.write(result[0]["generated_text"])
        except Exception as e:
            st.error(f"Error during analysis: {e}")
