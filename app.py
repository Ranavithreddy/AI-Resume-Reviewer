import os
from dotenv import load_dotenv
import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Page Configuration (must be the first Streamlit command)
st.set_page_config(page_title="AI Resume Reviewer")

# Load environment variables
if os.path.exists(".env"):
    load_dotenv()
else:
    st.warning(".env file not found. Please ensure environment variables are set in the cloud environment.")

# Load Hugging Face API Key
hf_api_key = os.getenv("HUGGING_FACE_API_KEY")
if not hf_api_key:
    st.error("HUGGING_FACE_API_KEY not found. Please set it in the environment variables.")

# Hugging Face Pipeline for Text Analysis
def analyze_text_with_hf(input_text, prompt):
    try:
        generator = pipeline("text-generation", model="gpt2", use_auth_token=hf_api_key)
        result = generator(prompt + input_text, max_length=150, num_return_sequences=1)
        return result[0]["generated_text"]
    except Exception as e:
        return f"Error: {str(e)}"

# Function to extract text from the first page of a PDF
def extract_first_page_text(uploaded_file):
    try:
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        first_page = pdf_document[0]
        text = first_page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Streamlit UI
st.header("AI Resume Reviewer")

input_text = st.text_area("Job Description:")

uploaded_file = st.file_uploader("Upload your resume (PDF):", type=["pdf"])

if uploaded_file is not None:
    st.write("PDF uploaded successfully.")

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

if submit1:
    if uploaded_file is not None:
        try:
            pdf_content = extract_first_page_text(uploaded_file)
            response = analyze_text_with_hf(pdf_content, input_prompt1)
            st.subheader("Analysis Result")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please upload a resume.")

if submit3:
    if uploaded_file is not None:
        try:
            pdf_content = extract_first_page_text(uploaded_file)
            response = analyze_text_with_hf(pdf_content, input_prompt3)
            st.subheader("Match Percentage")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please upload a resume.")
