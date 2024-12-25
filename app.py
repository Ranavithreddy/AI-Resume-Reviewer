import os
from dotenv import load_dotenv
import streamlit as st
from transformers import pipeline
import base64
import io
from PIL import Image
import pdf2image

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

# Function to process uploaded PDF
def input_pdf_setup(uploaded_file):
    # Path to Poppler binaries (Update based on your OS and installation path)
    poppler_path = "/usr/bin"  # Example for Linux/macOS; update if different

    if uploaded_file is not None:
        # Convert PDF to images
        images = pdf2image.convert_from_bytes(uploaded_file.read(), poppler_path=poppler_path)
        first_page = images[0]

        # Convert the first page to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

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
            pdf_content = input_pdf_setup(uploaded_file)
            response = analyze_text_with_hf(input_text, input_prompt1)
            st.subheader("Analysis Result")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please upload a resume.")

if submit3:
    if uploaded_file is not None:
        try:
            pdf_content = input_pdf_setup(uploaded_file)
            response = analyze_text_with_hf(input_text, input_prompt3)
            st.subheader("Match Percentage")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please upload a resume.")
