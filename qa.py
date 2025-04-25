import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import shutil

# 1. Extract text from the PDF document using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        full_text += page.get_text("text")  # Extract text from each page

    return full_text

# Load .env file
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

# 2. Initialize the OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# 3. Define the function to perform the document question answering
def document_qa(doc_text, question):
    # Create the prompt
    prompt = f"""
    You are a document question answering assistant.
    Here is the document content:
    {doc_text}

    Answer this question based on the document:
    {question}
    """

    # Call the API without streaming
    completion = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        stream=False  # Disable streaming
    )

    # Get the final response content
    response = completion.choices[0].message.content
    return response

# 4. Streamlit app setup
def main():
    # Streamlit page configuration
    st.set_page_config(page_title="Document QA App", layout="wide")

    # Title of the web app
    st.title("Document Question Answering with NVIDIA Llama-3 by AES")

    # Ensure the "documents" directory exists
    if not os.path.exists("documents"):
        os.makedirs("documents")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Save uploaded file to 'documents' folder
    if uploaded_file is not None:
        # Define file path
        file_path = os.path.join("documents", uploaded_file.name)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Show success message
        st.success(f"File uploaded successfully: {uploaded_file.name}")

        # Extract text from the uploaded PDF
        st.write("Extracting text from the document...")
        with st.spinner("Extracting..."):
            doc_text = extract_text_from_pdf(file_path)

        # Display extracted text (first 1000 characters as a preview)
        st.text_area("Extracted Document Snippet", doc_text[:1000], height=200)

        # Provide an option to view the full document text
        if st.button("View Full Document Text"):
            st.text_area("Full Document Text", doc_text, height=300)

        # Question input
        question = st.text_input("Enter your question about the document:")

        if st.button("Get Answer"):
            if question:
                st.write("Answering the question...")
                with st.spinner("Processing..."):
                    answer = document_qa(doc_text, question)
                    st.subheader("Answer:")
                    st.write(answer)
            else:
                st.warning("Please enter a question.")

        # Button to clear the uploaded file
        if st.button("Clear Uploaded File"):
            # Delete the uploaded file from 'documents' folder
            if os.path.exists(file_path):
                os.remove(file_path)
                st.success("Uploaded file cleared successfully.")
            else:
                st.warning("No file uploaded yet.")

# 5. Run the Streamlit app
if __name__ == "__main__":
    main()
