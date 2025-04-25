import fitz  # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st

# 1. Extract text from the PDF document using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        full_text += page.get_text("text")  # Extract text from each page
    return full_text

# 2. Load environment variables
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

# 3. Initialize the OpenAI client for NVIDIA's API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# 4. Define function for Document Question Answering
def document_qa(doc_text, question):
    prompt = f"""
    You are a document question answering assistant.
    Here is the document content:
    {doc_text}

    Answer this question based on the document:
    {question}
    """

    completion = client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        stream=False
    )

    return completion.choices[0].message.content

# 5. Main Streamlit app logic
def main():
    st.set_page_config(page_title="Vedantra QA App", layout="wide")
    st.title("📄 Document Question & Answering by AES")

    # Ensure 'documents' folder exists
    if not os.path.exists("documents"):
        os.makedirs("documents")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")

    # Save and process uploaded file
    if uploaded_file is not None:
        file_path = os.path.join("documents", uploaded_file.name)

        # Save file to disk if not already saved
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")

        # Extract and store document text in session_state
        if "doc_text" not in st.session_state:
            with st.spinner("Extracting text from the document..."):
                st.session_state.doc_text = extract_text_from_pdf(file_path)

    # If document text is available, show Q&A interface
    if "doc_text" in st.session_state:
        # Optionally show full document text
        with st.expander("🔍 View Document Text (optional)", expanded=False):
            st.text_area("Full Document Text", st.session_state.doc_text, height=300)

        # Ask question
        question = st.text_input("🤖 Ask a question about the document:")
        if st.button("Get Answer"):
            if question.strip():
                st.write("Answering your question...")
                with st.spinner("Processing..."):
                    answer = document_qa(st.session_state.doc_text, question)
                    st.subheader("💬 Answer:")
                    st.write(answer)
            else:
                st.warning("⚠️ Please enter a valid question.")

        # Clear file and session state
        if st.button("🗑️ Clear Uploaded File and Data"):
            if os.path.exists(file_path):
                os.remove(file_path)
            st.session_state.pop("doc_text", None)
            st.success("All data cleared. You can upload a new document.")

    else:
        st.info("📥 Please upload a PDF document to begin.")

# Run the app
if __name__ == "__main__":
    main()
