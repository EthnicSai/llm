import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

# Initialize the client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Set up Streamlit page
st.set_page_config(page_title="NVIDIA LLM Chat", layout="wide")
st.title("💬 Chat with NVIDIA Nemotron (LLama 3.3 Super 49B)")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "detailed thinking on"}
    ]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Type your message here...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Placeholder for assistant message
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        # Streamed response
        stream = client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            messages=st.session_state.messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
            frequency_penalty=0,
            presence_penalty=0,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                response_placeholder.markdown(full_response + "▌")

        # Final response render
        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
