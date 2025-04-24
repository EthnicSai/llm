import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

# Initialize OpenAI client for NVIDIA API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Helper to optionally strip <think> tags
def strip_think_tags(text, show_think=False):
    if show_think:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

# Page configuration
st.set_page_config(page_title="NVIDIA LLM Chat", layout="wide")
st.title("💬 Chat with DeepSeek (r1-distill-llama-8b)")

# Sidebar toggle to show/hide <think> content
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    show_think = st.checkbox("Show 'thinking' content", value=False)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "<think>detailed thinking on</think>"}
    ]

# Render previous messages (excluding 'system' unless it has something relevant)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(strip_think_tags(msg["content"], show_think))

# Input from user
user_input = st.chat_input("Type your message here...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Placeholder for assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        # Streaming response from NVIDIA API
        stream = client.chat.completions.create(
            model="deepseek-ai/deepseek-r1-distill-llama-8b",
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
                # Show filtered response while streaming
                response_placeholder.markdown(strip_think_tags(full_response, show_think) + "▌")

        # Final render without cursor
        response_placeholder.markdown(strip_think_tags(full_response, show_think))

        # Save assistant message with original content (think tags included)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
