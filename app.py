
import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Groq AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Purple background UI
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e0b3a, #4b1680, #241044);
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #17072b;
}

.stChatMessage {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
}

h1, h2, h3, p, label {
    color: white !important;
}

.stChatInput textarea {
    color: white;
}
</style>
""", unsafe_allow_html=True)


# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Check your .env file.")
    st.stop()

client = Groq(api_key=api_key)

MODEL = "qwen/qwen3.8-27b"


# Sidebar settings
st.sidebar.title("⚙️ Model Settings")

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=0.7,
    step=0.1
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=256,
    max_value=8192,
    value=2048,
    step=256
)

top_p = st.sidebar.slider(
    "Top P",
    min_value=0.0,
    max_value=1.0,
    value=0.9,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: {MODEL}")

# Clear chat
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


# Main UI
st.title("🤖 Groq AI Chatbot")
st.caption("Powered by Groq API | Qwen 3.8 27B")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
if prompt := st.chat_input("Ask me anything..."):

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=st.session_state.messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                reasoning_effort="none",
                reasoning_format="hidden",
                stream=True
            )

            full_response = ""

            for chunk in response:
                content = chunk.choices[0].delta.content

                if content:
                    full_response += content

                    response_placeholder.markdown(
                        full_response + "▌"
                    )

            response_placeholder.markdown(full_response)

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:
            st.error(f"Groq API Error: {str(e)}")

            # Remove failed user message
            st.session_state.messages.pop()
