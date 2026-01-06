import os
import zipfile
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="HF Chatbot", layout="centered")

MODEL_ZIP = "model.zip"
MODEL_DIR = "model"

# -----------------------------
# Extract model if needed
# -----------------------------
if not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# -----------------------------
# Load model (cached)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32
)
model.eval()

# -----------------------------
# UI
# -----------------------------
st.title("🤖 Chatbot App")
st.caption("Powered by your Hugging Face model")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Build conversation prompt
    prompt = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt += f"{role}: {msg['content']}\n"
    prompt += "Assistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)
