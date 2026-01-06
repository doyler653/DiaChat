import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

st.set_page_config(page_title="DiaChat Adapter", layout="wide")

# --- Model Setup ---

# Base model you want to apply your adapter to
BASE_MODEL = "decapoda-research/llama-7b-hf"
ADAPTER_REPO = "Doyler653/DiaChatLLM"

@st.cache_resource
def load_model():
    # Load tokenizer from adapter repo
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,  # optional for speed & VRAM
        device_map="auto"            # automatically places on GPU if available
    )

    # Apply adapter
    model = PeftModel.from_pretrained(model, ADAPTER_REPO)

    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- Streamlit UI ---

st.title("DiaChat Adapter Demo")
st.write("Ask anything and the model will respond using the adapter!")

user_input = st.text_input("Your message:")

if st.button("Generate"):
    if not user_input.strip():
        st.warning("Please enter a message first!")
    else:
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                top_p=0.95,
                temperature=0.8
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.text_area("Model response:", value=response, height=200)
