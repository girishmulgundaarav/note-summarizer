import streamlit as st
from dotenv import load_dotenv
import os
import PyPDF2

from summarizer import llm_summarize_stream
from qa import build_llm_retriever_stream
from image_extractor import extract_images_and_text

st.set_page_config(page_title="AI-Powered Note Summarizer", page_icon="🧠", layout="wide")
load_dotenv()

st.title("🧠 AI-Powered Note Summarizer with Image Support")

model_choice = st.selectbox(
    "Choose the model:",
    options=["gpt-5-mini", "gpt-5.1", "gpt-4o-mini"],
    index=0
)

if "summary" not in st.session_state:
    st.session_state.summary = ""
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "cancel" not in st.session_state:
    st.session_state.cancel = False

uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

col1, col2 = st.columns([1,1])
with col1:
    process = st.button("Process Notes", type="primary")
with col2:
    cancel = st.button("Cancel", type="secondary")

if cancel:
    st.session_state.cancel = True
    st.warning("⛔ Processing cancelled by user.")

if process and uploaded and not st.session_state.cancel:
    ext = uploaded.name.split(".")[-1].lower()
    if ext == "txt":
        text = uploaded.getvalue().decode("utf-8", errors="ignore")
        image_texts = []
    else:
        reader = PyPDF2.PdfReader(uploaded)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        uploaded.seek(0)
        image_texts = extract_images_and_text(uploaded)

    with st.spinner(f"⏳ Processing notes with {model_choice}..."):
        stream_fn = llm_summarize_stream(text, image_texts, model=model_choice)
        placeholder = st.empty()
        summary_text = ""
        for chunk in stream_fn():
            summary_text += chunk
            placeholder.markdown(summary_text)
        st.session_state.summary = summary_text
        st.session_state.qa_chain = build_llm_retriever_stream(text, image_texts, model=model_choice)

    st.success(f"✅ Notes processed successfully with {model_choice}!")

if st.session_state.summary:
    st.subheader(f"Summary (generated with {model_choice})")
    st.write(st.session_state.summary)

st.subheader("Ask Questions")
q = st.text_input("Your question")
ask = st.button("Get Answer", type="primary")

if ask and st.session_state.qa_chain and not st.session_state.cancel:
    with st.spinner(f"🤖 {model_choice} is thinking..."):
        placeholder = st.empty()
        answer_text = ""
        for chunk in st.session_state.qa_chain(q):
            answer_text += chunk
            placeholder.markdown(answer_text)