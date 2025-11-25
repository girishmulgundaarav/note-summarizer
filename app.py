import streamlit as st
from dotenv import load_dotenv
import os
import PyPDF2

from summarizer import llm_summarize
from qa import build_llm_retriever
from image_extractor import extract_images_and_text

load_dotenv()

st.set_page_config(page_title="AI-Powered Note Summarizer", page_icon="🧠", layout="wide")

st.title("🧠 AI-Powered Note Summarizer with Image Support")

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
        uploaded.seek(0)  # reset pointer for image extractor
        image_texts = extract_images_and_text(uploaded)

    # Spinner while processing
    with st.spinner("⏳ Processing notes and generating summary..."):
        st.session_state.summary = llm_summarize(text, image_texts)
        st.session_state.qa_chain = build_llm_retriever(text, image_texts)

    st.success("✅ Notes processed successfully!")

if st.session_state.summary:
    st.subheader("Summary")
    st.write(st.session_state.summary)

st.subheader("Ask Questions")
q = st.text_input("Your question")
ask = st.button("Get Answer", type="primary")

if ask and st.session_state.qa_chain and not st.session_state.cancel:
    with st.spinner("🤖 Thinking..."):
        ans = st.session_state.qa_chain(q)
    st.markdown("**Answer:**")
    st.info(ans)