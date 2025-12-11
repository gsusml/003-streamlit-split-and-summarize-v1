import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from io import StringIO


# LLM and key loading function
def load_LLM(api_key):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0
    )
    return llm


# Page title and header
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

# Intro: instructions
col1, col2 = st.columns(2)

with col1:
    st.markdown("ChatGPT cannot summarize long texts. Now you can do it with this app.")

with col2:
    st.write("Contact with [G5U5](https://gsusml84.es)")

# Input OpenAI API Key
st.markdown("## Enter Your Groq API Key")


def get_api_key():
    input_text = st.text_input(label="Groq API Key ", placeholder="Ex: gq-2twmA8tfCb8un4...",
                               key="api_key_input", type="password")
    return input_text


api_key = get_api_key()

# Input
st.markdown("## Upload the PDF file you want to summarize")

uploaded_file = st.file_uploader("Choose a file", type="pdf")

# Output
st.markdown("### Here is your Summary:")

if uploaded_file is not None:

    # PDF
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        st.stop()

    if not text.strip():
        st.error("Could not extract text from this PDF. It may be scanned or image-based.")
        st.stop()

    if len(text.split()) > 200000:
        st.warning("Please upload a shorter PDF (max 200000 words).")
        st.stop()

    if not api_key:
        st.warning("Please insert your Groq API Key.")
        st.stop()

    # Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=5000,
        chunk_overlap=350
    )

    splitted_documents = text_splitter.create_documents([text])

    # LLM summary
    llm = load_LLM(api_key)

    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce"
    )

    summary_output = summarize_chain.run(splitted_documents)

    st.write(summary_output)
