import streamlit as st
import fitz  # PyMuPDF
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Title
st.title("📄 BrochureBot")
st.write("Upload brochures and ask questions. Powered by OpenAI.")

# API Key
openai_api_key = st.secrets.get("OPENAI_API_KEY") or st.text_input("Enter your OpenAI API Key:", type="password")

# File uploader
uploaded_file = st.file_uploader("Upload a brochure (PDF)", type=["pdf"])

# Ask a question
question = st.text_input("Ask a question about the brochure")

if uploaded_file and openai_api_key:
    # Extract text from PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    # Split and embed text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_texts(chunks, embedding=embeddings)

    # Build Q&A system
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    # Answer question
    if question:
        with st.spinner("Thinking..."):
            answer = qa.run(question)
            st.success(answer)