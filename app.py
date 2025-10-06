import os
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template

# -----------------------------
# CONFIGURATION
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"  # Small & CPU-friendly

# -----------------------------
# LOCAL MODEL LOADERS
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_local_models():
    """Load both embeddings and local text-generation pipeline."""
    st.sidebar.info(f"Using Embeddings: {EMBEDDING_MODEL.split('/')[-1]}")
    st.sidebar.info(f"Using Local Model: {LLM_MODEL.split('/')[-1]}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    llm_pipeline = pipeline("text2text-generation", model=LLM_MODEL)
    return embeddings, llm_pipeline

# -----------------------------
# PDF HELPERS
# -----------------------------
def get_pdf_text(pdf_docs):
    """Extract text from multiple PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text


def get_text_chunks(text):
    """Split long text into overlapping chunks for vector search."""
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vectorstore(text_chunks, embeddings):
    """Create a FAISS vector store from text chunks."""
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# -----------------------------
# CONVERSATION CHAIN (LOCAL)
# -----------------------------
def get_conversation_chain(vectorstore, llm_pipeline, k=5):
    """Create a simple question-answering chain."""
    def ask(question):
        docs = vectorstore.similarity_search(question, k=k)
        context = " ".join([d.page_content for d in docs])

        prompt = f"""
You are an AI assistant that answers questions based strictly on the given context.
If the answer is not in the context, reply: "I could not find this in the document."

Context:
{context}

Question:
{question}

Answer:
"""
        result = llm_pipeline(prompt, max_length=256, do_sample=False)
        return result[0]["generated_text"].strip()
    return ask

# -----------------------------
# MAIN STREAMLIT APP
# -----------------------------
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    embeddings, llm_pipeline = load_local_models()

    # Sidebar settings
    st.sidebar.subheader("⚙️ Settings")
    context_k = st.sidebar.slider("Context Chunks", 2, 8, 5)
    st.sidebar.caption(f"🧠 Model: {LLM_MODEL.split('/')[-1]}")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about your documents:")

    if user_question and st.session_state.conversation:
        with st.spinner("Generating answer..."):
            answer = st.session_state.conversation(user_question)
        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("bot", answer))

    # Display chat history
    for role, msg in st.session_state.chat_history:
        template = user_template if role == "user" else bot_template
        st.write(template.replace("{{MSG}}", msg), unsafe_allow_html=True)

    # PDF uploader
    with st.sidebar:
        st.subheader("📂 Your PDFs")
        pdfs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if not pdfs:
                st.warning("Please upload at least one PDF first!")
                return

            with st.spinner("Processing PDFs and creating vector embeddings..."):
                text = get_pdf_text(pdfs)
                chunks = get_text_chunks(text)
                vectorstore = get_vectorstore(chunks, embeddings)
                st.session_state.conversation = get_conversation_chain(vectorstore, llm_pipeline, k=context_k)
                st.session_state.chat_history = []
                st.success("✅ PDFs processed successfully!")

if __name__ == "__main__":
    main()
