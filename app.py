import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template

# -----------------------------
# CONFIGURATION
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "facebook/bart-large-cnn"  # Free API-supported model
HF_API_URL = "https://api-inference.huggingface.co/models/"

# -----------------------------
# LOAD TOKEN FROM STREAMLIT SECRETS
# -----------------------------
try:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except KeyError:
    st.error("❌ Add your HUGGINGFACEHUB_API_TOKEN in Streamlit Secrets.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# -----------------------------
# Hugging Face API Call
# -----------------------------
def query_hf_model(model_name, prompt):
    """Make a POST request to Hugging Face Inference API."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    try:
        response = requests.post(HF_API_URL + model_name, headers=headers, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request failed: {e}")
        return "Sorry, the AI model could not be reached."

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        else:
            return str(data)
    elif response.status_code == 404:
        st.error(f"❌ Model not found: {model_name}")
        return "Sorry, this model is not available for API inference."
    else:
        st.error(f"❌ LLM API Error {response.status_code}: {response.text}")
        return "Sorry, the AI model encountered an error."

# -----------------------------
# Cached Embeddings
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    st.sidebar.info(f"Using Embeddings: {EMBEDDING_MODEL.split('/')[-1]}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# -----------------------------
# PDF & Vector Helpers
# -----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vectorstore(text_chunks, embeddings):
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# -----------------------------
# Conversation Chain
# -----------------------------
def get_conversation_chain(vectorstore, model_choice, k=5):
    def ask(question):
        docs = vectorstore.similarity_search(question, k=k)
        context = " ".join([d.page_content for d in docs])

        prompt = f"""
You are an AI assistant that answers questions based only on the provided context.
If the answer is not in the context, say "I could not find this in the document."

Context:
{context}

Question:
{question}

Detailed Answer:
"""
        return query_hf_model(model_choice, prompt)
    return ask

# -----------------------------
# MAIN STREAMLIT APP
# -----------------------------
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    embeddings = load_embeddings()

    # Sidebar
    st.sidebar.subheader("⚙️ Settings")
    model_choice = st.sidebar.selectbox(
        "Choose LLM model (API)",
        [LLM_MODEL],
        index=0
    )
    context_k = st.sidebar.slider("Context Chunks", 2, 8, 5)
    st.sidebar.caption(f"🧠 Model: {model_choice.split('/')[-1]}")

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
                st.session_state.conversation = get_conversation_chain(vectorstore, model_choice, k=context_k)
                st.session_state.chat_history = []
                st.success("✅ PDFs processed successfully!")

if __name__ == "__main__":
    main()
