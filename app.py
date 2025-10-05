import os
import streamlit as st
import requests
# Removed from local imports: from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Removed from local imports: HuggingFacePipeline, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from htmlTemplates import css, bot_template, user_template


# --- Production Model Configuration ---
EMBEDDING_MODEL_PROD = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_PROD = "google/flan-t5-small"
HF_API_URL = "https://api-inference.huggingface.co/models/"


# --- Environment Setup ---

# Streamlit Cloud uses st.secrets, no need for dotenv here.
try:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except KeyError:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets.")
    st.stop()
    
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


# --- Hugging Face API Call (LLM Inference) ---

def query_hf_model(model_name, prompt):
    """Makes a POST request to the Hugging Face Inference API."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}
    
    response = requests.post(HF_API_URL + model_name, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            # The API returns a list of generations
            return data[0]["generated_text"].strip()
        return str(data).strip()
    else:
        st.error(f"❌ LLM API Error {response.status_code} on model {model_name}: {response.text}")
        return "Sorry, the AI model encountered an error."


# --- Cached Embeddings (Small, Production-Ready Model) ---

@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Loads the memory-efficient embedding model for production."""
    st.sidebar.info(f"Using Embeddings: {EMBEDDING_MODEL_PROD.split('/')[-1]}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PROD)


# --- PDF Handling (No changes needed) ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vectorstore(text_chunks, embeddings):
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# --- Conversation Chain (API Exclusive) ---

def get_conversation_chain(vectorstore, model_choice, k=5):
    """
    Creates an 'ask' function that retrieves context and queries the HF API.
    llm object is not used in this production version.
    """
    def ask(question):
        # 1. Retrieve context
        docs = vectorstore.similarity_search(question, k=k)
        context = " ".join([d.page_content for d in docs])

        # 2. Construct RAG prompt
        prompt = f"""
        You are an AI assistant that answers questions based only on the provided context.
        If the answer is not in the context, say "I could not find this in the document."

        Context:
        {context}

        Question:
        {question}

        Answer in detail:
        """

        # 3. Query LLM via API
        return query_hf_model(model_choice, prompt)

    return ask


# --- Streamlit App ---

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Load embeddings (will use the small production model)
    embeddings = load_embeddings()

    # Sidebar settings
    st.sidebar.subheader("⚙️ Settings")
    st.sidebar.caption("🚀 Running in Streamlit Cloud (API Mode)")
    
    # User can select model for API inference
    model_choice = st.sidebar.selectbox(
        "Choose LLM model (API)",
        [LLM_MODEL_PROD, "google/flan-t5-base"],
        index=0 # Default to smallest for best performance
    )
    st.sidebar.caption(f"LLM API model: {model_choice.split('/')[-1]}")
    
    context_size = st.sidebar.slider("Number of chunks for context", min_value=2, max_value=8, value=5)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question and st.session_state.conversation:
        with st.spinner("Generating response via API..."):
            response = st.session_state.conversation(user_question)

        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("bot", response))

    # Display chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.write(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("📂 Your documents")
        pdf_docs = st.file_uploader("Upload PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF first!")
                return

            with st.spinner("Processing PDFs and creating vector embeddings..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks, embeddings)

                # Reset chat history and create chain using the selected API model
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, 
                    model_choice=model_choice, 
                    k=context_size
                )
                st.session_state.chat_history = []
                st.success("✅ PDFs processed and vector embeddings created!")

if __name__ == "__main__":
    main()
