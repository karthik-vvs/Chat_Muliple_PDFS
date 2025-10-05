import os
import streamlit as st
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from htmlTemplates import css, bot_template, user_template


# --- Detect environment ---
def is_deployed():
    """Returns True if running on Streamlit Cloud (no local .env file)."""
    return os.environ.get("STREAMLIT_RUNTIME", "") != ""


# --- Load environment variables ---
load_dotenv()
if is_deployed():
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


# --- Hugging Face API Call (for deployment) ---
HF_API_URL = "https://api-inference.huggingface.co/models/"


def query_hf_model(model_name, prompt):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}
    response = requests.post(HF_API_URL + model_name, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    else:
        return f"❌ Error {response.status_code}: {response.text}"


# --- Cached Embeddings ---
@st.cache_resource(show_spinner=False)
def load_embeddings():
    model_name = "hkunlp/instructor-xl" if not is_deployed() else "hkunlp/instructor-base"
    return HuggingFaceEmbeddings(model_name=model_name)


# --- Cached Local Model (for development) ---
@st.cache_resource(show_spinner=False)
def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=pipe)


# --- PDF Handling ---
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


# --- Conversation Chain ---
def get_conversation_chain(vectorstore, llm=None, model_name=None, k=5):
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

        Answer in detail:
        """

        if is_deployed():
            return query_hf_model(model_name, prompt)
        else:
            result = llm(prompt)
            if isinstance(result, str):
                return result.strip() or "I could not find this in the document."
            return str(result)

    return ask


# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    embeddings = load_embeddings()

    # Sidebar Settings
    st.sidebar.subheader("⚙️ Settings")

    model_choice = st.sidebar.selectbox(
        "Choose model",
        ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"],
        index=1,
    )

    context_size = st.sidebar.slider("Number of chunks for context", 2, 8, 5)

    if is_deployed():
        st.sidebar.info("🚀 Running in Streamlit Cloud (using Hugging Face API)")
        llm = None  # not used
    else:
        st.sidebar.success("🧠 Running Locally (full model mode)")
        llm = load_llm(model_choice)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question and st.session_state.conversation:
        response = st.session_state.conversation(user_question)

        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("bot", response))

    # Display chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.write(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)

    # Sidebar: PDF Upload
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

                if is_deployed():
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore, model_name=model_choice, k=context_size
                    )
                else:
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore, llm=llm, k=context_size
                    )

                st.session_state.chat_history = []
                st.success("✅ PDFs processed and vector embeddings created!")


if __name__ == "__main__":
    main()
