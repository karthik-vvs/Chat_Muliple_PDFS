import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from htmlTemplates import css, bot_template, user_template

# Cache embeddings model
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")

# Cache HuggingFace LLM
@st.cache_resource(show_spinner=False)
def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=pipe)

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

def get_conversation_chain(vectorstore, llm, k=5):
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
        result = llm(prompt)

        # Ensure valid response
        if isinstance(result, str):
            return result.strip() or "I could not find this in the document."
        return str(result)
    return ask

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    embeddings = load_embeddings()

    # Sidebar settings
    st.sidebar.subheader("⚙️ Settings")
    model_choice = st.sidebar.selectbox(
        "Choose model",
        ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"],
        index=1
    )
    context_size = st.sidebar.slider("Number of chunks for context", min_value=2, max_value=8, value=5)

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

                # Reset chat history and create chain
                st.session_state.conversation = get_conversation_chain(vectorstore, llm, k=context_size)
                st.session_state.chat_history = []
                st.success("✅ PDFs processed and vector embeddings created!")

if __name__ == "__main__":
    main()
