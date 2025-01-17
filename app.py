import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --------------------------
# Create required directories
# --------------------------
if not os.path.exists('files'):
    os.mkdir('files')
if not os.path.exists('jj'):
    os.mkdir('jj')

# --------------------------
# Streamlit App Title
# --------------------------
st.title("PDF Chatbot (Multiple PDFs)")

# 1) Dictionary of vector stores
if "pdf_vectorstores" not in st.session_state:
    st.session_state["pdf_vectorstores"] = {}

# 2) Setup the LLM (Ollama)
if "llm" not in st.session_state:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    st.session_state.llm = Ollama(
        base_url="https://salary-approval-local-resistance.trycloudflare.com",
        model="mistral",
        verbose=True,
        callback_manager=callback_manager
    )

# --------------------------
# PDF Uploader
# --------------------------
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
if uploaded_file is not None:
    # Save PDF locally if it doesn't exist
    pdf_path = os.path.join("files", uploaded_file.name)
    if not os.path.isfile(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Saved file: {pdf_path}")

    # Load and split the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)

    # 3) Create a unique folder for each PDF in the "jj" directory
    pdf_folder_name = os.path.splitext(uploaded_file.name)[0]  # e.g., "resume" from "resume.pdf"
    persist_dir = os.path.join("jj", pdf_folder_name)

    # Create the vector store for this PDF only if it's not already created
    if pdf_folder_name not in st.session_state["pdf_vectorstores"]:
        st.session_state["pdf_vectorstores"][pdf_folder_name] = Chroma.from_documents(
            doc_splits,
            embedding=OllamaEmbeddings(
                base_url="https://salary-approval-local-resistance.trycloudflare.com",
                model="mistral"
            ),
            persist_directory=persist_dir
        )
        st.success(f"Vector store created for {pdf_folder_name} at {persist_dir}")
    else:
        # If this folder name already exists, you could decide whether to add new docs or skip
        st.warning(f"A vector store already exists for {pdf_folder_name}")

# --------------------------
# Query Section
# --------------------------
# Let the user pick which PDF to query
pdf_choices = list(st.session_state["pdf_vectorstores"].keys())
if pdf_choices:
    selected_pdf = st.selectbox("Choose which PDF to chat with:", pdf_choices)
    if selected_pdf:
        user_query = st.text_input("Enter your question:")
        if user_query:
            # Create a retriever from the chosen store
            chosen_store = st.session_state["pdf_vectorstores"][selected_pdf]
            retriever = chosen_store.as_retriever()

            # Build a RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                retriever=retriever,
                chain_type="stuff",
                verbose=True
            )

            # Run the query
            response = qa_chain.run(user_query)
            st.write("**Assistant:**", response)
else:
    st.info("No PDFs loaded yet. Upload a PDF to begin.")
