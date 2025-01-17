import os
import sys
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Force usage of pysqlite3 instead of system sqlite3
__import__('pysqlite3')
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
st.title("PDF/Excel Chatbot (Multiple Files)")

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
# File Uploader
# --------------------------
uploaded_file = st.file_uploader("Upload your PDF or Excel file", type=['pdf', 'xlsx', 'xls'])

if uploaded_file is not None:
    # Determine file extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()  # e.g. ".pdf", ".xlsx", ".xls"
    
    # Save file locally if it doesn't exist
    file_path = os.path.join("files", uploaded_file.name)
    if not os.path.isfile(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Saved file: {file_path}")

    # 3) Create a unique folder for each file in the "jj" directory
    file_folder_name = os.path.splitext(uploaded_file.name)[0]  # e.g., "resume" from "resume.pdf"
    persist_dir = os.path.join("jj", file_folder_name)

    # --------------------------
    # Load and process PDF or Excel
    # --------------------------
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()

    elif file_extension in [".xlsx", ".xls"]:
        # Read each sheet and transform into Document objects
        xls = pd.ExcelFile(file_path)
        docs = []
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            # Convert the DataFrame to a string (CSV-like)
            text = df.to_csv(index=False)
            # Create a Document from this text
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path, "sheet_name": sheet_name}
                )
            )
    else:
        st.error("Unsupported file format.")
        st.stop()

    # --------------------------
    # Split documents
    # --------------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)

    # --------------------------
    # Create Vector Store for this file if not already created
    # --------------------------
    if file_folder_name not in st.session_state["pdf_vectorstores"]:
        st.session_state["pdf_vectorstores"][file_folder_name] = Chroma.from_documents(
            doc_splits,
            embedding=OllamaEmbeddings(
                base_url="https://salary-approval-local-resistance.trycloudflare.com",
                model="mistral"
            ),
            persist_directory=persist_dir
        )
        st.success(f"Vector store created for {file_folder_name} at {persist_dir}")
    else:
        # If this folder name already exists, you could decide whether to add new docs or skip
        st.warning(f"A vector store already exists for {file_folder_name}")

# --------------------------
# Query Section
# --------------------------
# Let the user pick which file to query
file_choices = list(st.session_state["pdf_vectorstores"].keys())
if file_choices:
    selected_file = st.selectbox("Choose which file to chat with:", file_choices)
    if selected_file:
        user_query = st.text_input("Enter your question:")
        if user_query:
            # Create a retriever from the chosen store
            chosen_store = st.session_state["pdf_vectorstores"][selected_file]
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
    st.info("No PDFs or Excel files loaded yet. Upload a file to begin.")
