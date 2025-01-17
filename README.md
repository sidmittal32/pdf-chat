# AI-Powered Document Q&A Web Application

## Overview
This project is a web-based application that allows users to upload documents and interact with them using AI-powered Q&A functionality. Users can ask both factual and contextual questions about the uploaded content and receive precise answers. The application supports document formats like PDFs and Excel files, providing an intuitive and user-friendly interface hosted on Streamlit.

---

## Features

### 1. **Document Upload**
- Users can upload documents in popular formats such as PDFs and Excel files.
- The application processes the uploaded file to extract readable content from both text-based and tabular formats.

### 2. **AI-Powered Q&A**
- Users can ask questions about the uploaded document.
- The application leverages the Mistral language model for generating accurate and relevant answers.
- Supports both factual and contextual questions.

### 3. **Multi-Document Support**
- Users can upload multiple documents and switch between them for querying.
- Each document is stored with its own vector embeddings for better retrieval.

### 4. **Deployment**
- The web application is hosted on Streamlit.
- The Mistral model runs locally using Docker, exposed via Cloudflare Tunnel.

### 5. **Code Repository**
- All source code is maintained on a GitHub repository.
- The repository includes clear commit logs and adheres to good coding practices.

---

## Architecture

1. **Frontend**:
   - Built with Streamlit for a responsive and user-friendly interface.
2. **Backend**:
   - Powered by the Mistral language model running in a Docker container.
   - Cloudflare Tunnel is used to expose the locally running Docker container to the public web.
3. **Q&A Engine**:
   - Parses document content and uses the AI model for answering user queries.
4. **Document Parsing**:
   - Processes PDF files to extract structured text.
   - Reads Excel files, converting sheets into text for AI analysis.

---

## How to Run Locally

### Prerequisites
- Python (3.9+)
- Docker
- Cloudflare CLI (`cloudflared`)
- Git

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sidmittal32/pdf-chat.git
   cd pdf-chat
   ```

2. **Set Up the Environment**:
   - Create a virtual environment:
     ```bash
     python -m venv env
     source env/bin/activate  # On Windows: env\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Docker Container**:
   - Pull and run the Docker container for the Mistral model:
     ```bash
     docker run -d -p 11434:11434 mistral-docker-image
     ```

4. **Expose Docker via Cloudflare Tunnel**:
   - Start a Cloudflare Tunnel:
     ```bash
     cloudflared tunnel --url http://localhost:11434
     ```
   - Note the public URL provided by Cloudflare and update your application code accordingly.

5. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

6. **Access the App**:
   - Open the Streamlit app in your browser using the provided URL (e.g., `http://localhost:8501`).


## Repository Structure
```
📁 pdf-chat
├── 📁 files               # Uploaded documents storage
├── 📁 jj                  # Chroma vector store (optional)
├── 📜 app.py              # Main Streamlit application
├── 📜 Dockerfile          # Docker configuration for the Mistral model
├── 📜 requirements.txt    # Python dependencies
├── 📜 README.md           # Project documentation
```

---

## Requirements

- **Backend**:
  - Python (3.9+)
  - Mistral model (running in Docker)
- **Frontend**:
  - Streamlit
- **Hosting**:
  - Cloudflare Tunnel (for exposing Docker container)
- **Libraries**:
  - Streamlit
  - LangChain
  - ChromaDB
  - DuckDB
  - PyPDF
  - Pandas (for Excel processing)

---

## Contributions

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with detailed commit messages.

---

## Credits
- AI Model: [Mistral](https://mistral.ai)
- Deployment: Streamlit, Docker, and Cloudflare Tunnel
- Document Parsing: Python libraries (LangChain, ChromaDB, Pandas)

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
