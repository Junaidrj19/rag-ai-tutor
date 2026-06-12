# 📚 RAG AI Tutor

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that answers questions from uploaded PDF documents using LangChain, FAISS, HuggingFace embeddings, and Large Language Models.

The system intelligently handles both **native digital PDFs** and **scanned/handwritten documents** by automatically selecting the appropriate extraction method for each page.

---

## 🚀 Features

* Chat with your PDF documents
* Automatic detection of digital vs scanned pages
* Native text extraction using **PyMuPDF**
* OCR fallback for scanned/handwritten pages using **Mistral OCR**
* Semantic search using **FAISS Vector Store**
* HuggingFace sentence embeddings
* LangChain retrieval pipeline
* Streamlit web interface
* Source-aware retrieval with page metadata

---

## 🏗️ Architecture

```text
                User uploads PDF
                        │
                        ▼
                 PyMuPDF Extraction
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
 Native digital page          Scanned/Handwritten page
        │                               │
        ▼                               ▼
 page.get_text()              Mistral OCR Fallback
        │                               │
        └───────────────┬───────────────┘
                        ▼
              Clean extracted text
                        ▼
      RecursiveCharacterTextSplitter
                        ▼
              HuggingFace Embeddings
                        ▼
                  FAISS Vector Store
                        ▼
                 Similarity Retrieval
                        ▼
                  Large Language Model
                        ▼
                    Final Response
```

---

## 🛠️ Tech Stack

* Python
* LangChain
* PyMuPDF (fitz)
* Mistral OCR
* FAISS
* HuggingFace Embeddings
* Streamlit
* python-dotenv

---

## 📂 Project Structure

```text
rag_langchain_ocr/

├── backend.py
├── streamlit_app.py
├── services/
│   ├── pdf_service.py
│   ├── chunk_service.py
│   ├── embedding_service.py
│   ├── retrieval_service.py
│   ├── llm_service.py
│   ├── prompt_service.py
│   └── chat_service.py
├── data/
├── faiss_index/
└── requirements.txt
```

---

## ⚙️ Extraction Strategy

Each page of a PDF is processed independently.

* If the page contains a native text layer, PyMuPDF extracts the text directly.
* If the page contains insufficient native text, it is classified as scanned or handwritten.
* The page is rendered at high resolution and processed through Mistral OCR.
* The extracted content is cleaned and converted into LangChain Documents for chunking and retrieval.

This hybrid strategy improves both extraction quality and processing efficiency.

---

## 📈 Current Pipeline

```text
PDF
 │
 ▼
PyMuPDF
 │
 ├── Native Text Available
 │          │
 │          ▼
 │   Native Extraction
 │
 └── Scanned Page
            │
            ▼
       Mistral OCR
            │
            ▼
      Cleaned Text
            │
            ▼
Chunking → Embeddings → FAISS → Retriever → LLM
```

---

## 🎯 Future Improvements

* Conversational memory
* Query rewriting
* Hybrid search (BM25 + Dense Retrieval)
* Reranking models
* Multimodal RAG support
* OCR confidence-based fallback
* Docker deployment
* Cloud deployment

---

## 👨‍💻 Author

**Mohammed Junaid**

Built as a hands-on project to explore modern Retrieval-Augmented Generation systems, document intelligence pipelines, vector databases, and LLM application development.
