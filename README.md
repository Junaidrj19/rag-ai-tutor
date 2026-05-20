# 🎓 AI-Powered B.Tech Tutor using RAG + LangChain

A custom Retrieval-Augmented Generation (RAG) based AI Tutor built using Python, LangChain, FAISS, Flask and Streamlit.

This project was built to deeply understand how modern AI systems work beyond simply calling an LLM API. The focus was on implementing the complete RAG pipeline including embeddings, vector search, retrieval, conversational memory and prompt augmentation.

---

# 🚀 Features

* 📄 PDF ingestion and preprocessing
* ✂️ Intelligent text chunking
* 🧠 Semantic embeddings generation
* 🗂️ FAISS vector database integration
* 🔍 Similarity-based retrieval
* 💬 Conversational memory support
* 🤖 Retrieval-Augmented Generation (RAG)
* ⚡ Real-time AI responses
* 🌐 Streamlit frontend interface
* 🔗 LangChain orchestration
* 🧩 Modular backend architecture

---

# 🧠 What is RAG?

Retrieval-Augmented Generation (RAG) combines:

1. Information Retrieval
2. Large Language Models (LLMs)

Instead of relying only on the LLM’s pretrained knowledge, the system retrieves relevant context from custom documents and injects it into the prompt before generation.

This significantly reduces hallucinations and allows the chatbot to answer based on uploaded knowledge sources.

---

# 🏗️ System Architecture

User Query
↓
Streamlit Frontend
↓
Flask Backend API
↓
LangChain Retrieval Pipeline
↓
FAISS Vector Database
↓
Relevant Chunks Retrieved
↓
Prompt Augmentation
↓
LLM Response Generation
↓
Answer Returned to User

---

# ⚙️ Tech Stack

## Backend

* Python
* Flask
* LangChain

## AI / ML

* FAISS
* Embeddings
* RAG Pipeline
* Prompt Engineering

## Frontend

* Streamlit

## APIs

* Gemini API / Groq API

---

# 📚 Concepts Implemented

## 1. Text Chunking

Large documents are split into smaller semantic chunks for efficient retrieval and better context relevance.

## 2. Embeddings

Text is converted into dense vector representations that capture semantic meaning mathematically.

## 3. Vector Database (FAISS)

Embeddings are stored and indexed using FAISS for fast similarity search and retrieval.

## 4. Semantic Retrieval

Instead of keyword matching, the system retrieves context based on semantic similarity.

## 5. Prompt Augmentation

Retrieved chunks are dynamically inserted into prompts before sending them to the LLM.

## 6. Conversational Memory

The chatbot maintains chat history and contextual continuity across interactions.

## 7. LangChain Orchestration

LangChain simplifies chaining together retrieval, prompts, memory and LLM calls.

---

# 📂 Project Structure

```bash
project/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
│
│
├── data/
│   └── Unit-I DS Data Structures copy.pdf
│
└── templates/
```

# 🔧 Installation

## 1. Clone Repository

```bash
git clone https://github.com/Junaidrj19/rag-ai-tutor.git

cd your-repo-name
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

## 3. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Add Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
GOOGLE_API_KEY=your_api_key
```

---

# ▶️ Running the Project

## Start Flask Backend

```bash
python app.py
```

## Start Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

---

# 📸 Demo

(Add screenshots/GIFs here)

---

# 🧪 Future Improvements

* Multi-PDF support
* Streaming responses
* Source citations
* Persistent chat history
* PostgreSQL integration
* FastAPI migration
* Docker deployment
* Kubernetes deployment
* User authentication
* Cloud deployment

---

# 🎯 Learning Outcomes

This project helped in understanding:

* How RAG systems work internally
* How vector databases operate
* Why embeddings are powerful
* How semantic retrieval differs from keyword search
* How conversational AI pipelines are engineered
* How LangChain simplifies AI workflows
* Backend + AI system integration

---

# 🤝 Contributing

Contributions, suggestions and improvements are welcome.

---

# 📜 License

MIT License

---

# 👨‍💻 Author

Junaid

Building AI systems and learning modern AI engineering step by step.
