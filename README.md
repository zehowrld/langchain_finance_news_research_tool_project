# Baymax: News Research Tool | Advanced RAG System for Financial Intelligence: Gemini + FAISS + Streamlit 📈

[![Deployed Streamlit App](https://img.shields.io/badge/Streamlit%20App-Live%20Demo-5D3FD3?style=flat-square&logo=streamlit)](https://finance-news-research.streamlit.app/)

## 🌟 Project Overview & Value Proposition

This project demonstrates an **Advanced Retrieval-Augmented Generation (RAG)** pipeline designed to provide contextually accurate and traceable insights from non-public or highly specialized data (financial news articles). It directly addresses the challenge of LLM hallucination by grounding all responses in a user-defined knowledge base, making it a reliable tool for financial analysis.

**This application showcases proficiency in:**

* Designing and implementing end-to-end RAG workflows, a critical skill in modern Generative AI engineering.

* Integrating enterprise-grade LLM services (Google Gemini) for high-quality semantic tasks.

* Developing scalable, user-facing Machine Learning applications (MLOps/Deployment via Streamlit).

* Managing high-performance vector databases (FAISS) for efficient semantic search at scale.

![Image](https://github.com/user-attachments/assets/43f5e48f-80da-4ad7-8fd6-062b5c59b552)
## 🧠 Methodology and Implementation

The system is engineered as a robust, decoupled pipeline ensuring efficient data processing and low-latency query handling.

### Core Architecture (RAG Pipeline)

<div align="center">
  <img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/6d0edead-0d8f-4909-9e58-8d3100d263e7" />
</div>

1. **Data Ingestion:** User-provided financial news URLs are scraped using the `UnstructuredURLLoader` to extract raw text data from diverse web formats.

2. **Text Processing:** Content is segmented into optimal chunks using `RecursiveCharacterTextSplitter` to maintain contextual relevance within each chunk for better retrieval.

3. **Vectorization:** Text chunks are transformed into dense, high-dimensional vector embeddings using **Google's `text-embedding-004` model**, ensuring high semantic fidelity and distinction.

4. **Vector Indexing:** Embeddings are indexed using **FAISS (Facebook AI Similarity Search)**. This library's optimization for L2 distance provides near-instantaneous similarity search performance. The index is persisted locally (in `faiss_index_gemini`) for quick re-use across sessions.

5. **Retrieval & Generation:** At query time, the system performs a vector search in FAISS to retrieve the top $k$ relevant document chunks. These chunks are injected into the prompt as context.

6. **Traceable Output:** The **Gemini `2.5-flash` model** generates a coherent response, explicitly citing the original source URLs for full auditability and fact-checking.

### Key Technologies and Skills Demonstrated

| **Category** | **Technology/Component** | **Focus Area** |
| :--- | :--- | :--- |
| **Generative AI** | Gemini 2.5-Flash, `text-embedding-004` | High-performance inference and state-of-the-art embedding generation. |
| **ML Engineering** | LangChain (`RetrievalQAWithSourcesChain`) | Orchestrating complex data flow, chaining LLM, and retriever components. |
| **Vector Storage** | FAISS, Vector Index Management | Expertise in high-speed, scalable similarity search algorithms. |
| **Deployment/MLOps** | Streamlit | Building and deploying interactive, user-friendly data science applications. |
| **Data Handling** | `UnstructuredURLLoader`, Python `pathlib`, `shutil` | Robust data ingestion and file system management. |

## ⚙️ Setup and Installation

Follow these steps to deploy and run the application.

### 1. Prerequisites

* Python 3.10+

* A **Google Gemini API Key** (required to run the app).

### 2. Local Environment Setup

Clone the repository and set up your environment:

```bash
# Clone the GitHub repository
git clone https://github.com/zehowrld/langchain_finance_news_research_tool_project.git
cd langchain_finance_news_research_tool_project

# Create and activate a Python virtual environment (recommended best practice)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
````

### 3\. Install Dependencies

Install all required Python packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4\. API Key Configuration

For local development convenience, the application will load your key from a local `.env` file.

  * Add your Gemini API Key in the following format:

<!-- end list -->

```
GEMINI_API_KEY="ENTER_YOUR_GEMINI_API_KEY_HERE"
```

### 5\. Run the Application

1. Execute the Streamlit script to launch the web application:

```bash
streamlit run app.py
```
2.  Interact with the Web App:   

    ### Option 1: Use Pre-Built URLs (Demo)

      * On the sidebar, click **"Process URLs & Build Index""** from existing data.
      * Ask a question based on the content, such as:
        > *What are the main features of punch iCNG?*

    ### Option 2: Analyze Custom URLs

      *  **Input URLs:** Enter one or more new URLs directly in the sidebar input field.
      *  **Process Data:** Click **"Process URLs & Build Index"** to initiate the data loading and processing.
      *  **Indexing Process:**
          * The system performs text splitting, generates embedding vectors, and indexes them efficiently using **FAISS**.
          * The FAISS index will be saved in a local file path in pickle format for future use.
      *  **Ask Questions:**
          * You can now ask a question and get the answer based on the articles you just processed.

## 📂 Project Structure

The project directory is organized to separate the main application logic, dependencies, and persisted data:

```
langchain_finance_news_research_tool_project/
├── app.py                     # Main Streamlit application script and RAG logic.
├── requirements.txt           # List of Python dependencies (LangChain, Streamlit, etc.).
├── .env                       # Local configuration for storing the GEMINI_API_KEY.
├── README.md                  # This documentation file.
└── notebooks/
    └── faiss_index_gemini/    # Directory where the FAISS vector index is persisted.
        ├── index.faiss        # The core FAISS index file.
        └── index.pkl          # Metadata associated with the index.
```

## 🔒 Public Deployment Note

This application is designed for sharing. Public users of the deployed Streamlit application are **required to enter their own Gemini API Key** in the sidebar. The application will not use or expose the developer's key, ensuring API usage is correctly managed by each user.
