import os
import shutil
import streamlit as st
import time
from pathlib import Path
from dotenv import load_dotenv

# --- LangChain and Gemini Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Retrieve the API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Path to the vector store
INDEX_DIR = "notebooks/faiss_index_gemini" 

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Baymax: News Research Tool", layout="centered")
st.title("Baymax: News Research Tool 📈")

# Sidebar
with st.sidebar:
    st.title("Configuration")
    
    if GEMINI_API_KEY:
        st.success("API Key loaded.")
    else:
        st.error("GEMINI_API_KEY not found in .env file.")
        
    st.subheader("News Article URLs")
    urls = []
    for i in range(3):
        url = st.text_input(f"URL {i+1}", key=f"url_{i+1}")
        urls.append(url)

    process_url_clicked = st.button("Process URLs & Build Index")
    
    st.markdown("---")
    show_retrieved_docs = st.checkbox("Debug: Show Retrieved Docs")

# --- Core Functions ---

@st.cache_resource
def get_vector_store(urls, api_key):
    """
    Builds or loads the vector index. 
    Returns (vectorstore, message_string).
    """
    if not api_key:
        return None, "API Key missing."
    
    index_path = Path(INDEX_DIR)
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )

        # Check for new URLs to process
        valid_urls = [url for url in urls if url and url.strip()]

        if valid_urls:
            # CASE 1: Build New Index (Force Refresh)
            
            # 1. Load Data
            loader = UnstructuredURLLoader(urls=valid_urls)
            data = loader.load()
            
            # --- SAFETY CHECK: VALIDATE CONTENT ---
            if not data or len(data) == 0:
                return None, "Error: No content could be extracted from these URLs."
            
            # Calculate total text length to detect 'Access Denied' pages
            total_text_length = sum(len(doc.page_content) for doc in data)
            
            if total_text_length < 500:
                return None, (f"Error: Content too short ({total_text_length} chars). "
                              "The website likely blocked the scraper. "
                              "Your previous index was NOT overwritten.")

            # 2. Split Text
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(data)
            
            if not docs:
                 return None, "Error: Documents were loaded but splitting resulted in empty chunks."

            # 3. Clean Old Index (Crucial Step)
            if index_path.exists():
                shutil.rmtree(index_path) # Recursively delete the folder

            # 4. Create & Save New Index
            vectorstore = FAISS.from_documents(docs, embeddings)
            
            index_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(folder_path=INDEX_DIR)
            
            return vectorstore, f"Success: New index built from {len(valid_urls)} URLs."

        elif index_path.exists():
            # CASE 2: Load Existing Index (Only if no new URLs provided)
            try:
                vectorstore = FAISS.load_local(
                    folder_path=INDEX_DIR, 
                    embeddings=embeddings, 
                    allow_dangerous_deserialization=True
                )
                return vectorstore, "Success: Loaded existing index from disk."
            except Exception as e:
                return None, f"Error loading existing index: {e}"
        
        else:
            return None, "No URLs provided and no existing index found."

    except Exception as e:
        return None, f"Critical Error: {e}"

@st.cache_resource
def get_rag_chain(_vectorstore, api_key):
    """
    Creates the RetrievalQA chain. 
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )

    # k=4 retrieves top 4 relevant chunks
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})
    
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, 
        retriever=retriever
    )
    return chain

# --- Application Logic ---

if process_url_clicked:
    if not GEMINI_API_KEY:
        st.error("Please set GEMINI_API_KEY in .env")
    else:
        with st.spinner("Processing..."):
            # Clear previous cache to force rebuild if URLs changed
            st.cache_resource.clear()
            
            # Call cached function
            v_store, msg = get_vector_store(urls, GEMINI_API_KEY)
            
            if v_store:
                st.session_state['vectorstore'] = v_store
                st.session_state['chain'] = get_rag_chain(v_store, GEMINI_API_KEY)
                st.success(msg)
            else:
                st.error(msg)

# Main Query Interface
query = st.text_input("Question:")

if query:
    # Try to load chain from session state or auto-load existing index if strictly needed
    if 'chain' not in st.session_state or st.session_state['chain'] is None:
         # Auto-load attempt if user didn't click process but index exists
         if GEMINI_API_KEY:
             v_store, msg = get_vector_store([], GEMINI_API_KEY)
             if v_store:
                 st.session_state['vectorstore'] = v_store
                 st.session_state['chain'] = get_rag_chain(v_store, GEMINI_API_KEY)
                 st.success("Loaded existing index.")
             else:
                 st.error("Please process URLs first.")

    if st.session_state.get('chain'):
        chain = st.session_state['chain']
        
        # Debug: Show what the retriever is finding
        if show_retrieved_docs and st.session_state.get('vectorstore'):
             with st.expander("Debug: Retrieved Documents"):
                # Use .invoke on the retriever directly
                retriever = st.session_state['vectorstore'].as_retriever(search_kwargs={"k": 4})
                docs = retriever.invoke(query)
                
                if not docs:
                    st.warning("No relevant documents found in the index!")
                
                for d in docs:
                    st.write(d.page_content)
                    st.caption(f"Source: {d.metadata.get('source')}")
                    st.markdown("---")

        with st.spinner("Thinking..."):
            try:
                # Use .invoke() to avoid deprecation warning
                result = chain.invoke({"question": query})
                
                st.header("Answer")
                st.write(result["answer"])
                
                st.subheader("Sources")
                if result.get("sources"):
                    st.write(result["sources"])
                else:
                    st.write("No specific sources cited.")
                    
            except Exception as e:
                st.error(f"Error generating answer: {e}")
    else:
        st.warning("Please process URLs or ensure an index exists.")