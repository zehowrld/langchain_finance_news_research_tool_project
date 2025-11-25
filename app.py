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

# Load environment variables locally 
load_dotenv()

# --- API Key & Configuration ---
OWNER_API_KEY = None
try:
    # 1. Try Streamlit Secrets (for convenience when deploying)
    OWNER_API_KEY = st.secrets.get("GEMINI_API_KEY")
except Exception:
    # 2. Fall back to local Environment variable (for convenience when running locally via .env)
    OWNER_API_KEY = os.getenv("GEMINI_API_KEY")

# Path to the vector store
INDEX_DIR = "notebooks/faiss_index_gemini" 

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Baymax: News Research Tool", layout="centered")
st.title("Baymax: News Research Tool 📈")

# --- Sidebar UI and Key Input ---
with st.sidebar:
    st.title("Configuration")
    
    # This input field is now the ONLY source of the API key used by the RAG functions.
    # Public users must paste their key here. It is pre-filled for the owner if found in .env or Secrets.
    api_key_input = st.text_input(
        "1. Gemini API Key (Required)",
        type="password",
        value=OWNER_API_KEY if OWNER_API_KEY else "",
        help="Enter your personal Google Gemini API Key to run the RAG process."
    )
    
    #  API Key check feedback
    if not api_key_input:
        st.warning("Please enter a Gemini API Key to proceed.")
    else:
        st.success("API Key detected.")
        
    st.subheader("2. News Article URLs")
    urls = []
    for i in range(3):
        url = st.text_input(f"URL {i+1}", key=f"baymax_url_{i+1}")
        urls.append(url)

    process_url_clicked = st.button("3. Process URLs & Build Index")
    
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
        # Initialize embeddings with the USER-PROVIDED key
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )

        # Check for new URLs to process
        valid_urls = [url for url in urls if url and url.strip()]

        if valid_urls:
            # CASE 1: Build New Index (Force Refresh)
            st.info("Loading data from URLs...")
            loader = UnstructuredURLLoader(urls=valid_urls)
            data = loader.load()
            
            # --- SAFETY CHECK: VALIDATE CONTENT ---
            if not data or len(data) == 0:
                return None, "Error: No content could be extracted from these URLs."
            
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
                st.info("Deleting old index before rebuilding...")
                shutil.rmtree(index_path) 

            # 4. Create & Save New Index
            st.info("Creating new vector index...")
            vectorstore = FAISS.from_documents(docs, embeddings)
            
            index_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(folder_path=INDEX_DIR)
            
            return vectorstore, f"Success: New index built from {len(valid_urls)} URLs and saved."

        elif index_path.exists():
            # CASE 2: Load Existing Index (Only if no new URLs provided)
            try:
                st.info("Loading existing index from disk...")
                vectorstore = FAISS.load_local(
                    folder_path=INDEX_DIR, 
                    embeddings=embeddings, 
                    allow_dangerous_deserialization=True
                )
                return vectorstore, "Success: Loaded existing index from disk."
            except Exception as e:
                print(f"Error loading existing index: {e}")
                return None, f"Error loading existing index: {e}"
        
        else:
            return None, "No URLs provided and no existing index found. Please provide URLs and a key."

    except Exception as e:
        print(f"Vector Store Critical Error: {e}")
        return None, f"Critical Error during index operations: {e}"

@st.cache_resource
def get_rag_chain(_vectorstore, api_key):
    """
    Creates the RetrievalQA chain. 
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key, # Uses the user-provided key
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
    # Use the key from the sidebar input
    if not api_key_input:
        st.error("Cannot process: Please enter a Gemini API Key.")
    else:
        with st.spinner("Processing URLs, Chunking Data, and Creating Embeddings..."):
            st.cache_resource.clear()
            
            # Use the key from the input field
            v_store, msg = get_vector_store(urls, api_key_input)
            
            if v_store:
                st.session_state['vectorstore'] = v_store
                # Use the key from the input field
                st.session_state['chain'] = get_rag_chain(v_store, api_key_input)
                st.success(msg)
            else:
                st.error(msg)
                st.session_state['vectorstore'] = None
                st.session_state['chain'] = None


# Main Query Interface
query = st.text_input("Question:")

if query:
    # 1. Attempt to load the chain if it's not in session state (e.g., app restart or index existed)
    if 'chain' not in st.session_state or st.session_state['chain'] is None:
        if api_key_input:
            # Auto-load existing index if key is present
            # Use the key from the input field
            v_store, msg = get_vector_store([], api_key_input) 
            if v_store:
                st.session_state['vectorstore'] = v_store
                # Use the key from the input field
                st.session_state['chain'] = get_rag_chain(v_store, api_key_input)
                st.success("Loaded existing index and RAG chain.")
            else:
                st.warning("Please process URLs first to build the index.")
                st.session_state['vectorstore'] = None
                st.session_state['chain'] = None
                st.stop()
        else:
            st.error("Cannot run query: API Key is missing. Please enter your key in the sidebar.")
            st.stop()
    
    # 2. Execute the query
    if st.session_state.get('chain'):
        chain = st.session_state['chain']
        
        # Debug: Show what the retriever is finding
        if show_retrieved_docs and st.session_state.get('vectorstore'):
            with st.expander("Debug: Retrieved Documents"):
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
                result = chain.invoke({"question": query})
                
                st.header("Answer")
                st.write(result["answer"])
                
                st.subheader("Sources")
                if result.get("sources"):
                    sources_list = [s.strip() for s in result["sources"].split(",")]
                    for source in sources_list:
                         if source:
                            st.markdown(f"- [{source}]({source})")
                else:
                    st.write("No specific sources cited.")
                    
            except Exception as e:
                st.error(f"Error generating answer (check API Key validity): {e}")