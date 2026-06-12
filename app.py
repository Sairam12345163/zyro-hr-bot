import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE CONFIGURATION & UI STYLING
st.set_page_config(page_title="Zyro Dynamics HR Desk", page_icon="🚀", layout="centered")
st.title("🚀 Zyro Dynamics HR Help Desk")
st.markdown("Welcome to your intelligent HR Assistant. Ask any question regarding company policy manuals.")

# Ensure keys are present
if "GROQ_API_KEY" not in os.environ:
    st.warning("Please ensure GROQ_API_KEY is configured in your hosting environment variables.")

# 2. CACHED RAG PIPELINE INITIALIZATION
@st.cache_resource(show_spinner="Initializing HR Knowledge Base System...")
def initialize_rag_system():
    # 🎯 FIXED DIRECTORY PATH MATCHING YOUR GITHUB REPO STRUCTURE
    data_dir = "./niat-masterclass-rag-challenge/zyro-dynamics-hr-corpus"
        
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 4})

try:
    retriever = initialize_rag_system()
except Exception as e:
    st.error(f"Initialization Error: Ensure the 'zyro-dynamics-hr-corpus' folder is placed with the app.")
    retriever = None

# 3. CONTEXT CHAIN SETUP
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

system_prompt = """You are an intelligent, polite HR Help Desk chatbot for Zyro Dynamics Pvt. Ltd. 
Your objective is to answer employee questions accurately using ONLY the provided document context below.

CRITICAL RULES:
1. Grounding: Answer the question using ONLY the provided text snippets. Do NOT make things up.
2. Out-of-Scope Handling: If the user asks about something completely unrelated to Zyro Dynamics HR policies, you MUST refuse clearly and politely.
3. Exact Refusal Phrase: If you must refuse or if the information is not found in the context, respond with EXACTLY this sentence:
"I can only answer HR-related questions from Zyro Dynamics policy documents."

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(system_prompt)

def format_docs_with_sources(docs):
    context_text = "\n\n".join(doc.page_content for doc in docs)
    citations = list(set([f"📄 {doc.metadata.get('source').split('/')[-1]} (Page {doc.metadata.get('page') + 1})" for doc in docs]))
    return context_text, citations

# 4. INTERACTIVE CHAT UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg:
            for source in msg["citations"]:
                st.caption(source)

if user_query := st.chat_input("Type your policy question here..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    if retriever:
        with st.chat_message("assistant"):
            with st.spinner("Searching records..."):
                retrieved_chunks = retriever.invoke(user_query)
                context_str, source_list = format_docs_with_sources(retrieved_chunks)
                
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({"context": context_str, "question": user_query})
                
                is_refusal = "I can only answer HR-related questions" in response
                st.markdown(response)
                
                if not is_refusal and source_list:
                    st.markdown("\n**Sources Cited:**")
                    for source in source_list:
                        st.caption(source)
                        
            log_entry = {"role": "assistant", "content": response}
            if not is_refusal:
                log_entry["citations"] = source_list
            st.session_state.messages.append(log_entry)
