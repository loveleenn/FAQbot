import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_faq_document(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

def create_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def build_qa_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

st.title("📄 Smart FAQ Bot")
uploaded_file = st.file_uploader("Upload your company FAQ PDF", type=["pdf"])

if uploaded_file is not None:
    with open("temp_faq.pdf", "wb") as f:
        f.write(uploaded_file.read())
    docs = load_faq_document("temp_faq.pdf")
    db = create_vector_store(docs)
    qa_chain = build_qa_chain(db)

    user_question = st.text_input("Ask a question from the document:")
    if user_question:
        answer = qa_chain.run(user_question)
        st.write("💬 Answer:", answer)
