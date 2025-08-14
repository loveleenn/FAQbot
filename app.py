
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile

st.set_page_config(page_title="FAQ ChatBot", page_icon="🤖")
st.title("📄 Ask Questions from your PDF!")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

if uploaded_file and openai_api_key:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and split PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Setup RetrievalQA
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    question = st.text_input("Ask a question about your PDF:")
    if question:
        with st.spinner("Searching for the answer..."):
            answer = qa_chain.run(question)
            st.write("Answer:", answer)
