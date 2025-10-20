from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os

def build_cdc_qa(groq_api_key: str):
    urls = [
        "https://www.cdc.gov/flu/symptoms/index.html",
        "https://www.cdc.gov/cancer/breast/basic_info/index.htm",
    ]
    docs = WebBaseLoader(urls).load()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=groq_api_key, temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
