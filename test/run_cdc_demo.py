#!/usr/bin/env python
# coding: utf-8

"""
Run a demo of CDC Retrieval QA pipeline:
- Uses LangChain, HuggingFace embeddings, FAISS, and ChatGroq
- Answers questions about medical topics from CDC pages
"""

from langchain.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

from utils.config import GROQ_API_KEY, CDC_URLS

# ======================
# Step 1: Load CDC Pages
# ======================
loader = WebBaseLoader(CDC_URLS)
docs = loader.load()

# ======================
# Step 2: Create Embeddings & VectorStore
# ======================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# ======================
# Step 3: Setup LLM
# ======================
llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY, temperature=0)

# ======================
# Step 4: Retrieval QA Chain
# ======================
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ======================
# Step 5: Ask a Question
# ======================
query = "What are the treatment options for breast cancer?"
response = qa_chain.run(query)

# ======================
# Step 6: Print Answer
# ======================
print("=== Question ===")
print(query)
print("\n=== Answer ===")
print(response)
