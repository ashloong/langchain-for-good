import os
import gradio as gr
from datetime import date
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import create_extraction_chain

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from prompts import INTENT_DETECTION_TEMPLATE, DATE_EXTRACTOR_PROMPT_TEMPLATE

# --- 1. Load Environment and Models (on startup) ---
load_dotenv()

llm = HuggingFaceEndpoint(
    #repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    repo_id="dousery/medical-reasoning-gpt-oss-20b", # Medical Reasoning GPT-OSS 20B - Strong medical knowledge
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    temperature=0.3
)

# --- 2. Initialize Retriever for Medical QA (on startup) ---
try:
    # Load documents from trusted sources
    loader = WebBaseLoader([
        "https://www.who.int/news-room/fact-sheets/detail/cancer",
        "https://www.cdc.gov/cancer/dcpc/resources/features/what-is-cancer.htm"
    ])
    docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and vector store
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

    # Create the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print("✅ Retriever initialized successfully.")

except Exception as e:
    retriever = None
    print(f"⚠️ Retriever initialization failed: {e}")
    print("Medical QA functionality will be disabled.")


# --- 3. Define Chains ---

# Intent Detection Chain
intent_prompt = ChatPromptTemplate.from_template(INTENT_DETECTION_TEMPLATE)
intent_chain = intent_prompt | llm | StrOutputParser()

# Time-Off Date Extraction Chain
timeoff_schema = {
    "properties": {
        "start_date": {"type": "string"},
        "end_date": {"type": "string"},
    },
    "required": ["start_date", "end_date"],
}
timeoff_chain = create_extraction_chain(timeoff_schema, llm, DATE_EXTRACTOR_PROMPT_TEMPLATE)
timeoff_prompt = ChatPromptTemplate.from_template(DATE_EXTRACTOR_PROMPT_TEMPLATE)
timeoff_chain = create_extraction_chain(timeoff_schema, llm, prompt=timeoff_prompt)

# Medical QA Chain
qa_prompt_template = """
You are a helpful medical assistant. Answer the user's question based only on the following context.
If the context does not contain the answer, state that you cannot answer the question. Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
qa_prompt = ChatPromptTemplate.from_template(qa_prompt_template)
qa_chain = qa_prompt | llm | StrOutputParser()


# --- 4. Main Prediction Logic ---
def get_response(message, history):
    try:
        # Step 1: Detect intent
        intent_response = intent_chain.invoke({"user_message": message})
        
        if "GREETING" in intent_response:
            return "Hello! How can I assist you today? Are you looking for medical information or requesting time off?"
            
        elif "TIMEOFF" in intent_response:
            today = str(date.today())
            extraction = timeoff_chain.invoke({"input": message, "todays_date": today})
            if extraction and extraction.get('text'):
                dates = extraction['text'][0]
                return f"Time-off request received. Start Date: {dates.get('start_date')}, End Date: {dates.get('end_date')}"
            else:
                return "I couldn't determine the dates for your time-off request. Please be more specific."

        # Default to 'OTHER' (Medical QA)
        else:
            if retriever is None:
                return "I'm sorry, the medical information service is currently unavailable."
            
            # Step 2: Retrieve relevant documents
            docs = retriever.get_relevant_documents(message)
            context = "\n\n".join([d.page_content for d in docs])
            
            # Step 3: Generate answer
            return qa_chain.invoke({"context": context, "question": message})

    except Exception as e:
        return f"I encountered an error: {e}. Please try again."

# --- 5. Gradio Interface ---
iface = gr.ChatInterface(
    fn=get_response,
    title="Multi-Purpose Assistant",
    description="Ask medical questions or request time off.",
).launch()
