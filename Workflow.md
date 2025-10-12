**Workflow steps** to build **medical support agent** using **LangChain + Hugging Face (free API)**:


1️⃣ Project goal & setup

2️⃣ Environment & dependencies

3️⃣ Step-by-step LangChain workflow 

4️⃣ MVP

5️⃣ How to extend the project

---

### **1️⃣ Goal and Setup**

Goal: Build a stateful medical conversation agent that conducts multi-turn dialogues to gather symptoms, assess risk, and provide educational guidance using cyclical LangGraph workflows.

Core LangGraph Components:

State Management: Persistent conversation memory

Cyclical Flows: Adaptive questioning based on previous responses

Conditional Routing: Dynamic path selection based on medical urgency

Human-in-the-Loop: Seamless integration of user responses



* Install required packages:

  ```bash
  pip install langchain langgraph langchain-huggingface huggingface_hub 
  pip install langchain-community langchain-chroma python-dotenv beautifulsoup4 playwright
  ```

Install Playwright:

  ```bash
  playwright install
  ```

* Get **Hugging Face API key** → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  Add to `.env`:

  ```
  HUGGINGFACEHUB_API_TOKEN=hf_your_key
  ```

---

### **2️⃣ Load Medical Texts**

* Use `PyPDFLoader` to load medical documents (e.g., WHO or CDC guidelines).
* Split into text chunks for embedding:

  ```python
  from langchain_community.document_loaders import PyPDFLoader
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  ```

* Use web loaders to extract medical content from trusted sources:

``` 
from langchain_community.document_loaders import WebBaseLoader, AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
```

Method 1: WebBaseLoader (simple)

```
loader = WebBaseLoader(["https://www.who.int/health-topics", 
                       "https://www.cdc.gov/health-topics.html"])
docs = loader.load()
  ```

Method 2: AsyncHtmlLoader + Html2TextTransformer (for multiple pages)

  ```
loader = AsyncHtmlLoader(["https://www.mayoclinic.org/diseases-conditions"])
html_docs = loader.load()
html2text = Html2TextTransformer()
clean_docs = html2text.transform_documents(html_docs)
  ```
---

### **3️⃣ Create Vector Store**

* Convert text chunks into embeddings using a **free HF model** (e.g., `all-MiniLM-L6-v2`).
* Store them locally in **Chroma DB** for retrieval:

  ```python
  from langchain_community.embeddings import HuggingFaceEmbeddings
  from langchain_community.vectorstores import Chroma
  ```

---

### **4️⃣ Initialize Hugging Face LLM**

* Choose a conversational model (e.g., `mistralai/Mistral-7B-Instruct-v0.2`).

  ```python
  from langchain_huggingface import HuggingFaceEndpoint
  llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.3)
  ```

---

### **5️⃣ Build Retrieval Chain**

* Connect your LLM to the vector store for context-aware QA:

  ```python
  from langchain.chains import RetrievalQA
  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
  medical_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
  ```

---

### **6️⃣ Create Symptom Prompt**

* Design a safe, structured prompt for medical guidance (no diagnoses):

  ```python
  from langchain.prompts import PromptTemplate
  symptom_prompt = PromptTemplate(
      input_variables=["symptoms", "context"],
      template="You are a helpful medical assistant... (use trusted context)"
  )
  ```

---

### **7️⃣ Combine Everything**

* Define a function that retrieves relevant info and queries the model:

  ```python
  def medical_agent_query(symptom_text):
      docs = retriever.get_relevant_documents(symptom_text)
      context = "\n\n".join([d.page_content for d in docs])
      prompt = symptom_prompt.format(symptoms=symptom_text, context=context)
      return llm.invoke(prompt)
  ```

---

### **8️⃣ Test the Agent**

```python
response = medical_agent_query("I have a dry cough and mild fever.")
print(response)
```

✅ Output: concise, educational medical summary.

---

### **9️⃣ Add Trusted Web Sources**

* Use `DuckDuckGoSearchResults` to gather recent info from WHO, CDC, Mayo Clinic.

---

### **🔟 Extend with LangGraph Workflow**

* Create nodes for:

  * `SymptomInput`
  * `Retriever`
  * `LLMAnalyzer`
  * `Feedback`
* Connect edges to form a looped dialogue flow.

Install LangGraph
```bash
pip install langgraph
```

---



