# Caregiver Agentic AI

This repository contains the implementation of a **Caregiver Agentic AI**, designed to assist caregivers through natural language understanding, medical note summarization, and action item extraction.
It uses **LangGraph**, **LangChain**, and **Groq LLMs** to build modular, agentic reasoning pipelines.

---

## 🚀 Getting Started

### Prerequisites

* Python **3.10+**
* A valid **GROQ API key** (set in your `.env` file)
* Virtual environment (recommended)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/mahsa-ebrahimian/langchain-for-good.git
   cd langchain-for-good
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up your environment variables**

   Create a `.env` file in the project root and add your Groq API key:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

---

## 🧠 Usage

Run the main entry point to start the Caregiver Agent pipeline:

```bash
python main.py
```

This will:

* Initialize the **CaregiverCompanionAgent**
* Summarize and explain medical notes
* Query CDC medical information
* Run the **LangGraph** caregiver pipeline for structured reasoning

---

## 📂 Project Structure

```
langchain-for-good/
│
├── main.py                            # Entry point to run the full caregiver pipeline
├── agents/
│   └── caregiver_agent.py             # Defines the CaregiverCompanionAgent
├── graphs/
│   └── caregiver_graph.py             # Builds the LangGraph caregiving workflow
├── orchestrators/
│   └── run_caregiver_graph.py         # Orchestrates pipeline execution
├── pipelines/
│   └── cdc_retrieval_qa.py            # CDC knowledge retrieval QA chain
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## 🤝 Contributing

If you’d like to contribute:

1. Fork the repository
2. Create a new branch

   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit

   ```bash
   git commit -m "Add feature description"
   ```
4. Push and open a Pull Request

---

**Code4Impact Hackathon: LangChain Agents for Social Good** 🌍

