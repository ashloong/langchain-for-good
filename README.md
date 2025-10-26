# HealthLight Agentic AI: Your go-to Universal Health Companion

This repository contains the implementation of a **HealthLight Agentic AI**, designed to assist caregivers and patients with the following: natural language understanding, medical note summaries, action item extraction, and finding local providers.

It uses **LangGraph**, **LangChain**, and **Groq/ChatGPT LLMs** to build modular, agentic reasoning pipelines.

---

## Our Google Colab Notebook:

**[Link to HealthLight Agent with Gradio Web UI](https://colab.research.google.com/drive/1IPOwPDo9bpn5JcnG7zlqXx-k0W-7vQm1?usp=sharing#scrollTo=D6BIoI8qr9i8)**

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

   Copy the `.env.example` file in the project root and add your API keys.

---

## 🧠 Usage

Run the main entry point to start the Healthlight Agent!

```bash
python main.py
```

---

## 📂 Project Structure

```
langchain-for-good/
│
├── main.py                            # Entry point to run the full caregiver pipeline
├── agents/
│   └── caregiver_agent.py             # Defines the CaregiverCompanionAgent
│   └── provder_agent.py
├── graphs/
│   └── caregiver_graph.py             # Builds the LangGraph caregiving workflow
│   └── provider_graph.py
│   └── final_graph.py
├── orchestrators/
│   └── run_caregiver_graph.py         # Orchestrates pipeline execution
│   └── run_provider_graph.py
│   └── run_final_graph.py
├── pipelines/
│   └── cdc_retrieval_qa.py            # CDC knowledge retrieval QA chain
│   └── provider_json_retrieval.py     # Anthem Medi-Cal provider retrieval
├── requirements.txt                   # All dependencies
├── .env                               # API keys
└── README.md                          # Project setup and documentation
```

---

## Visual Studio Code QUICKSTART

### 1️⃣ Clone the repository

1. Open **VS Code**.

2. Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on Mac) to open the **Command Palette**.

3. Type **“Git: Clone”** and select it.

4. Paste the repository URL: **[https://github.com/ashloong/langchain-for-good.git](https://github.com/ashloong/langchain-for-good.git)**

5. Choose a local folder where you want the project saved.

6. When prompted, click **Open** after cloning completes.

### 2️⃣ Let VS Code set up

When the folder opens, VS Code will automatically:

- Create a local `.venv` virtual environment (if it doesn’t exist)

- Install all dependencies from `requirements.txt`

- Register the kernel as `.venv` for notebooks

Complete *(about 1-2 minutes)* when the terminal output contains:

    Terminal will be reused by tasks, press any key to close it.

### 3️⃣ Run notebooks or scripts

- Open any `.ipynb` file

- Click **Select Kernel** in the top-right corner and choose `.venv (Python)` from the list. VS Code will remember this selection automatically for future sessions.

- Press `Shift + Enter` to run a cell, or `Ctrl + F5` to run an entire script

---

**Code4Impact Hackathon: LangChain Agents for Social Good** 🌍
