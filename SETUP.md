## Quick Start (Visual Studio Code)

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