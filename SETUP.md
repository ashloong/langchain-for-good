## Quick Start (Visual Studio Code)

### 1️⃣ Clone the repository

1. Open **VS Code**.

2. Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on Mac) to open the **Command Palette**.

3. Type **“Git: Clone”** and select it.

4. Paste the repository URL.

5. Choose a local folder where you want the project saved.

6. When prompted, click **Open** after cloning completes.

### 2️⃣ When the folder opens, VS Code will automatically:

- Create a local `.venv` virtual environment (if it doesn’t exist)

- Install all dependencies from `requirements.txt`

- Add Jupyter support (`notebook`, `ipykernel`, etc.)

- Register the kernel as **Python (.venv)** for notebooks

If prompted to “Run tasks on folder open,” click **Allow**.

You don’t need to manually run `pip install -r requirements.txt`,
VS Code handles it via `.vscode/tasks.json`.

### 3️⃣ Run notebooks or scripts

- Open any `.ipynb` file

- Make sure the selected kernel (top right) is **Python (.venv)**

- Press `Shift + Enter` to run a cell, or `Ctrl + F5` to run an entire script