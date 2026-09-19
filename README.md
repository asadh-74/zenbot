🤖 AtomBot: Agentic Research & Knowledge Assistant
AtomBot is an Agentic Knowledge Assistant built using LangChain, FAISS, and Streamlit. It uses Retrieval-Augmented Generation (RAG) combined with an agentic tool-calling loop powered by Groq (GPT-OSS).

Unlike a standard RAG application, AtomBot acts as an Agent. It evaluates the user's question, decides whether to search its internal knowledge base or use a calculator, executes the tool, and formulates a final conversational response based on the tool's output and the conversation history.

🌟 Features
Agentic Tool Selection: Automatically decides between querying the knowledge base or using a calculator.

Conversational Memory: Remembers the context of the ongoing conversation for seamless follow-up questions.

Local Vector Database: Uses FAISS and HuggingFace embeddings (all-MiniLM-L6-v2) to store and retrieve document knowledge locally.

Flexible Data Ingestion: Configured to ingest .jsonl datasets (can easily be adapted for PDFs, CSVs, or databases).

Streamlit Web UI: A clean, easy-to-use chat interface.

Cloud Ready: Fully configured for deployment on Streamlit Community Cloud.

🛠️ Tech Stack
Language: Python 3.11 / 3.12

Framework: LangChain (Core, Community, Groq)

LLM: Groq (openai/gpt-oss-20b)

Embeddings: Sentence Transformers / HuggingFace

Vector Store: FAISS (CPU)

Frontend: Streamlit

📂 Project Structure
Plaintext
bot/
│
├── app.py                   # Main Streamlit application UI
├── agent.py                 # Core agent logic and tool binding
├── tools.py                 # Definitions for calculator and knowledge_search tools
├── create_vectorstore.py    # Script to convert data/train.jsonl into a FAISS index
├── test_retrieval.py        # CLI script to test FAISS retrieval
├── test_tools.py            # CLI script to test isolated tools
├── test_agent.py            # CLI script to test the agent loop without UI
│
├── requirements.txt         # Project dependencies
├── runtime.txt              # Specifies Python version (3.12) for Streamlit Cloud
├── .gitignore               # Git ignore rules
├── .env                     # Local environment variables (Groq API Key)
│
├── data/
│   └── train.jsonl          # Source knowledge dataset
│
└── faiss_index/             # Generated local vector database (committed to repo)
    ├── index.faiss
    └── index.pkl
🚀 Local Setup & Installation
1. Clone the repository and navigate to the project folder:

Bash
git clone <your-repo-url>
cd bot
2. Create and activate a virtual environment:

Bash
python -m venv venv

# Windows
venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate
3. Install dependencies:

Bash
pip install -r requirements.txt
4. Set up your Environment Variables:
Create a .env file in the root directory and add your Groq API key:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
(Note: Never commit your .env file to version control. It is already included in .gitignore)

🧠 Building the Knowledge Base
By default, the bot reads from data/train.jsonl. Every JSON object acts as a searchable document.

1. Add your data:
Ensure your data is placed in data/train.jsonl following this structure:

JSON
{"instruction": "What courses are offered?", "input": "", "output": "1. Data Science and AI..."}
2. Generate the FAISS Index:
Run the vector store creation script. This will process the JSONL file, generate embeddings, and save them to the faiss_index/ directory.

Bash
python create_vectorstore.py
Note: The faiss_index folder should be committed to GitHub so the cloud deployment doesn't have to rebuild it.

💻 Running the Application
To start the Streamlit web interface locally, run:

Bash
streamlit run app.py
You can test the agent's capabilities by asking:

Knowledge question: "What courses are offered?"

Calculator question: "What is 125 * 48?"

Conversational follow-up: "Can you tell me more about that?"

☁️ Deployment to Streamlit Community Cloud
This project is configured for easy deployment on Streamlit Cloud.

Commit and Push: Ensure all files (including runtime.txt, requirements.txt, and the faiss_index/ folder) are pushed to your GitHub repository.

Create App: Go to Streamlit Community Cloud and click Create app.

Connect Repository: Select your GitHub repository, set the branch, and choose app.py as the Main file.

Add Secrets: Before clicking deploy, go to Advanced settings (or App Settings -> Secrets) and add your Groq API key:

Ini, TOML
GROQ_API_KEY = "your_actual_api_key"
Deploy: Click deploy! Streamlit will install the requirements, load the pre-built FAISS index, and launch the application.
