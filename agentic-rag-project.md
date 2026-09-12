# RAGBot 

<img width="1600" height="790" alt="image" src="https://gist.github.com/user-attachments/assets/a5b3a872-8439-410e-9fb2-3d5ce4a64d6c" />



### Project Goal

Build and deploy a simple **Agentic Knowledge Assistant** that can:

1. Read a PDF
2. Search the PDF using RAG
3. Answer questions using GPT-OSS
4. Remember the conversation
5. Decide when to use a tool
6. Use a calculator
7. Search the knowledge base as a tool
8. Return a final answer
9. Run through Streamlit
10. Deploy to Streamlit Cloud
11. Later convert the workflow to LangGraph

---

# PART 1: PROJECT SETUP

## Step 1: Create the Project

Create the project folder:

```text
debot/
```

Open the folder in VS Code.

---

## Step 2: Create Virtual Environment

Open the terminal inside the project:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\Activate.ps1
```

Verify:

```powershell
python --version
```

Recommended:

```text
Python 3.11.x
```

### Checkpoint

Terminal should show:

```text
(venv) PS A:\Code\AI18\AgenticAI_TA01\debot>
```

---

# PART 2: INITIAL PROJECT FILES

## Step 3: Create `requirements.txt`

Create:

```text
requirements.txt
```

Add:

```text
streamlit
langchain
langchain-community
langchain-core
langchain-groq
langchain-text-splitters
faiss-cpu
sentence-transformers
torchvision
pypdf
python-dotenv
```

Install:

```powershell
pip install -r requirements.txt
```

---

## Step 4: Create `.gitignore`

Create:

```text
.gitignore
```

Add:

```text
venv/
.env
__pycache__/
*.py[cod]
.streamlit/secrets.toml
.DS_Store
Thumbs.db
```

### Important

Do **not** add:

```text
faiss_index/
```

because we want to commit our pre-built FAISS index to GitHub.

---

## Step 5: Create `.env`

Create:

```text
.env
```

Add:

```text
GROQ_API_KEY=your_groq_api_key
```

Do not commit this file to GitHub.

---

# PART 3: TEST GROQ

## Step 6: Create `test_groq.py`

Create:

```text
test_groq.py
```

Add:

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

response = llm.invoke(
    "Explain RAG in one simple sentence."
)

print(response.content)
```

Run:

```powershell
python test_groq.py
```

### Checkpoint

GPT-OSS should return an answer.

---

# PART 4: LOAD THE PDF

## Step 7: Create Data Folder

Create:

```text
data/
```

Put your document inside:

```text
data/
└── feature_engineering.pdf
```

Your current document is an **18-page feature engineering PDF**.

---

## Step 8: Create `test_pdf.py`

Create:

```text
test_pdf.py
```

Add:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "data/AI18-EDA-Revision-Session.pdf"
)

documents = loader.load()

print(f"Number of pages: {len(documents)}")

print("\nFirst page:\n")
print(documents[0].page_content[:2000])
```

Run:

```powershell
python test_pdf.py
```

### Checkpoint

You should see:

```text
Number of pages: 18
```

and extracted PDF text.

---

# PART 5: CHUNKING

## Step 9: Create `test_chunking.py`

Create:

```text
test_chunking.py
```

Add:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader(
    "data/AI18-EDA-Revision-Session.pdf"
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Number of pages: {len(documents)}")
print(f"Number of chunks: {len(chunks)}")

print("\nFirst chunk:\n")
print(chunks[0].page_content)

print("\nMetadata:")
print(chunks[0].metadata)
```

Run:

```powershell
python test_chunking.py
```

### Checkpoint

We should understand:

```text
PDF
 ↓
Pages
 ↓
Chunks
```

---

# PART 6: CREATE FAISS

## Step 10: Create `create_vectorstore.py`

Create:

```text
create_vectorstore.py
```

Add:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PDF_PATH = "data/AI18-EDA-Revision-Session.pdf"


loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"Loaded {len(documents)} pages.")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


print("Creating FAISS vector store...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created.")


vectorstore.save_local(
    "faiss_index"
)

print("FAISS index saved.")
```

Run:

```powershell
python create_vectorstore.py
```

---

## Step 11: Verify FAISS

You should now have:

```text
faiss_index/
├── index.faiss
└── index.pkl
```

### Checkpoint

Understand:

```text
PDF
 ↓
Chunks
 ↓
Embeddings
 ↓
FAISS Index
```

The expensive document processing happens **locally once**.

---

# PART 7: TEST RETRIEVAL

## Step 12: Create `test_retrieval.py`

Create:

```text
test_retrieval.py
```

Add:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


query = input(
    "\nAsk a question about the document: "
)

results = retriever.invoke(query)


print("\n" + "=" * 60)
print("RETRIEVED DOCUMENT CHUNKS")
print("=" * 60)


for i, document in enumerate(
    results,
    start=1
):

    print(f"\n--- Result {i} ---")

    print(
        f"Page: {document.metadata.get('page', 'Unknown')}"
    )

    print("\nContent:")

    print(document.page_content)
```

Run:

```powershell
python test_retrieval.py
```

Ask:

```text
What is feature engineering?
```

### Checkpoint

We should see relevant chunks from the PDF.

---

# PART 8: BUILD BASIC RAG

## Step 13: Create `rag.py`

Create:

```text
rag.py
```

This file contains the basic RAG logic.

Use:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def create_rag_chain():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are AtomBot, a helpful assistant.

        Use only the provided document context
        to answer the question.

        If the answer is not available in the context,
        say:

        "I could not find this information in the document."

        Do not make up information.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )


    def ask(question):

        documents = retriever.invoke(question)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        response = llm.invoke(
            prompt.format(
                context=context,
                question=question
            )
        )

        return response.content, documents


    return ask
```

---

# PART 9: TEST BASIC RAG

## Step 14: Create `test_rag.py`

```text
test_rag.py
```

Add:

```python
from rag import create_rag_chain


ask = create_rag_chain()


question = input(
    "\nAsk a question about the document: "
)


answer, documents = ask(question)


print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(answer)


print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for i, document in enumerate(
    documents,
    start=1
):

    print(
        f"Source {i} | "
        f"Page {document.metadata.get('page', 'Unknown')}"
    )
```

Run:

```powershell
python test_rag.py
```

Test:

```text
What is feature engineering?
```

### Checkpoint 1

At this point we have built:

```text
PDF
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS
 ↓
Retriever
 ↓
Context
 ↓
GPT-OSS
 ↓
Answer
```

**Stop here and understand RAG before continuing.**

---

# PART 10: STREAMLIT RAG

## Step 15: Create `app.py`

Create:

```text
app.py
```

Add:

```python
import streamlit as st
from rag import create_rag_chain


st.set_page_config(
    page_title="AtomBot",
    page_icon="🤖",
    layout="centered"
)


st.title("AtomBot")

st.caption(
    "Knowledge Assistant"
)


@st.cache_resource
def load_rag():

    return create_rag_chain()


ask = load_rag()


question = st.chat_input(
    "Ask a question about the document..."
)


if question:

    with st.chat_message("user"):
        st.write(question)


    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, documents = ask(question)

        st.write(answer)

        with st.expander("View Sources"):

            for i, document in enumerate(
                documents,
                start=1
            ):

                page = document.metadata.get(
                    "page",
                    "Unknown"
                )

                if isinstance(page, int):
                    page += 1

                st.markdown(
                    f"**Source {i} | Page {page}**"
                )

                st.write(
                    document.page_content
                )
```

Run:

```powershell
streamlit run app.py
```

### Checkpoint

We now have a working RAG application.

---

# PART 11: ADD CONVERSATION HISTORY

## Step 16: Upgrade `rag.py`

Update the `ask()` function so it accepts:

```python
def ask(question, history=""):
```

Add conversation history to the prompt:

```text
Conversation history:
{history}
```

```py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def create_rag_chain():
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are AtomBot, a helpful assistant that answers questions
        using the provided document.

        Use the document context and conversation history to answer
        the user's question.

        If the answer is not available in the document, say:
        "I could not find this information in the document."

        Do not make up information.

        Conversation history:
        {history}

        Document context:
        {context}

        Current question:
        {question}

        Answer:
        """
    )

    def ask(question, history=""):
        documents = retriever.invoke(question)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        response = llm.invoke(
            prompt.format(
                history=history,
                context=context,
                question=question
            )
        )

        return response.content, documents

    return ask
```

The RAG system becomes:

```text
Question
 +
Conversation History
 +
Retrieved Context
 ↓
GPT-OSS
 ↓
Answer
```



## Step 17: Update `app.py`

Use:

```python
st.session_state.messages
```

to store previous messages.

The interface becomes:

```text
User:
What is feature engineering?

AtomBot:
...

User:
Why is it important?

AtomBot:
...
```

```py
import streamlit as st
from rag import create_rag_chain


st.set_page_config(
    page_title="AtomBot",
    page_icon="🤖",
    layout="centered"
)


st.title("AtomBot")
st.caption("Agentic Research & Knowledge Assistant")


@st.cache_resource
def load_rag():
    return create_rag_chain()


ask = load_rag()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input(
    "Ask a question about the document..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    history = ""

    for message in st.session_state.messages[:-1]:
        history += (
            f'{message["role"].capitalize()}: '
            f'{message["content"]}\n'
        )

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, documents = ask(
                question,
                history
            )

        st.write(answer)

        with st.expander("View Sources"):

            for i, document in enumerate(
                documents,
                start=1
            ):

                page = document.metadata.get(
                    "page",
                    "Unknown"
                )

                if isinstance(page, int):
                    page = page + 1

                st.markdown(
                    f"**Source {i} | Page {page}**"
                )

                st.write(
                    document.page_content
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
 ```


### Checkpoint 2

understand:

**RAG + conversation history = conversational RAG**

---

# PART 12: INTRODUCE AGENTS

## Step 18: RAG vs Agent


### RAG

```text
Question
 ↓
Search document
 ↓
Answer
```

### Agent

```text
Question
 ↓
Think about what is needed
 ↓
Choose action/tool
 ↓
Execute
 ↓
Observe result
 ↓
Answer
```

Key teaching statement:

> RAG gives the model knowledge. An agent can decide what action or tool to use.

---

# PART 13: CREATE THE FIRST TOOL

## Step 19: Create `tools.py`

Create:

```text
tools.py
```

Add:

```python
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:

        return "Could not calculate the expression."
```

---

## Step 20: Create `test_tools.py`

```text
test_tools.py
```

Add:

```python
from tools import calculator


result = calculator.invoke(
    {
        "expression": "125 * 48"
    }
)


print("Result:", result)
```

Run:

```powershell
python test_tools.py
```

Expected:

```text
Result: 6000
```

---

# PART 14: CONNECT GPT-OSS TO TOOLS

## Step 21: Create `agent.py`

Create:

```text
agent.py
```

Add:

```python
from langchain_groq import ChatGroq
from tools import calculator
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


tools = [
    calculator
]


llm_with_tools = llm.bind_tools(
    tools
)


def run_agent(question):

    response = llm_with_tools.invoke(
        question
    )

    return response
```

---

## Step 22: Create `test_agent.py`

```text
test_agent.py
```

Add:

```python
from agent import run_agent


question = input(
    "\nAsk something: "
)


response = run_agent(
    question
)


print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)

print(response.content)


print("\n" + "=" * 60)
print("TOOL CALLS")
print("=" * 60)

print(response.tool_calls)
```

Run:

```powershell
python test_agent.py
```

Ask:

```text
What is 125 * 48?
```

### Checkpoint

You should see GPT-OSS selecting:

```text
calculator
```

```py
Ask something: What is 125 * 48?

============================================================
MODEL RESPONSE
============================================================


============================================================
TOOL CALLS
============================================================
[{'name': 'calculator', 'args': {'expression': '125 * 48'}, 'id': 'fc_e11614a4-ef4b-4375-8f32-9ada89f35a59', 'type': 'tool_call'}]
```


# PART 15: TURN RAG INTO A TOOL

## Step 23: Update `tools.py`

Now `tools.py` contains **two tools**:

```text
calculator
knowledge_search
```

Use:

```python
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:

        return "Could not calculate the expression."


@tool
def knowledge_search(query: str) -> str:
    """Search the feature engineering document."""

    documents = retriever.invoke(query)

    if not documents:

        return "No relevant information was found."


    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    return context
```

---

# PART 16: GIVE BOTH TOOLS TO GPT-OSS

## Step 24: Update `agent.py`

Import:

```python
from tools import calculator, knowledge_search
```

Define:

```python
tools = [
    calculator,
    knowledge_search
]
```

Bind them:

```python
llm_with_tools = llm.bind_tools(
    tools
)
```

```py
from langchain_groq import ChatGroq
from tools import calculator, knowledge_search
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


tools = [
    calculator,
    knowledge_search
]


llm_with_tools = llm.bind_tools(tools)


def run_agent(question):

    response = llm_with_tools.invoke(question)

    return response
```

---

## Step 25: Test Tool Selection

Run:

```powershell
python test_agent.py
```

Ask:

```text
What is feature engineering?
```

Expected tool:

```text
knowledge_search
```

Then:

```text
What is 125 * 48?
```

Expected tool:

```text
calculator
```


### Checkpoint 3

We have now seen:

```text
                  ┌── Calculator
                  │
Question → GPT-OSS
                  │
                  └── Knowledge Search
                            ↓
                           FAISS
```

---

# PART 17: BUILD THE AGENT LOOP

## Step 26: Update `agent.py`

The final `agent.py` should execute the selected tool and send its result back to GPT-OSS.

Core flow:

```text
User Question
      ↓
GPT-OSS
      ↓
Tool Call
      ↓
Execute Tool
      ↓
Tool Result
      ↓
GPT-OSS
      ↓
Final Answer
```

The important code concepts are:

```python
response.tool_calls
```

then:

```python
tool.invoke(...)
```

then:

```python
ToolMessage(...)
```

and finally:

```python
llm.invoke(messages)
```


```py
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from tools import calculator, knowledge_search
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


tools = [
    calculator,
    knowledge_search
]

tool_map = {
    "calculator": calculator,
    "knowledge_search": knowledge_search
}

llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """
You are AtomBot, a helpful assistant.

You have access to exactly two tools:

1. calculator
2. knowledge_search

Use these tools when necessary.

Do not use web search or any other external tool.

After receiving a tool result, answer the user's question
using that result.

When providing the final answer, do not call any tool.
Return only the final answer in normal text.
"""


def run_agent(question):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]

    response = llm_with_tools.invoke(messages)

    if not response.tool_calls:
        return response

    messages.append(response)

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        tool = tool_map[tool_name]

        tool_result = tool.invoke(tool_args)

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )

    final_response = llm.invoke(messages)

    return final_response
```
### Checkpoint 4

Test:

```text
What is 125 * 48?
```

Expected:

```text
6000
```

Test:

```text
What is feature engineering?
```

Expected:

A response based on the PDF.

---

# PART 18: CONNECT THE AGENT TO STREAMLIT

## Step 27: Update `app.py`

Instead of:

```python
from rag import create_rag_chain
```

use:

```python
from agent import run_agent
```

The application now becomes:

```text
Streamlit
   ↓
run_agent()
   ↓
GPT-OSS
   ↓
Tool Selection
   ↓
Tool
   ↓
Tool Result
   ↓
GPT-OSS
   ↓
Final Answer
```


## Step 28: Add Chat History to `app.py`

Create:

```python
st.session_state.messages
```

Store:

```text
user message
assistant message
```

Display them using:

```python
st.chat_message()
```

### Checkpoint 5

The complete local application works through Streamlit.

```py
import streamlit as st
from agent import run_agent


st.set_page_config(
    page_title="AtomBot",
    page_icon="🤖",
    layout="centered"
)


st.title("AtomBot")
st.caption("Agentic Research & Knowledge Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input(
    "Ask a question..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)


    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = run_agent(question)

        st.write(answer.content)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
    
```

---

# PART 19: CLEAN THE PROJECT

## Step 29: Keep the Core Files

The main project should now be:

```text
debot/
│
├── app.py
├── agent.py
├── tools.py
├── requirements.txt
├── .gitignore
├── .env
│
├── data/
│   └── feature_engineering.pdf
│
└── faiss_index/
    ├── index.faiss
    └── index.pkl
```

The testing files can remain during development:

```text
test_groq.py
test_pdf.py
test_chunking.py
test_retrieval.py
test_rag.py
test_tools.py
test_agent.py
```

They do not need to be part of the final teaching deployment if you want the repository to look cleaner.

---

# PART 20: DEPLOYMENT

## Step 30: Create `runtime.txt`

Create:

```text
runtime.txt
```

Add:

```text
python-3.12
```

---

## Step 31: Check `.gitignore`

Make sure it contains:

```text
venv/
.env
__pycache__/
*.py[cod]
.streamlit/secrets.toml
.DS_Store
Thumbs.db
```

**Do not ignore:**

```text
faiss_index/
```

---

## Step 32: Push to GitHub

Run:

```powershell
git status
```

Check that `.env` is not included.

Then:

```powershell
git add .
```

```powershell
git commit -m "Add AtomBot agentic assistant"
```

```powershell
git push
```

---

# PART 21: STREAMLIT CLOUD

## Step 33: Create Streamlit Cloud App

Open Streamlit Community Cloud.

Select:

```text
Create app
```

Choose:

```text
GitHub Repository
```

Select your repository.

Main file:

```text
app.py
```

Python:

```text
3.12
```

---

## Step 34: Add Groq Secret

In Streamlit Cloud:

```text
Settings
 ↓
Secrets
```

Add:

```toml
GROQ_API_KEY = "your_actual_api_key"
```

Never put the key in GitHub.

---

## Step 35: Deploy

Click deploy.

The deployment flow is:

```text
GitHub
   ↓
Streamlit Cloud
   ↓
Install requirements
   ↓
Load embedding model
   ↓
Load pre-built FAISS
   ↓
Load GPT-OSS
   ↓
Run app.py
```

# New Data Source

Instead of:

```text
PDF
 ↓
Pages
 ↓
Chunks
 ↓
Embeddings
 ↓
FAISS
```

we will have:

```text
train.jsonl
 ↓
JSON records
 ↓
Documents
 ↓
Embeddings
 ↓
FAISS
```

Our `train.jsonl` contains records like:

```json
{"instruction": "What does the bootcamp cover in terms of curriculum?", "input": "", "output": "Our bootcamp explores cutting-edge data science technologies..."}
```

Each JSON object will become one searchable document.


# Step 1: Change the Data Folder

Remove the PDF from the project or simply stop using it.

The structure becomes:

```text
bot/
│
├── app.py
├── agent.py
├── tools.py
├── requirements.txt
├── .gitignore
├── .env
│
├── data/
│   └── train.jsonl
│
└── faiss_index/
    ├── index.faiss
    └── index.pkl
```

Put your provided data into:

```text
data/train.jsonl
```


# Step 2: We No Longer Need PDF Loading

We do **not** need:

```python
PyPDFLoader
```

We also do not need:

```python
RecursiveCharacterTextSplitter
```

because every JSONL record is already a relatively small knowledge unit.

For example:

```text
Question:
What is atomcamp?

Answer:
We are a young ed-tech platform...
```

is already a good searchable document.


# Step 3: Update `create_vectorstore.py`

This is the **main file we need to change**.

Replace everything inside:

```text
create_vectorstore.py
```

with:

```python
import json

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


JSONL_PATH = "data/train.jsonl"


print("Loading JSONL dataset...")


documents = []


with open(
    JSONL_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line_number, line in enumerate(
        file,
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        data = json.loads(line)

        instruction = data.get(
            "instruction",
            ""
        )

        input_text = data.get(
            "input",
            ""
        )

        output = data.get(
            "output",
            ""
        )


        content = f"""
Question:
{instruction}

Additional Input:
{input_text}

Answer:
{output}
"""


        documents.append(
            Document(
                page_content=content,
                metadata={
                    "record": line_number,
                    "instruction": instruction
                }
            )
        )


print(
    f"Loaded {len(documents)} records."
)


print("Loading embedding model...")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


print("Embedding model loaded.")


print("Creating FAISS vector store...")


vectorstore = FAISS.from_documents(
    documents,
    embeddings
)


print("FAISS vector store created.")


vectorstore.save_local(
    "faiss_index"
)


print("FAISS index saved to 'faiss_index/'")
```


# Step 4: Rebuild the FAISS Index

This step is **very important**.

The existing:

```text
faiss_index/
```

was created from the PDF.

Therefore, we need to delete the old index first.

Delete:

```text
faiss_index/
```

Then run:

```powershell
python create_vectorstore.py
```

You should see something similar to:

```text
Loading JSONL dataset...
Loaded 28 records.
Loading embedding model...
Embedding model loaded.
Creating FAISS vector store...
FAISS vector store created.
FAISS index saved to 'faiss_index/'
```

The exact number will depend on how many records are in the actual file.


# Step 5: Check the New FAISS Index

We should again have:

```text
faiss_index/
├── index.faiss
└── index.pkl
```

But now the index contains the **Atomcamp JSONL knowledge**.


# Step 6: Test Retrieval

Before testing the agent, let's make sure the new data actually works.

Update:

```text
test_retrieval.py
```

to:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


query = input(
    "\nAsk a question: "
)


results = retriever.invoke(query)


print("\n" + "=" * 60)
print("RETRIEVED KNOWLEDGE")
print("=" * 60)


for i, document in enumerate(
    results,
    start=1
):

    print(f"\n--- Result {i} ---")

    print(
        f"Record: "
        f"{document.metadata.get('record', 'Unknown')}"
    )

    print("\nContent:")

    print(document.page_content)
```

Run:

```powershell
python test_retrieval.py
```

Test:

```text
What courses are offered?
```

should retrieve the record:

```text
Question:
What courses are offered?

Answer:
1. Data Science and AI...
2. Data Analytics...
...
```


# Step 7: Test Another Question

Try:

```text
What is the Agentic AI Bootcamp?
```

It should retrieve:

```text
Question:
Agentic AI Bootcamp

Answer:
If you're feeling lost and unsure...
```

Try:

```text
Is attendance mandatory?
```

It should retrieve:

```text
Question:
Is attendance mandatory?

Answer:
Absolutely! Active participation is crucial...
```

# Step 8: Test the Agent

Now run:

```powershell
python test_agent.py
```

Try:

```text
What does the bootcamp cover in terms of curriculum?
```

The model should select:

```text
knowledge_search
```

Then try:

```text
What is 125 * 48?
```

The model should select:

```text
calculator
```

Then:

```text
Who has developed you?
```

It should select:

```text
knowledge_search
```

and return information about Sagar.


# Step 9: Test Streamlit

Run:

```powershell
streamlit run app.py
```

Try these questions:

### Knowledge question

```text
What courses are offered?
```

### Knowledge question

```text
What is the Data Science Bootcamp curriculum?
```

### Knowledge question

```text
Is attendance mandatory?
```

### Knowledge question

```text
What are the prerequisites?
```

### Calculator

```text
What is 250 * 48?
```

### Another knowledge question

```text
Tell me about the Agentic AI Bootcamp.
```


# What Changed?

Very little.

### Before

```text
create_vectorstore.py
        ↓
PDF
        ↓
PyPDFLoader
        ↓
Chunks
        ↓
FAISS
```

### Now

```text
create_vectorstore.py
        ↓
train.jsonl
        ↓
JSON records
        ↓
Documents
        ↓
FAISS
```

Everything after FAISS remains:

```text
FAISS
 ↓
knowledge_search
 ↓
Agent
 ↓
GPT-OSS
 ↓
Streamlit
```


# Final Project Structure

After the change:

```text
bot/
│
├── app.py
├── agent.py
├── tools.py
├── create_vectorstore.py
├── test_retrieval.py
├── test_tools.py
├── test_agent.py
│
├── requirements.txt
├── runtime.txt
├── .gitignore
├── .env
│
├── data/
│   └── train.jsonl
│
└── faiss_index/
    ├── index.faiss
    └── index.pkl
```

###  point to be noted

The knowledge source can be:

```text
PDF
JSON
JSONL
CSV
Database
Website
API
```

while the **agent layer stays the same**.

