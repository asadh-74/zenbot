import os
import streamlit as st
from agent import build_agent

st.set_page_config(page_title="AtomBot", page_icon="🤖")
st.title("🤖 AtomBot")
st.caption("Ask your PDF or JSONL knowledge base; calculate with a tool.")
if "GROQ_API_KEY" not in os.environ:
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

@st.cache_resource
def load_agent():
    return build_agent()

try:
    ask = load_agent()
except Exception as exc:
    st.error(str(exc))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
with st.sidebar:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
if question := st.chat_input("Ask a question..."):
    history = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer = ask(question, history)
        except Exception as exc:
            st.error(f"Could not answer: {exc}")
        else:
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
