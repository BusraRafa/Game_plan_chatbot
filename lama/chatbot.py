import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.title("llama-3.2 Chat-bot")

model = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")


def generate_response(messages):
    chat_template = ChatPromptTemplate.from_messages(messages)
    chain = chat_template | model | StrOutputParser()
    return chain.invoke({})


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    
    with st.chat_message("assistant"), st.spinner("Generating response..."):
        response = generate_response(st.session_state.messages)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
