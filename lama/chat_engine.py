import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Simulated user profiles (normally comes from backend)
from mock_users import USER_PROFILES  # Your mock user file

st.set_page_config(page_title="Personalized Sports Chatbot", layout="centered")
st.title("🏅 Personalized Sports Chatbot")

# --- Simulate login ---
selected_user = st.selectbox(
    "Select user",
    list(USER_PROFILES.keys()),
    format_func=lambda key: USER_PROFILES[key]["username"]
)

user_profile = USER_PROFILES[selected_user]
about_summary = f"""
You are a concise, smart, and context-aware assistant who gives sharp, relevant replies only.

This user is a sports coach. They specialize in: **{user_profile['sport']}**.

Here’s what the user said about themselves:
---
{user_profile['details']}
---
Use this info to personalize your tone, advice, examples, and especially team-specific responses.
If they ask about "my team", infer from the text above.

Do not give general explanations. Focus only on what they ask.
Keep answers short and inline unless explicitly asked for depth.
"""

# --- Session memory per user ---
if "user_memory" not in st.session_state:
    st.session_state.user_memory = {}
if selected_user not in st.session_state.user_memory:
    st.session_state.user_memory[selected_user] = []

# --- LangChain setup ---
llm = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")
output_parser = StrOutputParser()

def generate_response(messages):
    prompt = ChatPromptTemplate.from_messages([
        ("system", about_summary)
    ] + [(msg["role"], msg["content"]) for msg in messages])
    
    chain = prompt | llm | output_parser
    return chain.invoke({})

# --- Display message history ---
for msg in st.session_state.user_memory[selected_user]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input & reply ---
if prompt := st.chat_input("Ask your question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.user_memory[selected_user].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        response = generate_response(st.session_state.user_memory[selected_user])
        st.markdown(response)

    st.session_state.user_memory[selected_user].append({"role": "assistant", "content": response})
