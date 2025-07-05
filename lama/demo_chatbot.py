# chat_engine.py
import requests
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Dummy token for dev (replace with real token from login)
DUMMY_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzUxNjUwMjkzLCJpYXQiOjE3NTE2NDY5OTMsImp0aSI6ImZjYWYwZTE1MDBiMjQzMTg5ZWIzZGM3MGZjZGM2NGVjIiwidXNlcl9pZCI6MX0.PFVGAlTNPgi47VJ8Mk6MzHQCfF_1B4C7lh6BivpLjuY"
API_BASE = "https://authenti-cation-system.vercel.app"
HEADERS = {"Authorization": f"JWT {DUMMY_TOKEN}"}

# LangChain model setup
llm = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")

# --- Fetch user about info ---
def get_user_about():
    try:
        res = requests.get(f"{API_BASE}/api/about/", headers=HEADERS)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print("Error fetching about:", e)
    return {"sport_coach": "", "details": ""}

# --- Create a new chat session ---
def create_chat_detail():
    payload = {"topic_summary": "Personalized coaching chat", "participants": []}
    try:
        res = requests.post(f"{API_BASE}/api/chat-detail/", json=payload, headers=HEADERS)
        if res.status_code == 201:
            return res.json()["id"]
    except Exception as e:
        print("Error creating chat session:", e)
    return None

# --- Save message to backend ---
def save_message(chat_id, content):
    payload = {"chat": chat_id, "content": content}
    try:
        requests.post(f"{API_BASE}/api/chat-history/", json=payload, headers=HEADERS)
    except Exception as e:
        print("Error saving message:", e)

# --- Build personalized prompt ---
def build_prompt(about_info, messages):
    sport = about_info.get("sport_coach", "sports")
    details = about_info.get("details", "No background info provided.")

    system_msg = (
        f"You are a helpful assistant specialized in {sport}.\n"
        f"The user said about themselves: {details}\n"
        "Use this context to personalize your sports coaching advice."
    )

    chat_template = ChatPromptTemplate.from_messages(
        [("system", system_msg)] + [(m["role"], m["content"]) for m in messages]
    )
    return chat_template | llm | StrOutputParser()

# --- Streamlit Interface ---
st.set_page_config(page_title="Personalized Sports Chatbot")
st.title("🏅 Personalized Sports Chatbot")

# Initialize state
if "chat_id" not in st.session_state:
    st.session_state.chat_id = create_chat_detail()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get About Info
about = get_user_about()

# DEBUG: Show about info in sidebar and print in terminal
st.sidebar.title("🔍 Debug Info")
st.sidebar.json(about)
print("DEBUG: About Info →", about)

if not about.get("sport_coach"):
    st.sidebar.warning("⚠️ No 'sport_coach' found — prompt will be generic.")
if not about.get("details"):
    st.sidebar.warning("⚠️ No 'details' found — chatbot won't be personalized.")

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask your sports question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.chat_id, prompt)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        response = build_prompt(about, st.session_state.messages).invoke({})
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(st.session_state.chat_id, response)
