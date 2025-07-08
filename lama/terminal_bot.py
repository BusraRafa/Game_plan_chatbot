import sys
import time
import threading
#for json
import json
from datetime import datetime
import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mock_users import USER_PROFILES 
#for memory summary
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

# --- model ---
#ollama pull mistral:7b-instruct-q4_K_M
# --- Summarize user details ---
# Summarizer model
# --- Model Initialization ---
llm = ChatOllama(
    model="mistral:7b-instruct-v0.3-q3_K_M",
    base_url="http://localhost:11434",
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repeat_penalty=1.1,
    num_ctx=4096
)

# Summarizer model (lighter settings)
summarizer = ChatOllama(
    model="mistral:7b-instruct-v0.3-q3_K_M",
    base_url="http://localhost:11434",
    temperature=0.3,
    top_k=20,
    top_p=0.85,
    repeat_penalty=1.05,
    num_ctx=1024
)

output_parser = StrOutputParser()

# --- Select user (simulate login) ---
print("Available users:")
for key, user in USER_PROFILES.items():
    print(f"{key}: {user['username']}")

selected_user = input("Enter user key (e.g., 'user1'): ").strip()

if selected_user not in USER_PROFILES:
    print("Invalid user key. Exiting.")
    exit()

user_profile = USER_PROFILES[selected_user]

# --- Summarize user profile ---
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize this user profile in 2 sentences, preserving coaching values and favorite players:\n\n{details}"
)
summary_chain = summary_prompt | summarizer | output_parser
short_user_details = summary_chain.invoke({"details": user_profile["details"]})

#print(f"\n--- User profile summary ---\n{short_user_details}\n")

chat_history = []

# ----------for json--------------
chat_log = {
    "username": user_profile["username"],
    "sport": user_profile["sport"],
    "details": user_profile["details"],
    "timestamp": datetime.now().isoformat(),
    "chat_details": []
} 

# --- Personalized system message ---
system_message = f"""
You are a detailed and intelligent sports assistant — like a personal sports analyst — designed to support coaches with insightful updates and tailored guidance. Your job is to respond conversationally — **like ChatGPT normally does**,speaking in a natural (not like a journalist), but with rich detail — just like ESPN or NBA.com — when updating about sports players, teams, or performance.

The user is a sports coach who specializes in: **{user_profile['sport']}**.
Here’s what the user said about themselves (summarized):
---
{short_user_details}
---

Use this info to personalize your answers. When they ask about:
- a **player**, include their recent performance, season stats, injuries, leadership role, and how the coach can learn from them.
- a **team**, summarize their recent matches, standings, highlights, key players, and challenges.
- a **strategy** or **coaching help**, provide focused, relevant suggestions with professional-level insight.

Do NOT start with "Coming to you live..." or anything overly theatrical. Avoid sounding like a news presenter or reporter. Respond like a helpful assistant or analyst who knows the user’s interest the latest and shares it clearly and casually.  
Keep answers structured, insightful, and sport-specific.

If the user asks for suggestions, give advice tailored to how they coach and what they value (teamwork, discipline, leadership, etc.).
"""

print(f"\n🟢 Chat started for: {user_profile['username']}")
print("Type 'exit' to quit.\n")

stop_thinking = False

def show_thinking_animation():
    while not stop_thinking:
        for dots in [".  ", ".. ", "..."]:
            if stop_thinking:
                break
            sys.stdout.write(f"\rThinking{dots}")
            sys.stdout.flush()
            time.sleep(0.4)
    sys.stdout.write("\r" + " " * 30 + "\r") 

# --- Chat loop ---
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit","bye"]:
        print("Goodbye! 👋")
        break

    chat_history.append(("user", user_input))
    chat_log["chat_details"].append({"role":"user","content":user_input})
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_message)] + chat_history
    )
    chain = prompt | llm | output_parser

    # Start thinking animation in background
    stop_thinking = False
    t = threading.Thread(target=show_thinking_animation)
    t.start()

    # Call the model
    response = chain.invoke({})

    # Stop animation
    stop_thinking = True
    t.join()

    print(f": {response.strip()}\n")
    chat_history.append(("assistant", response.strip()))
    chat_log["chat_details"].append({"role":"assistant","content":response.strip()})
    
# --- FINAL SUMMARY USING THE MODEL ---
formatted_chat = "\n".join(
    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_log["chat_details"]]
)

summary_prompt = ChatPromptTemplate.from_template(
    "Summarize this conversation into a single paragraph. Focus on what the user asked, what they were interested in, and what the assistant provided:\n\n{chat}"
)
summary_chain = summary_prompt | llm | output_parser
summary_text = summary_chain.invoke({"chat": formatted_chat}).strip()

chat_log["summary"] = summary_text

print("\n Generated Summary:\n")
print(summary_text)

#saving json file
filename=f"chat_logs/{user_profile['username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
os.makedirs("chat_logs",exist_ok=True)

with open(filename,"w",encoding="utf-8") as f:
    json.dump(chat_log,f,ensure_ascii=False,indent=2)
    
print(f"\n the chat is saved to the {filename}")
