import sys
import time
import threading
import json
from datetime import datetime
import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mock_users import USER_PROFILES 

# --- Model setup ---
llm = ChatOllama(
    model="mistral:7b-instruct-q4_K_M", 
    base_url="http://localhost:11434",
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repeat_penalty=1.1,
    num_ctx=4096
)
output_parser = StrOutputParser()

# --- Select user ---
print("Available users:")
for key, user in USER_PROFILES.items():
    print(f"{key}: {user['username']}")
selected_user = input("Enter user key (e.g., 'user1'): ").strip()

if selected_user not in USER_PROFILES:
    print("Invalid user key. Exiting.")
    exit()

user_profile = USER_PROFILES[selected_user]
chat_history = []
chat_summary = ""  # Running summary

# --- JSON logging ---
chat_log = {
    "username": user_profile["username"],
    "sport": user_profile["sport"],
    "details": user_profile["details"],
    "timestamp": datetime.now().isoformat(),
    "chat_details": []
}

# --- Base system prompt ---
system_message = f"""
You are a detailed and intelligent sports assistant — like a personal sports analyst — designed to support coaches with insightful updates and tailored guidance. Your job is to respond conversationally — **like ChatGPT normally does**, speaking in a natural (not like a journalist), but with rich detail — just like ESPN or NBA.com — when updating about sports players, teams, or performance.

The user is a sports coach who specializes in: **{user_profile['sport']}**.
Here’s what the user said about themselves:
---
{user_profile['details']}
---

Use this info to personalize your answers. When they ask about:
- a **player**, include their recent performance, season stats, injuries, leadership role, and how the coach can learn from them.
- a **team**, summarize their recent matches, standings, highlights, key players, and challenges.
- a **strategy** or **coaching help**, provide focused, relevant suggestions with professional-level insight.

Avoid sounding like a news presenter. Be casual, insightful, and sport-specific.
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
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye! 👋")
        break

    chat_history.append(("user", user_input))
    chat_log["chat_details"].append({"role": "user", "content": user_input})

    # ⏳ Summarize if needed
    if len(chat_history) > 6:
        old_chat = "\n".join([f"{r[0]}: {r[1]}" for r in chat_history[:-4]])
        summary_prompt = ChatPromptTemplate.from_template(
        """Summarize this conversation so far. 
        Focus on what the user is asking, what the assistant is saying, and key insights to remember in future replies.

        {chat}"""
        )

        summary_chain = summary_prompt | llm | output_parser
        chat_summary = summary_chain.invoke({"chat": old_chat}).strip()

    # 🔁 Build updated system message
    dynamic_system_message = system_message
    if chat_summary:
        dynamic_system_message += f"\n\nSummary of earlier conversation:\n{chat_summary}"

    # 🧠 Only keep last 4–5 messages
    recent_history = chat_history[-5:]

    prompt = ChatPromptTemplate.from_messages(
        [("system", dynamic_system_message)] + recent_history
    )
    chain = prompt | llm | output_parser

    # Show animation
    stop_thinking = False
    t = threading.Thread(target=show_thinking_animation)
    t.start()

    # Get model output
    response = chain.invoke({})

    stop_thinking = True
    t.join()

    print(f": {response.strip()}\n")
    chat_history.append(("assistant", response.strip()))
    chat_log["chat_details"].append({"role": "assistant", "content": response.strip()})

# --- FINAL SUMMARY ---
formatted_chat = "\n".join([
    f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_log["chat_details"]
])

summary_prompt = ChatPromptTemplate.from_template(
    "Summarize this conversation into a single paragraph. Focus on what the user asked, what they were interested in, and what the assistant provided:\n\n{chat}"
)
summary_chain = summary_prompt | llm | output_parser
summary_text = summary_chain.invoke({"chat": formatted_chat}).strip()
chat_log["summary"] = summary_text
chat_log["intermediate_summary"] = chat_summary

print("\nGenerated Summary:\n")
print(summary_text)

# --- Save to JSON ---
filename = f"chat_logs/{user_profile['username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
os.makedirs("chat_logs", exist_ok=True)

with open(filename, "w", encoding="utf-8") as f:
    json.dump(chat_log, f, ensure_ascii=False, indent=2)

print(f"\nThe chat is saved to: {filename}")
