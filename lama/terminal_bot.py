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
llm = ChatOllama(model="llama3.2:3b", 
                 base_url="http://localhost:11434",
                 temperature=0.7,
                 top_k=40,
                 top_p=0.9,
                 repeat_penalty=1.1,
                 num_ctx=4096
                 )
output_parser = StrOutputParser()



#def chat(chat_history dict = none,user_input,  )

# --- Select user (simulate login) ---
print("Available users:")
for key, user in USER_PROFILES.items():
    print(f"{key}: {user['username']}")
selected_user = input("Enter user key (e.g., 'user1'): ").strip()

if selected_user not in USER_PROFILES:
    print("Invalid user key. Exiting.")
    exit()

user_profile = USER_PROFILES[selected_user]
chat_history = []

#----------for json--------------
chat_log={
    "username":user_profile["username"],
    "sport":user_profile["sport"],
    "details":user_profile["details"],
    "timestamp":datetime.now().isoformat(),
    "chat_details":[]
}


# --- Personalized system message ---
system_message = f"""
You are a top-tier sports analyst chatbot that gives real-time style, detailed updates like ESPN or NBA.com would. 
The user is a sports coach specializing in: {about.sport_coach}.
Here’s what the user said about themselves:
---
{about.details}
---
When they ask about a player (like Stephen Curry) or team (like Golden State Warriors), provide:

- Recent performance (with game stats and outcomes)
- Latest news (injuries, trades, form)
- Role in the team and leadership
- Season highlights and playoff hopes

Speak with confident, sports-journalist tone — like you're reporting on live TV. Always tailor the update to what the user cares about (coaching, leadership, player development). 
If they ask for suggestions, offer coaching-level strategic insights based on their favorite team's style.
Be sharp, insightful, and passionate — like a seasoned NBA insider.
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
