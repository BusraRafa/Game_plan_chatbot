from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from chat.models import Chat, Message
from about.models import About
from django.utils import timezone

from datetime import timedelta

# LLM 
llm = ChatOllama(model="llama3.2:3b", 
                 base_url="http://localhost:11434",
                 temperature=0.7,
                 top_k=40,
                 top_p=0.9,
                 repeat_penalty=1.1,
                 num_ctx=4096
                 )
output_parser = StrOutputParser()

def generate_response_from_chat(chat, user, user_input): 
    # 1. Getting About info 
    try:
        about = user.about
    except About.DoesNotExist:
        return "User profile missing. Please complete your About section.", ""

    # 2. system prompt based on user profile
        # ---- Personalized system prompt ----
    system_message = f"""
    You are a concise, smart, and context-aware assistant who gives sharp, relevant replies only.
    This user is a sports coach. They specialize in: **{about.sport_coach}**.
    Here’s what the user said about themselves:
    ---
    {about.details}
    ---
    Use this info to personalize your tone, advice, examples, and especially team-specific responses.
    If they ask about "my team", infer from the text above.
    Do not give general explanations. Focus only on what they ask.
    Keep answers short and inline unless explicitly asked for depth.
    """

    # 3. Loading existing chat history from Message table
    chat_history = [("system", system_message)]
    messages = Message.objects.filter(chat=chat).order_by("timestamp")

    for msg in messages:
        role = "user" if msg.sender == user else "assistant"
        chat_history.append((role, msg.content))

    # 4. Addding current user input
    chat_history.append(("user", user_input))

    # 5. Building LangChain pipeline
    prompt = ChatPromptTemplate.from_messages(chat_history)
    chain = prompt | llm | output_parser
    response = chain.invoke({}).strip()

    # 6. Saving both user and assistant messages to DB
    Message.objects.create(chat=chat, sender=user, content=user_input)
    Message.objects.create(chat=chat, sender=None, content=response)

    # 7. Updating the chat duration
    chat.total_chat_duration = timezone.now() - chat.created_at
    # 8. Auto-summary generation 
    full_chat_text = "\n".join(
        f"{'User' if msg.sender == user else 'Assistant'}: {msg.content}"
        for msg in Message.objects.filter(chat=chat).order_by("timestamp")
    )

    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this conversation into a single paragraph. Focus on what the user asked, what they were interested in, and what the assistant provided:\n\n{chat}"
    )
    summary_chain = summary_prompt | llm | output_parser
    summary_text = summary_chain.invoke({"chat": full_chat_text}).strip()

    chat.topic_summary = summary_text
    chat.save()  # saving both summary and duration


# 9. Return assistant reply + updated chat log in ******text**********
    updated_chat_log = f"{full_chat_text}\nUser: {user_input}\nAssistant: {response}"