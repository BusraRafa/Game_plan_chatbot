from django.utils import timezone
from chat.models import Chat, Message
from about.models import About
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from django.contrib.auth import get_user_model

User = get_user_model()

# LLM setup
#llama3.2:3b
llm = ChatOllama(model="llama3.2:3b-instruct-q4_K_M", 
                 base_url="http://localhost:11434",
                 temperature=0.7,
                 top_k=40,
                 top_p=0.9,
                 repeat_penalty=1.1,
                 num_ctx=4096)
output_parser = StrOutputParser()

def generate_response_from_chat(chat, user, user_input): 
    # 1. Get About info
    try:
        about = user.about
    except About.DoesNotExist:
        return "User profile missing. Please complete your About section.", ""

    # 2. Build personalized system message
    system_message = f"""
    You are a detailed and intelligent sports assistant — like a personal sports analyst — designed to support coaches with insightful updates and tailored guidance. Your job is to respond conversationally — **like ChatGPT normally does**, speaking in a natural (not like a journalist), but with rich detail — just like ESPN or NBA.com — when updating about sports players, teams, or performance.
    
    The user is a sports coach who specializes in: **{about.sport_coach}.**.
Here’s what the user said about themselves:
    
    ---
    {about.details}
    ---
    Use this info to personalize your answers. When they ask about:
- a **player**, include their recent performance, season stats, injuries, leadership role, and how the coach can learn from them.
- a **team**, summarize their recent matches, standings, highlights, key players, and challenges.
- a **strategy** or **coaching help**, provide focused, relevant suggestions with professional-level insight.

    Avoid sounding like a news presenter or reporter. Respond like a helpful assistant or analyst who knows the user’s interest the latest and shares it clearly and casually.Be casual, insightful, and sport-specific.
    """

    # 3. Collect chat history
    chat_history = [("system", system_message)]
    messages = Message.objects.filter(chat=chat).order_by("timestamp")

    for msg in messages:
        role = "user" if msg.sender == user else "assistant"
        chat_history.append((role, msg.content))

    # 4. Append current message
    chat_history.append(("user", user_input.strip()))

    # 5. Generate response
    prompt = ChatPromptTemplate.from_messages(chat_history)
    chain = prompt | llm | output_parser
    try:
        response = chain.invoke({}).strip()
    except Exception as e:
        return f"[AI Error]: {str(e)}", ""

    # 6. Save both messages
    bot_user, _ = User.objects.get_or_create(username="chatbot")
    chat.participants.add(bot_user)
    Message.objects.create(chat=chat, sender=user, content=user_input.strip())
    Message.objects.create(chat=chat, sender=bot_user, content=response)

    # 7. Update chat duration
    chat.total_chat_duration = timezone.now() - chat.created_at

    # 8. Auto-summary
    full_chat_text = "\n".join(
        f"{'User' if msg.sender == user else 'Assistant'}: {msg.content}"
        for msg in messages
    ) + f"\nUser: {user_input.strip()}\nAssistant: {response}"

    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this conversation into a single paragraph. Focus on what the user asked, what they were interested in, and what the assistant provided:\n\n{chat}"
    )
    summary_chain = summary_prompt | llm | output_parser
    try:
        summary_text = summary_chain.invoke({"chat": full_chat_text}).strip()
        chat.topic_summary = summary_text
        chat.save()
    except Exception:
        pass  # Fail silently if summarization fails

    return response, full_chat_text