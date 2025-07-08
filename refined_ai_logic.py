from django.utils import timezone
from chat.models import Chat, Message
from about.models import About
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from django.contrib.auth import get_user_model

User = get_user_model()

# LLM setup
llm = ChatOllama(model="llama3.2:3b", 
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