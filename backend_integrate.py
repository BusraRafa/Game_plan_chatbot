from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.contrib.auth import get_user_model
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes

from chat.models import Chat, Message
from chat.ai_logic import generate_response_from_chat

User = get_user_model()

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def chat_with_assistant(request, chat_id):
    user = request.user
    user_input = request.data.get("message")
    if not user_input:
        return Response({"error": "No message provided."}, status=status.HTTP_400_BAD_REQUEST)

    # Get the chat ensuring user is a participant
    chat = get_object_or_404(Chat, id=chat_id, participants=user)

    try:
        # Call your AI logic function (returns reply and updated chat text)
        reply, chat_log = generate_response_from_chat(chat, user, user_input)

        # Ensure chatbot user is participant and save messages (already done in ai_logic)
        bot_user, _ = User.objects.get_or_create(username="chatbot")
        if bot_user not in chat.participants.all():
            chat.participants.add(bot_user)

        # Return AI reply and chat log
        return Response({
            "reply": reply,
            "chat_log": chat_log,
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": f"AI logic error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)