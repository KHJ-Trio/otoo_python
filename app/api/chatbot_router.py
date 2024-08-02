# api/chatbot_router.py
import logging
from fastapi import APIRouter, Depends, HTTPException
from models.models import ChatbotModel, get_chatbot_model
from modules.module_chatbot import FullRequest, generate_chat_response

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

router = APIRouter()

@router.post("/chatbot", tags=["chatbot"])
async def chatbotLLM(full_request: FullRequest, model: ChatbotModel = Depends(get_chatbot_model)):
    try:
        mode_request = full_request.mode_request
        recent_messages_request = full_request.recent_messages_request
        response = await generate_chat_response(model, mode_request, recent_messages_request)
        return response
    except ValueError as e:
        logger.error("ValueError: %s", e)
        raise HTTPException(status_code=400, detail=f"입력 값 오류: {e}")
    except RuntimeError as e:
        logger.error("RuntimeError: %s", e)
        raise HTTPException(status_code=500, detail=f"런타임 오류 발생: {e}")
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail=f"예상치 못한 오류 발생: {e}")