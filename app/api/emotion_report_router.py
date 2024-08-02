# api/emotion_report_router.py
import logging
from fastapi import APIRouter, Depends, HTTPException
from models.models import EmotionReportModel, get_emotion_report_model
from modules.module_emotionReport import messagesRequest, generate_messages_response

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

router = APIRouter()

@router.post("/emotionReport")
async def emotionReportLLM(messages_request: messagesRequest, model: EmotionReportModel = Depends(get_emotion_report_model)):
    try:
        response = await generate_messages_response(model, messages_request)
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