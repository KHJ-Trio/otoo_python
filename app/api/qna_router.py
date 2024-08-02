# api/emotion_report_router.py
import logging
from fastapi import APIRouter, Depends, HTTPException
from models.models import QnaModel, get_qna_model, QnAEditorModel, get_qna_editor_model
from modules.module_qna import messagesRequest, generate_qna_response, generate_qna_edit_response

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

router = APIRouter()

@router.post("/qna")
async def qnaLLM(messages_request: messagesRequest, model: QnaModel = Depends(get_qna_model)):
    try:
        response = await generate_qna_response(model, messages_request)
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
    
@router.post("/qna-edit")
async def qnaEdit(messages_request: messagesRequest, model: QnAEditorModel = Depends(get_qna_editor_model)):
    try:
        response = await generate_qna_edit_response(model, messages_request)
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