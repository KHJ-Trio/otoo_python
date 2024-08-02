# api/analysis_router.py
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from models.models import AnalysisModel, get_analysis_model
from modules.module_analysis import analyze_text

class AnalysisRequest(BaseModel):
    text: str
    type: str  # 'conflict' 또는 'love' 또는 'friendship'

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

router = APIRouter()

@router.post("/analyze")
async def analyze(request: AnalysisRequest, model: AnalysisModel = Depends(get_analysis_model)):
    try:
        response = await analyze_text(model, request.text, request.type)
        return {"response": response}
    except ValueError as e:
        logger.error("ValueError: %s", e)
        raise HTTPException(status_code=400, detail=f"입력 값 오류: {e}")
    except RuntimeError as e:
        logger.error("RuntimeError: %s", e)
        raise HTTPException(status_code=500, detail=f"런타임 오류 발생: {e}")
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail=f"예상치 못한 오류 발생: {e}")
