import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from models.models import get_ocr_model
from typing import List
from dotenv import load_dotenv

load_dotenv()
ocr_model = get_ocr_model()

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

router = APIRouter()

@router.post("/ocr")
async def process_images(files: List[UploadFile] = File(...), type: str = Form(...)):
    try:
        response = await ocr_model.process_uploaded_files(files, type)
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
