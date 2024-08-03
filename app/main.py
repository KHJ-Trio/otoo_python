from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
from api.emotion_report_router import router as emotion_report_router
from api.chatbot_router import router as chatbot_router
from api.analysis_router import router as analysis_router
from api.ocr_router import router as ocr_router
from api.qna_router import router as qna_router
from api.stt_router import router as stt_router
from dotenv import load_dotenv
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()
# 로그 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()


origins = [
    "https://fastapi.otoo.kr",
    "https://react.otoo.kr",
    "https://restapi.otoo.kr",
    "http://localhost:8080",
    "http://localhost:8001",
    "http://localhost:3000"

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#
@app.get("/health")
async def health_check():
    return {"message": "Welcome to FastAPI"}

# 라우터 등록
app.include_router(emotion_report_router)
app.include_router(chatbot_router)
app.include_router(analysis_router)
app.include_router(ocr_router)
app.include_router(qna_router)
app.include_router(stt_router)

# 모든 예외 로깅
@app.middleware("https")
async def log_exceptions(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}", exc_info=True)
        raise

# uvicorn main:app --reload --port 8001
