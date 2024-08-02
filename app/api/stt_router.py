from pydantic import BaseModel
from typing import List, Dict
from fastapi import APIRouter, HTTPException, Depends
from models.stt_models import get_stt_model, STTModel
from fastapi.responses import JSONResponse

class Utterance(BaseModel):
    start_at: int
    duration: int
    spk: int
    spk_type: str
    msg: str

class STTResults(BaseModel):
    utterances: List[Utterance]
    verified: List[bool]

class STTResponse(BaseModel):
    id: str
    status: str
    results: STTResults

router = APIRouter()

@router.post("/stt")
async def process_stt(response: STTResponse, model: STTModel = Depends(get_stt_model)):
    try:
        result = await model.analyze_stt(response)
        return JSONResponse(content=result.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
