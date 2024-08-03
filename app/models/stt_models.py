from prompts.prompts import single_speaker_prompt, multi_speaker_prompt

import google.generativeai as genai
from pydantic import BaseModel
from typing import List, Dict
from fastapi import HTTPException
import json
import re
from dotenv import load_dotenv
import os
from db.db_util import get_model_name

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Google Gemini API 설정
genai.configure(api_key=GOOGLE_API_KEY)

# 데이터 모델 정의
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

class Analysis(BaseModel):
    speaker_a: str
    speaker_b: str

class Fault(BaseModel):
    fault: str
    percentage: int

class Explanation(BaseModel):
    speaker_a: str
    speaker_b: str

class Solutions(BaseModel):
    solutionsA: str
    solutionsB: str

class EmotionAnalysis(BaseModel):
    speaker_a: str
    speaker_b: str

class IncidentStage(BaseModel):
    a_behavior: str
    a_emotion: str
    b_behavior: str
    b_emotion: str

class Incident(BaseModel):
    development: IncidentStage
    deployment: IncidentStage
    crisis: IncidentStage
    climax: IncidentStage
    ending: IncidentStage

class Nickname(BaseModel):
    nickname_a: str
    nickname_b: str

class AnalysisResponse(BaseModel):
    situation_analysis: Analysis
    faults: Dict[str, Fault]
    conclusion: Dict[str, str]
    explanation: Explanation
    solutions: Solutions
    emotion_analysis: EmotionAnalysis
    Incident: Incident
    nicknames: Nickname

# STTModel 클래스 정의
class STTModel:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model_name = None

    async def initialize(self):
        if self.model_name is None:
            self.model_name = await get_model_name()

    async def analyze_stt(self, response: STTResponse) -> AnalysisResponse:
        await self.initialize()
        speakers = {utterance.spk for utterance in response.results.utterances}
        
        if len(speakers) == 1:
            prompt = single_speaker_prompt
        else:
            prompt = multi_speaker_prompt

        messages = [{"role": "user", "parts": prompt}]
        
        for utterance in response.results.utterances:
            if len(speakers) == 1:
                messages.append({"role": "user", "parts": f"I: {utterance.msg}"})
            else:
                if utterance.spk == 0:
                    messages.append({"role": "user", "parts": f"A: {utterance.msg}"})
                else:
                    messages.append({"role": "user", "parts": f"B: {utterance.msg}"})
        
        try:
            model = genai.GenerativeModel(self.model_name,
                                          generation_config = genai.GenerationConfig(
                                          temperature=0.1,
                                          ))
            response = model.generate_content(messages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling Google Gemini API: {str(e)}")

        content = response.text.strip()

        # JSON 코드 블록 제거
        content = re.sub(r'^```json\n|```$', '', content)

        try:
            response_json = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Error parsing JSON response: {str(e)}")
        
        return AnalysisResponse(**response_json)

# STT 모델 인스턴스 생성
stt_model = STTModel(api_key=GOOGLE_API_KEY)

# STT 모델 인스턴스 반환 함수
def get_stt_model():
    return stt_model
