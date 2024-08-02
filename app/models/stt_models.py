from pydantic import BaseModel
from typing import List, Dict
from prompts.prompts import single_speaker_prompt, multi_speaker_prompt
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from fastapi import HTTPException
from db.db_util import get_model_name
import json
import re

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

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
    nickname_a : str
    nickname_b : str

class AnalysisResponse(BaseModel):
    situation_analysis: Analysis
    faults: Dict[str, Fault]
    conclusion: Dict[str, str]
    explanation: Explanation
    solutions: Solutions
    emotion_analysis: EmotionAnalysis
    Incident: Incident
    nicknames: Nickname

class STTModel:
    def __init__(self, api_key):
        self.client = AsyncOpenAI(api_key=api_key)
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

        messages = [{"role": "system", "content": prompt}]
        
        for utterance in response.results.utterances:
            if len(speakers) == 1:
                messages.append({"role": "user", "content": f"I: {utterance.msg}"})
            else:
                if utterance.spk == 0:
                    messages.append({"role": "user", "content": f"A: {utterance.msg}"})
                else:
                    messages.append({"role": "user", "content": f"B: {utterance.msg}"})
        
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

        content = completion.choices[0].message.content.strip()

        # Remove any leading/trailing markdown code fences
        content = re.sub(r'^```json\n|```$', '', content)

        try:
            response_json = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Error parsing JSON response: {str(e)}")
        
        return AnalysisResponse(**response_json)

stt_model = STTModel(api_key=api_key)

def get_stt_model():
    return stt_model
