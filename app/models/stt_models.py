from pydantic import BaseModel
from typing import List, Dict
from prompts.prompts import single_speaker_prompt, multi_speaker_prompt
from dotenv import load_dotenv
import os
from fastapi import HTTPException
# from db.db_util import get_model_name
import json
import re
import google.generativeai as genai

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

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
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    async def analyze_stt(self, response: STTResponse) -> AnalysisResponse:
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
            completion = self.model.generate_content(messages)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

        content = completion.text

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
