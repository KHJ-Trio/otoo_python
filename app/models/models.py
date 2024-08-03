import google.generativeai as genai
from pydantic import BaseModel
from typing import List, Dict
from fastapi import HTTPException
import json
import re
from dotenv import load_dotenv
import os
from db.db_util import get_model_name
from prompts.prompts import conflict_prompt, love_prompt, friendship_prompt, emotion_report_prompt, chatbot_prompt_mode_2, chatbot_prompt_default, qna_prompt
from modules.module_pre import clean_chat
from models.image_processor import ImageProcessor
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from pymongo import MongoClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

class EmotionReportModel:
    def __init__(self, api_key):
        self.client = api_key
    async def initialize(self):
        self.model_name = await get_model_name()
    async def generate_messages_response(self, messages_request):
        await self.initialize()
        messages = messages_request.messages
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            loaded_vectorstore = FAISS.load_local("faiss/emotion_index", embeddings, allow_dangerous_deserialization=True)
            retriever = loaded_vectorstore.as_retriever()
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001",
                             convert_system_message_to_human=True)
            prompt = PromptTemplate.from_template(emotion_report_prompt)
            # 체인 설정
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            question = messages
            print('question',question)
            response = chain.invoke(question)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling API: {str(e)}")
        return response
class QnaModel:
    def __init__(self, api_key):
        self.client = api_key
    async def initialize(self):
        self.model_name = await get_model_name()
    async def generate_qna_response(self, messages_request):
        await self.initialize()
        messages = messages_request.messages
        try:
            # MongoDB 클라이언트 초기화 및 컬렉션 접근
            mongo = MongoClient(os.getenv('MONGO_CLIENT_URL'))
            db = mongo[os.getenv('MONGO_DB')]
            collection = db[os.getenv('MONGO_COLLENTION_QNA')]
            documents = list(collection.find({}, {'_id': 0, 'content': 1, 'embedding': 1}))
            documents = [Document(page_content=doc["content"], embedding=doc["embedding"]) for doc in documents]
            # FAISS에 임베딩 데이터 로드
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )
            # 검색기 생성
            retriever = vectorstore.as_retriever()
            # LLM 및 프롬프트 템플릿 초기화
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001",
                             convert_system_message_to_human=True)
            prompt = PromptTemplate.from_template(qna_prompt)
            # 체인 설정: 검색기, LLM, 출력 파서
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            # 체인에 질문 입력 및 응답 생성
            question = messages
            response = chain.invoke(question)
            print(f"response:{response}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
        return response
class QnAEditorModel:
    def __init__(self, api_key):
        self.client = api_key
    async def initialize(self):
        self.model_name = await get_model_name()
    async def generate_qna_edit_response(self, messages_request):
        await self.initialize()
        messages = messages_request.messages
        try:
            client = MongoClient(os.getenv('MONGO_CLIENT_URL'))
            db = client[os.getenv('MONGO_DB')]
            collection = db[os.getenv('MONGO_COLLENTION_QNA')]
            collection.delete_many({})
            text_stream = [Document(page_content= messages, metadata= {'source': 'QnA'})]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            split_documents = text_splitter.split_documents(text_stream)
            # 임베딩 생성
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            embedded_docs = [{"content": doc.page_content, "embedding": embeddings.embed_documents([doc.page_content])[0]} for doc in split_documents]
            # MongoDB에 저장
            collection.insert_many(embedded_docs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")
        return HTTPException(status_code=200, detail="QnA 데이터베이스 업데이트 완료")
class ChatbotModel:
    def __init__(self, api_key):
        self.api_key = api_key
    async def initialize(self):
        self.model_name = await get_model_name()
    async def generate_chat_response(self, mode_request, recent_messages_request):
        await self.initialize()
        recent_messages = recent_messages_request.RecentMessages
        mode = mode_request.mode
        if mode == '2':
            prompt = chatbot_prompt_mode_2
        else:
            prompt = chatbot_prompt_default
        messages = [
            {
                "role": "user",
                "parts": prompt
            }
        ]
        for message in recent_messages:
            role, content = message.split(" : ", 1)
            print(f"message :{message}, role :{role}, content :{content}")
            if role == 'user':
                messages.append({
                    "role": "user",
                    "parts": [content]
                })
            elif role == 'assistant':
                print('here')
                messages.append({
                    "role": "model",
                    "parts": [content]
                })
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(messages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling Google Gemini API: {str(e)}")
        content_value = response.text.strip()
        return content_value
class AnalysisModel:
    def __init__(self, api_key):
        self.api_key = api_key
    async def initialize(self):
        self.model_name = await get_model_name()
    async def analyze(self, text, analysis_type):
        await self.initialize()
        if analysis_type == 'conflict':
            prompt = conflict_prompt
        elif analysis_type == 'love':
            prompt = love_prompt
        elif analysis_type == 'friendship':
            prompt = friendship_prompt
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis type")
        try:
            text = clean_chat(text)
        except Exception as e:
            f"지원하지 않는 데이터 형식: {str(e)}"
        messages = [
            {"role": "user", "parts": [prompt]},
            {"role": "user", "parts": [text]}
        ]
        try:
            model = genai.GenerativeModel(self.model_name,
                                          generation_config = genai.GenerationConfig(
                                          temperature=0.1,
                                          ))
            response = model.generate_content(messages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling Google Gemini API: {str(e)}")
        return response.text.strip()
class OcrModel:
    def __init__(self, api_key):
        self.api_key = api_key
        self.image_processor = ImageProcessor()
    async def initialize(self):
        self.model_name = await get_model_name()
    async def analyze(self, text_data, analysis_type):
        await self.initialize()
        if analysis_type == 'conflict':
            prompt = conflict_prompt
        elif analysis_type == 'love':
            prompt = love_prompt
        elif analysis_type == 'friendship':
            prompt = friendship_prompt
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis type")
        try:
            text = f"상대방: {', '.join(text_data['상대방'])}\n나: {', '.join(text_data['나'])}"
        except:
            raise HTTPException(status_code=400, detail="Invalid analysis data")
        messages = [
            {"role": "user", "parts": [prompt]},
            {"role": "user", "parts": [text]}
        ]
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(messages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling Google Gemini API: {str(e)}")
        return response.text.strip()
    async def process_uploaded_files(self, files, type):
        results = await self.image_processor.process_images(files)
        all_left_text, all_right_text = results
        if not all_left_text and not all_right_text:
            raise ValueError("조건에 맞는 데이터를 넣어주세요")
        text_data = {
            "상대방": all_left_text,
            "나": all_right_text
        }
        analysis_result = await self.analyze(text_data, type)
        return analysis_result

emotion_report_model = EmotionReportModel(api_key=GOOGLE_API_KEY)
chatbot_model = ChatbotModel(api_key=GOOGLE_API_KEY)
analysis_model = AnalysisModel(api_key=GOOGLE_API_KEY)
ocr_model = OcrModel(api_key=GOOGLE_API_KEY)
qna_model = QnaModel(api_key=GOOGLE_API_KEY)
qna_editor_model = QnAEditorModel(api_key=GOOGLE_API_KEY)

def get_emotion_report_model():
    return emotion_report_model
def get_chatbot_model():
    return chatbot_model
def get_analysis_model():
    return analysis_model

ocr_model_instance = None

def get_ocr_model():
    global ocr_model_instance
    if ocr_model_instance is None:
        ocr_model_instance = ocr_model
    return ocr_model_instance
def get_qna_model():
    return qna_model
def get_qna_editor_model():
    return qna_editor_model
