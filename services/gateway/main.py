import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("APIGateway")

app = FastAPI(
    title="AI Financial Copilot API Gateway",
    version="1.0.0",
    description="Gateway API для маршрутизации запросов к микросервисам Fraud Detection и GenAI Copilot."
)

#CORS настройка для мобильных и веб-клиентов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Адреса микросервисов из переменных окружения (Docker network names)
GENAI_SERVICE_URL = os.getenv("GENAI_SERVICE_URL", "http://genai_copilot:8001")
FRAUD_SERVICE_URL = os.getenv("FRAUD_SERVICE_URL", "http://fraud_detector:8002")

#Pydantic модели Data Transfer Objects (DTO)

class TransactionDTO(BaseModel):
    transaction_id: str
    amount: float
    currency: str = "RUB"
    category: str
    merchant_name: str
    timestamp: str
    is_anomaly: bool = False
    risk_score: float = 0.0

class UserProfileDTO(BaseModel):
    user_id: str
    monthly_budget: float
    currency: str = "RUB"
    transactions: List[TransactionDTO]

# Эндпоинты API Gateway
@app.get("/health", status_code=Status.HTTP_200_OK)
async def health_check():
    """Проверка работоспособности API Gateway"""
    return {"status": "ok", "service": "api-gateway"}


@app.post("/api/v1/copilot/analyze", status_code=Status.HTTP_200_OK)
async def analyze_user_financials(profile: UserProfileDTO):
    """
    Эндпоинт для получения аналитики GenAI Copilot.
    Пробрасывает запрос в внутренний микросервис genai_copilot.
    """
    logger.info(f"Получен запрос на финансовый анализ для user_id={profile.user_id}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # 1. Запрос к GenAI Copilot сервису
            response = await client.post(
                f"{GENAI_SERVICE_URL}/analyze",
                json=profile.model_dump()
            )

            if response.status_code != 200:
                logger.error(f"Ошибка микросервиса GenAI Copilot: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=Status.HTTP_502_BAD_GATEWAY,
                    detail="Ошибка при обращении к микросервису GenAI Copilot"
                )

            return response.json()

        except httpx.RequestError as exc:
            logger.error(f"Ошибка сети при обращении к GenAI Copilot: {exc}")
            raise HTTPException(
                status_code=Status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Микросервис GenAI Copilot временно недоступен"
            )

@app.post("/api/v1/transactions/check-fraud", status_code=Status.HTTP_200_OK)
async def check_transaction_fraud(transaction: TransactionDTO):
    """
    Эндпоинт проверки транзакции на форд/аномалии.
    Пробрасывает запрос в микросервис fraud_detector.
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.post(
                f"{FRAUD_SERVICE_URL}/predict",
                json=transaction.model_dump()
            )
            return response.json()
        except httpx.RequestError:
            raise HTTPException(
                status_code=Status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Сервис Fraud Detection недоступен"
            )