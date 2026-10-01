import os
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from sympy.physics.units import temperature

from rag_service import FinancialRAGService
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenAICopilotService")


# ---------------------------------------------------------------------------
# 1. Схема входных данных (Pydantic Models)
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    transaction_id: str
    amount: float
    currency: str = "USD"
    category: str
    merchant_name: str
    timestamp: str
    is_anomaly: bool = False
    risk_score: float = 0.0


class UserFinancialProfile(BaseModel):
    user_id: str
    monthly_budget: float
    currency: str = "USD"
    transactions: List[Transaction]


# ---------------------------------------------------------------------------
# 2. Схема выходных данных (Structured Output Schema for Gemini)
# ---------------------------------------------------------------------------

class BudgetAdvice(BaseModel):
    category: str = Field(description="Категория расходов, к которой относится совет")
    status: str = Field(description="Статус расхода: Normal, Warning, Overbudget")
    recommendation: str = Field(description="Конкретная рекомендация по оптимизации трат")


class FinancialAnalysisResponse(BaseModel):
    overall_health_score: int = Field(
        description="Оценка финансового здоровья пользователя от 1 до 100"
    )
    summary: str = Field(
        description="Краткий аналитический итог по финансовой активности за период"
    )
    suspicious_transactions_comment: Optional[str] = Field(
        default=None,
        description="Комментарий и объяснение по аномальным или подозрительным транзакциям, если они обнаружены"
    )
    budget_advices: List[BudgetAdvice] = Field(
        description="Список рекомендаций по конкретным категориям бюджетирования"
    )
    actionable_tips: List[str] = Field(
        description="2-3 быстрых действия, которые пользователь может совершить прямо сейчас для экономии"
    )


# ---------------------------------------------------------------------------
# 3. Основной класс сервиса GenAI Copilot
# ---------------------------------------------------------------------------

class GenAIFinancialCopilot:
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация клиентом google-genai.
        Если api_key не передан, SDK автоматически возьмет значение из переменной GEMINI_API_KEY.
        """
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        # Используем лёгкую и быструю модель Gemini 1.5 Flash для низкого latency
        self.model_name = "models/gemini-1.5-pro"

    def analyze_financials(self, profile: UserFinancialProfile) -> FinancialAnalysisResponse:
        """
        Анализирует финансовый профиль пользователя и возвращает валидированный JSON.
        """
        system_instruction = (
            "Ты — высококвалифицированный персональный AI Financial Copilot в Финтех-приложении. "
            "Твоя задача — анализировать историю транзакций пользователя, оценивать финансовое здоровье, "
            "комментировать подозрительные транзакции (фрод/аномалии) и давать четкие, практические советы по экономии. "
            "Отвечай строго на русском языке, вежливо, лаконично и структурировано."
        )

        prompt = f"""
        Проанализируй финансовый профиль пользователя:
        - ID пользователя: {profile.user_id}
        - Месячный бюджет: {profile.monthly_budget} {profile.currency}
        - История транзакций: {profile.transactions}

        Сформируй полный финансовый отчет с оценкой здоровья, рекомендациями по категориям, 
        разбором аномальных транзакций (если есть flag is_anomaly=True) и быстрыми советами.
        """

        try:
            # Настройка конфигурации запроса с использованием Pydantic-схемы для Structured Output
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,  # Низкая температура для стабильного и детерминированного JSON
                response_mime_type="application/json",
                response_schema=FinancialAnalysisResponse,
            )

            logger.info(f"Отправка запроса в Gemini API ({self.model_name}) для user_id={profile.user_id}...")

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            # Благодаря response_schema, SDK автоматически гарантирует соответствие JSON структуре.
            # Для парсинга в объект Pydantic переводим response.text напрямую:
            parsed_result = FinancialAnalysisResponse.model_validate_json(response.text)
            logger.info("Анализ успешно завершен и провалидирован.")
            return parsed_result

        except Exception as e:
            logger.error(f"Ошибка при взаимодействии с Gemini API: {str(e)}")
            raise e

class RAGQueryRequest(BaseModel):
    user_id: str
    user_query: str

class RAGCopilotResponse(BaseModel):
    answer: str
    retrieved_content: List[dict]

class RAGFinancialCopilot:
    def __init__(self, rag_service: FinancialRAGService):
        self.rag = rag_service
        self.ai_client = genai.Client()
        self.model_name = "gemini-2.5-flash"

    def answer_with_context(self, request: RAGQueryRequest) -> RAGCopilotResponse:
        # 1. Извлекаем из Qdrant релевантные транзакции
        context_transactions = self.rag.search_similar_transactions(
            user_id=request.user_id,
            query=request.user_query,
            top_k=5
        )

        # 2. Формируем контекст для LLM
        formatted_context = "\n".join([
            f"-{t.get('date', 'Н/Д')}: {t.get('description')} ({t.get('amount')} руб.)"
            for t in context_transactions
        ])

        system_instruction = (
            "Ты - персональный AI Финансовый Ассистент."
            "Используй представленную историю транзакций пользователя, чтобы дать точный, "
            "аргументированный и структурированный ответ на его вопрос. "
            "Если данных недостаточно, честно скажи об этом."
        )

        prompt = f"""
        История релевантных транзаккий пользователя:
        {formatted_context if formatted_context else "Транзакций по данному запросу не найдено."}
        
        Вопрос пользователя : {request.user_query}
        """

        # 3. Запрос к Gemini
        response = self.ai_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )
        )

        return RAGCopilotResponse(
            answer=response.text,
            retrieved_content=context_transactions
        )

# ---------------------------------------------------------------------------
# 4. Пример запуска сервиса (Demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Создаем тестовые данные транзакций (включая одну аномальную от ML-сервиса)
    sample_profile = UserFinancialProfile(
        user_id="usr_882910",
        monthly_budget=150000.0,
        currency="RUB",
        transactions=[
            Transaction(
                transaction_id="tx_001",
                amount=3500.0,
                category="Groceries",
                merchant_name="Пятерочка",
                timestamp="2026-03-01T12:30:00"
            ),
            Transaction(
                transaction_id="tx_002",
                amount=12000.0,
                category="Restaurants",
                merchant_name="Ресторан Шеф",
                timestamp="2026-03-02T20:15:00"
            ),
            Transaction(
                transaction_id="tx_003",
                amount=85000.0,
                category="Electronics",
                merchant_name="Unknown Crypto Exchange",
                timestamp="2026-03-03T03:11:00",
                is_anomaly=True,
                risk_score=0.92
            ),
            Transaction(
                transaction_id="tx_004",
                amount=1500.0,
                category="Transport",
                merchant_name="Яндекс Гоу",
                timestamp="2026-03-04T09:00:00"
            )
        ]
    )

    # Инициализация и запуск сервиса
    copilot = GenAIFinancialCopilot()

    # Для теста уберем обработчик ошибок, если ключа нет
    if not os.getenv("GEMINI_API_KEY"):
        print("Ошибка: Переменная окружения GEMINI_API_KEY не установлена!")
    else:
        result: FinancialAnalysisResponse = copilot.analyze_financials(sample_profile)

        # Вывод полученного строго валидированного JSON
        print("\n=== РЕЗУЛЬТАТ АНАЛИЗА GENAI COPILOT ===")
        print(result.model_dump_json(indent=2, ensure_ascii=False))
