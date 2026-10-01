from rag_service import FinancialRAGService
from genai_copilot_service import RAGFinancialCopilot, RAGQueryRequest

# Инициализация
rag = FinancialRAGService(qdrant_host="localhost")
copilot = RAGFinancialCopilot(rag_service=rag)

USER_ID = "usr_882910"

# 1. Заполняем Qdrant тестовыми транзакциями
sample_transactions = [
    (101, "Покупка продуктов в Супермаркете Перекресток", {"amount": 3400, "category": "Food", "date": "2026-08-01"}),
    (102, "Оплата подписки Spotify Premium", {"amount": 299, "category": "Subscriptions", "date": "2026-08-02"}),
    (103, "Ужин с друзьями в ресторане Вкусно и точка",
     {"amount": 1250, "category": "Restaurants", "date": "2026-08-03"}),
    (104, "Оплата спортивного зала Fitness Club", {"amount": 4500, "category": "Sports", "date": "2026-08-04"}),
]

print("Индексация транзакций в Qdrant...")
for tx_id, desc, meta in sample_transactions:
    rag.index_transaction(transaction_id=tx_id, user_id=USER_ID, text_description=desc, metadata=meta)

# 2. Задаем умный вопрос Copilot'у
request = RAGQueryRequest(
    user_id=USER_ID,
    user_query="Сколько я трачу на развлечения, еду вне дома и подписки?"
)

print("\n Запрос к RAG Copilot...")
result = copilot.answer_with_context(request)

print("\n--- ОТВЕТ GEMINI ---")
print(result.answer)

print("\n --- ИЗВЛЕЧЕННЫЙ ИЗ QDRANT КОНТЕКСТ ---")
for ctx in result.retrieved_content:
    print(ctx)