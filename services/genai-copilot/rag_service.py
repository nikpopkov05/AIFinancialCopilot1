import os
from typing import List, Dict, Any
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class FinancialRAGService:
    def __init__(
            self,
            qdrant_host: str = "localhost",
            qdrant_port: int = 6333,
            collection_name: str = "user_transactions"
    ):
        self.collection_name = collection_name

        # 1. Инициализация клиентов Google GenAI и Qdrant
        self.ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

        # 2. Модель для вектора (text-embedding-004 выдает размерность 768)
        self.embedding_model = "text-embedding-004"
        self.vector_size = 768

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Создает коллекцию в Qdrant, если ее еще нет."""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                ),
            )

    def _get_embedding(self, text: str) -> List[float]:
        """Генерация эмбеддинга через Google GenAI SDK."""
        response = self.ai_client.models.embed_content(
            model=self.embedding_model,
            contents=text
        )
        return response.embedding.values

    def index_transaction(self, transaction_id: int, user_id: str, text_description: str, metadata: Dict[str, Any]):
        """
        Индексация транзакции в Qdrant
        text_description: "Покупка в ресторане Вкусно и точка на сумму 850 рублей, категория фастфуд"
        """
        vector = self._get_embedding(text_description)

        payload = {
            "user_id": user_id,
            "description": text_description,
            **metadata
        }

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=transaction_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def search_similar_transactions(self, user_id: str, query: str, top_k: int = 5):
        """Векторный поиск релевантных транзакций по смысловому запросу."""
        query_vector = self._get_embedding(query)

        #Поиск по векторному сходству с фильтрацией по user_id
        search_result = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter={
                "must": [
                    {"key": "user_id", "match": {"value": user_id}}
                ]
            }
        )

        return [hit.payload for hit in search_result]