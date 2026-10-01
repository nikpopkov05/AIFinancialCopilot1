from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

print("--- ДОСТУПНЫЕ МОДЕЛИ ДЛЯ ТВОЕГО КЛЮЧА ---")
for model in client.models.list():
    # Проверяем, поддерживает ли модель генерацию текста
    if "generateContent" in getattr(model, "supported_generation_methods", []):
        print(model.name)