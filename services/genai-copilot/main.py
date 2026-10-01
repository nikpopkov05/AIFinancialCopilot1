import os
from fastapi import FastAPI, HTTPException, Status
from genai_copilot_service import GenAIFinancialCopilot, UserFinancialProfile, FinancialAnalysisResponse

app = FastAPI(title="GenAI Copilot Internal Microservice")

copilot = GenAIFinancialCopilot()

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "genai-copilot"}

@app.post("/analyze", response_model=FinancialAnalysisResponse)
def analyze(profile: UserFinancialProfile):
    try:
        result = copilot.analyze_financials(profile)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=Status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )