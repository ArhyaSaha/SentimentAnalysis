import os
import json
import asyncio
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import strawberry
from strawberry.fastapi import GraphQLRouter
from model_handler import ModelHandler

model_handler = ModelHandler()

@strawberry.type
class PredictionResult:
    label: str
    score: float

@strawberry.type
class Query:
    @strawberry.field
    def health(self) -> str:
        return "Sentiment Analysis API is running!"

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def predict_sentiment(self, text: str) -> PredictionResult:
        """Predict sentiment for given text"""
        result = await model_handler.predict(text)
        return PredictionResult(
            label=result["label"],
            score=result["score"]
        )

schema = strawberry.Schema(query=Query, mutation=Mutation)

app = FastAPI(
    title="Sentiment Analysis API",
    description="GraphQL API for binary sentiment analysis using Hugging Face Transformers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API", "graphql_endpoint": "/graphql"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_handler.is_loaded()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)