import os
import json
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agents.debater import Debater

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Climate Policy Debate Simulator")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3:8b")

# Models
class DebateRequest(BaseModel):
    topic: str
    rounds: int = Field(ge=1, le=5)

class DebateMessage(BaseModel):
    round: int
    agent: str
    message: str
    stance: str
    timestamp: str

class DebateResponse(BaseModel):
    messages: List[DebateMessage]

# Routes
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/policies/{country_code}")
def get_policy(country_code: str):
    country_code = country_code.lower()
    file_path = f"data/policies/{country_code}_policy.json"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Policy for {country_code} not found")
    
    with open(file_path, "r") as f:
        return json.load(f)

@app.post("/debate/start", response_model=DebateResponse)
async def start_debate(request: DebateRequest):
    agents = [
        Debater("USA", LLM_MODEL_NAME, OLLAMA_BASE_URL),
        Debater("EU", LLM_MODEL_NAME, OLLAMA_BASE_URL),
        Debater("China", LLM_MODEL_NAME, OLLAMA_BASE_URL)
    ]
    
    debate_history = []
    messages = []
    
    for r in range(1, request.rounds + 1):
        for agent in agents:
            response = await agent.generate_response(request.topic, debate_history)
            
            message = DebateMessage(
                round=r,
                agent=response["agent"],
                message=response["message"],
                stance=response["stance"],
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            messages.append(message)
            debate_history.append({
                "agent": message.agent,
                "message": message.message
            })
            
    return DebateResponse(messages=messages)

@app.get("/")
def read_root():
    with open("static/index.html", "r") as f:
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=f.read())
