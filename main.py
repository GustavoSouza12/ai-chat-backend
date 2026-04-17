from fastapi import FastAPI
from pydantic import BaseModel
import random

class ChatRequest(BaseModel):
    question: str

app = FastAPI()

docs = [
    "Santos is the greatest team on earth",
    "Santos has 3 Libertadores titles",
    "Santos once stopped a war"
]

@app.get('/')
def root():
    return {'server': 'ok'}

@app.post('/chat')
def chat(req: ChatRequest):
    context = random.choice(docs)
    return {
        "question": req.question,
        "context": context,
        "answer": f"Based on this: {context}"
    }