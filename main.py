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
    best_doc = None
    best_score = 0
    question_words = req.question.lower().split()

    for doc in docs:
        doc_splited = doc.lower().split()
        score = 0
        for word in question_words:
        
            if word in doc_splited:
                score += 1

        if score > best_score:
            best_score = score
            best_doc = doc

    if best_doc:
        return {
            "question": req.question,
            "context": best_doc,
            "score": best_score,
            "answer": f"based on this: {best_doc}"
        }

    return {
        "answer": "I don't know"
    }

        

