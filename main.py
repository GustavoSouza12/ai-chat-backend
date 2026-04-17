from fastapi import FastAPI
from pydantic import BaseModel
import random

class ChatRequest(BaseModel):
    question: str

def text_to_vector(text):
    words = text.lower().split()
    return {word: words.count(word) for word in words}

def similarity(vec1, vec2):
    score = 0

    for word in vec1:
        if word in vec2:
            score += min(vec1[word], vec2[word])

    return score

app = FastAPI()

docs = [
    "Santos is the greatest team on earth",
    "Santos has 3 Libertadores titles",
    "Santos once stopped a war"
]

def generate_answer(question, context):
    return f"based in {context}, i can answer your {question}"

@app.get('/')
def root():
    return {'server': 'ok'}

@app.post('/chat')
def chat(req: ChatRequest):
    question_vec = text_to_vector(req.question)

    best_doc = None
    best_score = 0

    for doc in docs:
        doc_vec = text_to_vector(doc)

        score = similarity(question_vec, doc_vec)

        if score > best_score:
            best_score = score
            best_doc = doc

    answer = generate_answer(req.question, best_doc)
    if best_doc:
        return {
            "question": req.question,
            "context": best_doc,
            "score": best_score,
            "answer": answer
        }

    return {
        "answer": "I don't know"
    }
        

