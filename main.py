from fastapi import FastAPI
from pydantic import BaseModel
import random 
from sentence_transformers import SentenceTransformer
from services.llm import generate_answer
from services.retrieval import get_top_docs
from services.data_loader import load_data, dataframe_to_docs
import numpy as np

app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    session_id: str



model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

df = load_data("data/faturamento_historico.xlsx")
docs = dataframe_to_docs(df)
doc_embeddings = model.encode(docs)


sessions = {}
@app.get("/")
def root():
    return {
        "status": "ok"
    }

@app.post("/chat")
def response(req: ChatRequest):
    history = sessions.get(req.session_id,[])
    question_embedding = model.encode(req.question)

    top_docs, scores = get_top_docs(
        question_embedding,
        doc_embeddings,
        docs,
        k=2
    )

    context = "\n".join(top_docs)
    best_score = scores[0]

    if best_score < 0.2:
        return {
            "answer": "I don't know..."
        }
    answer = generate_answer(req.question, context, history)
    history.append(
        {
            "role": "user",
            "content": req.question
        }
    )

    history.append({
        "role":"assistant",
        "content": answer
    })

    sessions[req.session_id] = history

    sessions[req.session_id] = history[-6:]

    return {
        "response": req.question,
        "context": context,
        "score": float(best_score),
        "answer": answer

    }
    
