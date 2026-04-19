from fastapi import FastAPI
from pydantic import BaseModel
import random 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

docs = [
    "Pelé é um jogador de futebol que jogou no Santos",
    "Pelé tem 3 copas do mundo",
    "ele é muito bom e jogou no Santos"
]

model = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings = model.encode(docs)

def generate_top_docs(question_embedding, doc_embeddings, docs, k=2):
    scores = cosine_similarity([question_embedding], doc_embeddings)[0]

    top_indices = np.argsort(scores)[-k:][::-1]
    top_docs = [docs[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    return top_docs, top_scores

chat_history = []
def generate_answer(question, context):

        model="gpt-4o-mini",
        messages=[
            {
                "role":"system",
                "content":"You are a strict assistant. Answer ONLY using the provided context. You can use context and conversation history. If the answer is not explicitly in the context, say you don't know."
            }
        ]

        messages.extend(chat_history)

        messages.append({
              "role": "user",
              "content": f"Use the context below to answer. Context: {context}\nQuestion: {question}. "
        })

        response = client.chat.completions.create(
             model="gpt-4o-mini",
             messages=messages
        )

        return response.choices[0].message.content

@app.get("/")
def root():
    return {
        "status": "ok"
    }

@app.post("/chat")
def response(req: ChatRequest):

    question_embedding = model.encode(req.question)

    top_docs, scores = generate_top_docs(
        question_embedding,
        doc_embeddings,
        docs,
        k=2
    )

    context = "\n".join(top_docs)
    best_score = scores[0]

    if best_score < 0.5:
        return {
            "answer": "I don't know..."
        }
    answer = generate_answer(req.question, context)

    chat_history.append(
        {
            "role": "user",
            "content": req.question
        }
    )

    chat_history.append({
        "role":"assistant",
        "content": answer
    })


    return {
        "response": req.question,
        "context": context,
        "score": float(best_score),
        "answer": answer

    }
    
