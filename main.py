from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# -----------------------------
# APP
# -----------------------------
app = FastAPI()

# -----------------------------
# MODEL (EMBEDDINGS)
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# DATASET (BASE KNOWLEDGE)
# -----------------------------
docs = [
    "Santos has 3 Libertadores titles",
    "Santos is one of the biggest football clubs in Brazil",
    "Santos once stopped a war",
    "Pelé played for Santos"
]

# Pré-computa embeddings dos docs (importante pra performance)
doc_embeddings = model.encode(docs)

# -----------------------------
# INPUT MODEL
# -----------------------------
class ChatRequest(BaseModel):
    question: str

# -----------------------------
# RETRIEVAL FUNCTION
# -----------------------------
import numpy as np

def get_top_docs(question_embedding, doc_embeddings, docs, k=2):
    
    # 1. similarity
    scores = cosine_similarity([question_embedding], doc_embeddings)[0]

    # 2. index top k
    top_indices = np.argsort(scores)[-k:][::-1]

    # 3. get docs
    top_docs = [docs[i] for i in top_indices]

    # 4. get scores
    top_scores = [scores[i] for i in top_indices]

    return top_docs, top_scores

def generate_answer(question, context):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a strict assistant. ONLY answer using the provided context. If the answer is not in the context, say: I don't know based on the provided context."
            },
            {
                "role": "user",
                "content": f"Context: {context}\nQuestion: {question}"
            }
        ]
    )

    return response.choices[0].message.content

# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok"}

# -----------------------------
# CHAT ENDPOINT (RAG CORE)
# -----------------------------
@app.post("/chat")
def chat(req: ChatRequest):

    # 1. embedding da pergunta
    question_embedding = model.encode(req.question)

    # 2. retrieval (busca do melhor contexto)
    top_docs, scores = get_top_docs(
        question_embedding,
        doc_embeddings,
        docs,
        k=2
    )

    context = "\n".join(top_docs)

    # 3. geração simulada (LLM fake por enquanto)
    best_score = scores[0]
    if best_score < 0.5:
        return {
            "question": req.question,
            "answer": "I don't know based on the available data."
        }
    
    answer = generate_answer(req.question, context)

    # 4. resposta final
    return {
        "question": req.question,
        "context": context,
        "score": float(best_score),
        "answer": answer
    }