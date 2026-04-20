from openai import OpenAI
from core.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(question, context, history):
    messages = [
        {
            "role": "system",
            "content": "You are a strict assistant. Answer ONLY using the provided context. You can use context and conversation history. If the answer is not explicitly in the context, say you don't know."
        }
    ]

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{question}"
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content