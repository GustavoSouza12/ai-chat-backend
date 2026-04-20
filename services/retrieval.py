import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_docs(question_embedding, doc_embeddings, docs, k=2):
    scores = cosine_similarity([question_embedding], doc_embeddings)[0]

    top_indices = np.argsort(scores)[-k:][::-1]
    top_docs = [docs[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    return top_docs, top_scores