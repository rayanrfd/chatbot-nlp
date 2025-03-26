import faiss
import json
import torch
import ollama
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    pipeline
)

FAISS_INDEX_PATH = "db/faiss_index.bin"
META_PATH = "db/faiss_metadata.txt"
CHUNKS_PATH = "db/faiss_chunks.json"

index = faiss.read_index(FAISS_INDEX_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = [line.strip().split("\t") for line in f.readlines()] 

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks_data = json.load(f) 

print("FAISS index, metadata, and chunks restored successfully!")

checkpoint = 'sentence-transformers/all-mpnet-base-v2'
model = AutoModel.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=384)  
    outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, axis=1).detach().numpy().squeeze(0).astype(np.float32)


def search(query, top_k=5):
    query_embedding = get_embedding(query).reshape(1, -1)
    faiss.normalize_L2(query_embedding) 
    
    distances, indices = index.search(query_embedding, top_k)  
    
    results = []
    for idx, score in zip(indices[0], distances[0]):  
        if str(idx) in chunks_data: 
            chunk_text = chunks_data[str(idx)]
            
            for meta_id, filename in metadata:
                if int(meta_id) == idx:
                    results.append({
                        "chunk": chunk_text,
                        "score": score,
                        "source": filename
                    })
                    break
        else:
            print(f"⚠️ Warning: FAISS returned invalid index {idx}, skipping.")

    return results

classifier = pipeline('zero-shot-classification')
labels = ["Python Programming", "Not Python Programming"]

def filter_input(input):
    """
    Uses zero-shot classification to ensure the query is Python-related.
    """
    results = classifier(input, candidate_labels=labels)
    if results['labels'][0] == 'Not Python programming':
        return 'Please make sure the question is related to Python Programming'
    return input

def query_llama(context, question):
    if not context.strip():  
        return "Sorry, I couldn't find any relevant information in the database."

    prompt = f"""You are an AI assistant specialized in Python programming.
Answer the following question **strictly** using the provided context. 
Use phrases like "As stated in the document..." or include direct quotes in your response.

If the context does not contain an answer, say: "I don't have enough information to answer this question."

**Context:**
{context}

**Question:**
{question}
"""

    response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]



def answer_query(user_query, top_k=5, threshold=0.6):
    filtered_query = filter_input(user_query)

    if "Please make sure" in filtered_query:
        return filtered_query  

    results = search(filtered_query, top_k=top_k)

    relevant_chunks = [r for r in results if r["score"] >= threshold]

    if not relevant_chunks: 
        return "Sorry, I couldn't find any relevant information in the database."

    context = "\n\n".join([r["chunk"] for r in relevant_chunks])
    sources = set(r["source"] for r in relevant_chunks)

    response = query_llama(context, user_query)

    final_output = f"{response}\n\nSources used:\n" + "\n".join(sources)
    return final_output


if __name__=="__main__":
        while True:
            query = input("You : ")
            if query == "bye":
                    break
            response = answer_query(query)
            print("\nResponse:\n")
            print(response)
