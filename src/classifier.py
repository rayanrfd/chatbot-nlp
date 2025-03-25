from transformers import pipeline
import numpy as np

classifier = pipeline('zero-shot-classification', model="huggingface/CodeBERTa-language-id")#
labels = ["Python Programming", "Not Python Programming"]

def is_python(query):
    result = classifier(query, candidate_labels=labels)
    index = np.argmax(result['scores'])
    return result['labels'][index]

print(is_python("What's the difference between arrow function and normal functions"))
