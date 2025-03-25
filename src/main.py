from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from transformers import pipeline
import numpy as np

model = init_chat_model(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", model_provider="together")

classifier = pipeline('zero-shot-classification', model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
labels = ["Python Programming", "Not Python Programming"]

def is_python(query):
    result = classifier(query, candidate_labels=labels)
    index = np.argmax(result['scores'])
    return result['labels'][index]

def main():
    while True:
        user_input = input("You : ")
        if user_input in ["bye"]:
            break
        if is_python(user_input) == 'Not Python Programming':
            print("You're question should be related to Python Programming !!!")
        else:    
            response = model.invoke([HumanMessage(content=user_input)])
            print('Assistant : ',response.content)


if __name__ == "__main__":
    main()
