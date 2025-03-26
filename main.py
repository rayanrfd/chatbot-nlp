from langchain.chat_models import init_chat_model
from langchain_core.prompts import prompt, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import numpy as np
import streamlit as st

#model = init_chat_model(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", model_provider="together")
model = ChatOllama(model="llama2")

classifier = pipeline('zero-shot-classification')
labels = ["Python Programming", "Not Python Programming"]

@st.cache_data
def is_python(query):
    result = classifier(query, candidate_labels=labels)
    index = np.argmax(result['scores'])
    return result['labels'][index]

if "messages" not in st.session_state:
    st.session_state.messages = []

for messages in st.session_state.messages:
    if isinstance(messages, HumanMessage):
        with st.chat_message('user'):
            st.markdown(messages.content)
    else :
        with st.chat_message('ai'):
            st.markdown(messages.content)

st.title("Python Chatbot")

question = st.chat_input("...")
if question:
    with st.chat_message("user"):
        st.markdown(f"User : {question}")
        st.session_state.messages.append(HumanMessage(question))

    if is_python(question) == "Not Python Programming":
        with st.chat_message('ai'):
            st.markdown("Assistant : Please make sure the question is related to Python Programming")
            st.session_state.messages.append(AIMessage("Assistant : Please make sure the question is related to Python Programming"))
    else:
        prompt_template = ChatPromptTemplate(
                    [
                (
                    "system",
                    """
                    You are a helpful python programming assistant. Answer all questions to the best of your ability, 
                    but only python programming related questions. Assume that the user is asking about python programming
                    if the question can be interpreted in both python programming and not python programming,
                    example : The user asks about the difference between with and while, he is asking about pythons keywords, not
                    english grammar.
                    Do not forget the development good practices.
                    """
                ),
                ("user", "{question}")
            ]
                )
        prompt = prompt_template.invoke({"question": question})
        response = model.invoke(prompt)
        #print("Assistant : ", response.content)
        with st.chat_message('ai'):
            st.markdown(f"Assistant : {response.content}")
            st.session_state.messages.append(AIMessage(response.content))

        
