import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
from utils.utils import load_config

# Load configuration
config_path = "config/cfg.yaml"
config = load_config(config_path)
question_answerer = pipeline("question-answering", model=config["output_dir"], tokenizer=config["output_dir"])

def answer_question(question, context):
    if isinstance(context, list):
        context = " ".join(context)
    answer = question_answerer(question=question, context=context)
    print(answer['answer'])
    return answer['answer']

st.set_page_config(layout="wide")
st.title("COVID-QA Question Answering System")
st.write("Enter a context and a question to get an answer from the fine-tuned model.")

context = st.text_area("Context", height=200, placeholder= "COVID-19 is an infectious disease caused by the most recently discovered coronavirus.\n"
                                                         '["Covid-19", "is", "an", "infectious", "disease", "caused", "by", "the", "most", "recently", "discovered", "coronavirus"]')
question = st.text_input("Question", placeholder="What is COVID-19?")

if st.button("Get Answer"):
    if context and question:
        answer = answer_question(question, context)
        st.write("Answer:", answer)
    else:
        st.write("Please provide both context and question.")