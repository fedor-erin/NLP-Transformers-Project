import numpy as np
import streamlit as st
from transformers import PreTrainedTokenizerFast, DistilBertForSequenceClassification

@st.cache(allow_output_mutation=True)
def load_tokenizer(path="./model.pt"):
    return PreTrainedTokenizerFast.from_pretrained(path)

@st.cache(allow_output_mutation=True)
def load_model(path="./model.pt"):
    return DistilBertForSequenceClassification.from_pretrained(path)
    
def get_article_category(tokenizer, model, label2id, text, only_top95=False):
    encoding = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    encoding = {k: v for k, v in encoding.items() if k in ['input_ids', 'attention_mask']}
    preds = model(**encoding)
    probs = preds.logits.softmax(dim=-1)[0].tolist()
    predictions = {label: round(prob, 3) for (label, prob) in zip(label2id.keys(), probs)}
    predictions = sorted(predictions.items(), key=lambda x: -x[1])
    if only_top95:
        top95_perc_index = list(np.cumsum([x[1] for x in predictions]) <= 0.95).index(False)
        predictions = predictions[:top95_perc_index]
    return predictions

st.markdown("### Article's category classification model")
st.markdown("<center><img width=200px src='https://blog.arxiv.org/files/2021/02/arxiv-logo-1.png'></center>", unsafe_allow_html=True)

tokenizer = load_tokenizer()
model = load_model()

with open("./model.pt/label2id.txt", "r") as f:
    label2id = eval(f.read())

title = st.text_area("Enter the article's title:")
abstract = st.text_area("Enter the article's abstract:")
text = title + ' ' + abstract

if len(text.split()) > 2:
    predictions = get_article_category(tokenizer, model, label2id, text, only_top95=True)
    st.markdown("This article relates to the following categories with specified probabilities:")
    for pred in predictions:
        st.markdown(f'* {pred[0]} (p={pred[1]})')
