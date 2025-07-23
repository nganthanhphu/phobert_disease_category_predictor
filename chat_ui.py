import time

import streamlit as st
from phobert_disease_predictor import get_model_path, load_phobert_model, n_predict, AutoTokenizer, LabelEncoder, encoder, dataset_loader

st.set_page_config(page_title="Disease Predictor", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    body, .stApp {background: #23272f !important;}
    .user-msg {background: #3a4250; color: #fff; padding: 10px 16px; border-radius: 16px; margin-bottom: 8px; display: block; max-width: 80%; min-width: 60px; margin-left: auto; text-align: right; width: fit-content;}
    .bot-msg {background: #2d313a; color: #fff; padding: 10px 16px; border-radius: 16px; margin-bottom: 8px; display: block; max-width: 80%; min-width: 60px; margin-right: auto; text-align: left; width: fit-content;}
    .chat-container {display: flex; flex-direction: column; gap: 0.5rem;}
    .warning {position: fixed; left: 0; right: 0; bottom: 0; background: #3a4250; color: #ffe58f; padding: 12px 0; text-align: center; border-top: 1px solid #ffe58f; z-index: 100;}
    .block-container {padding-bottom: 180px !important;}
    input[type="text"] {background: #23272f; color: #fff; border: 1px solid #3a4250; border-radius: 8px; padding: 8px 12px;}
    </style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=True)
def initialize_model():
    df = dataset_loader("hf://datasets/PB3002/ViMedical_Disease/ViMedical_Disease.csv")
    label_encoder = LabelEncoder()
    df = encoder(df, 'Disease', 'Disease_encoded', label_encoder)
    num_labels = len(label_encoder.classes_)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    model = load_phobert_model(get_model_path("joon985/PhoBERTDiseasePredictor", "model.pth"), num_labels)
    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = initialize_model()

st.title("Disease Predictor")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Câu hỏi", "", key="input", placeholder="Nhập triệu chứng của bạn", label_visibility ="hidden")
    send_btn = st.form_submit_button("Gửi")
    if send_btn and user_input.strip():
        st.session_state["messages"].append({"role": "user", "content": user_input})
        diseases = n_predict(user_input, model, tokenizer, label_encoder, top_k=3)
        bot_reply = "<br>".join([f"Bệnh {i + 1}: {d}" for i, d in enumerate(diseases)])
        st.session_state["messages"].append({"role": "bot", "content": bot_reply})
        st.rerun()

st.markdown(
    '<div class="warning">Thông tin chỉ mang tính chất tham khảo, không thay thế cho tư vấn y tế chuyên nghiệp.</div>',
    unsafe_allow_html=True)
