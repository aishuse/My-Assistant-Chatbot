import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Load profile text
base_dir = os.path.dirname(os.path.abspath(__file__))
profile_path = os.path.join(base_dir, 'profile.txt')

if os.path.exists(profile_path):
    with open(profile_path, 'r', encoding='utf-8') as f:
        profile_text = f.read()
else:
    profile_text = "No profile found."

# Initialize model
model = ChatGroq(temperature=0.7, model_name="llama3-70b-8192")
parser = StrOutputParser()

# Prompt template
chat_template = ChatPromptTemplate.from_messages([
("system", "You are a helpful and conversational AI assistant. Answer user questions confidently and directly based on Aiswarya Baby’s profile below. If asked whether something was created by her, and the profile confirms it, clearly say 'yes' in your reply. Provide short, friendly, and informative answers.\n\n{profile}"),
    ("human", "{input}")
])

chain = chat_template | model | parser

# Streamlit app UI
st.set_page_config(page_title="Ask About Aiswarya", page_icon="🤖", layout="wide")

# Profile Header
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #FF6347;
            text-align: center;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #dddddd;
        }
        .chat-style {
            background-color: #1c1c1c;
            padding: 10px;
            border-radius: 10px;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# st.image("aiswarya.jpg", width=160)
st.markdown('<div class="title">Ask My AI Chatbot 🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything about Aiswarya Baby’s profile, education, skills, or experience</div>', unsafe_allow_html=True)
st.markdown("---")

# Chat Interface
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    try:
        with st.spinner("Thinking..."):
            response = chain.invoke({
                "profile": profile_text,
                "input": user_input
            })
        st.markdown(f"<div class='chat-style'><b>You:</b> {user_input}<br><b>AI:</b> {response}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")
