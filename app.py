import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Load profile text
base_dir = os.path.dirname(os.path.abspath(__file__))
profile_path = os.path.join(base_dir, 'profile.txt')

if os.path.exists(profile_path):
    with open(profile_path, 'r', encoding='utf-8') as f:
        profile_text = f.read()
else:
    profile_text = "No profile found."

# Initialize LangChain model
model = ChatGroq(temperature=0.7, model_name="llama3-70b-8192")
parser = StrOutputParser()

# Prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and conversational AI assistant. Answer user questions confidently and directly based on Aiswarya Baby’s profile below. If asked whether something was created by her, and the profile confirms it, clearly say 'yes' in your reply. Provide short, friendly, and informative answers.\n\n{profile}"),
    ("human", "{input}")
])
chain = chat_template | model | parser

# Set Streamlit page config
st.set_page_config(page_title="Aiswarya’s AI Assistant", page_icon="🤖", layout="wide")

# Custom CSS for wider layout and better styling
st.markdown("""
    <style>
        .main {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .title {
            font-size: 44px;
            font-weight: bold;
            text-align: center;
            color: #FF4B4B;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #bbbbbb;
            margin-bottom: 30px;
        }
        .chat-bubble {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 16px;
            color: #f0f0f0;
            margin: 10px 0;
            font-size: 16px;
        }
        .user {
            border-left: 5px solid #FF4B4B;
        }
        .ai {
            border-left: 5px solid #1E90FF;
        }
        .stTextInput > div > input {
            color: white;
            background-color: #2a2a2a;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# App content wrapper
with st.container():
    st.markdown('<div class="title">Aiswarya’s AI Assistant 🤖</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ask anything about Aiswarya Baby’s profile, skills, projects, or experience</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Chat form
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

            st.markdown(f"<div class='chat-bubble user'><strong>You:</strong> {user_input}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble ai'><strong>AI:</strong> {response}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
