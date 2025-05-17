from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Load model
model = ChatGroq(temperature=0.7, model_name="llama3-70b-8192")

# Load profile.txt content
base_dir = os.path.dirname(os.path.abspath(__file__))
profile_path = os.path.join(base_dir, 'profile.txt')

if os.path.exists(profile_path):
    with open(profile_path, 'r', encoding='utf-8') as f:
        profile_text = f.read()
else:
    profile_text = "No profile found."

# Prompt template that embeds your profile into context
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant answering questions about Aiswarya Baby. Use the profile information below to answer questions accurately and professionally:\n\n{profile}"),
    ("human", "{input}")
])

# Parser and chain
parser = StrOutputParser()
chain = chat_template | model | parser

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    try:
        result = chain.invoke({
            "profile": profile_text,
            "input": user_input
        })
        print("AI:", result)
    except Exception as e:
        print("Error:", e)
