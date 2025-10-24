import os
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load API_KEY from local environment
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load API_KEY from streamlit secret
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if OPENAI_API_KEY:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key = OPENAI_API_KEY
        )
    
system_message = '''Kamu adalah AI Assistant cerdas yang ahli di bidang Data Science dan Artificial Intelligence.
Tugas utama kamu adalah untuk menjawab pertanyaan User terkait dengan Data Science, AI, Machine Learning, Deep Learning, dan bidang terkait lainnya.
Jangan jawab pertanyaan diluar bidang tersebut.
Selalu melayani User dengan ramah dan asik. Silahkan gunakan emoji apabila diperlukan.'''

def chat_ai(question, history):
    complete_prompt = f'''**System Message**: {system_message}

**History Chat:**
{history}

**Question** : {question}
'''
    response = llm.invoke(complete_prompt)
    return complete_prompt, response

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me recipes question"):
    messages_history = st.session_state.get("messages", [])[-20:]
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    # Display user message in chat message container
    with st.chat_message("Human"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("AI"):
        complete_prompt, response = chat_ai(prompt, history)
        answer = response.content
        st.markdown(answer)
        st.session_state.messages.append({"role": "AI", "content": answer})

    with st.expander("History Chat"):
        for chat in st.session_state.messages:
            st.markdown(f"**{chat['role']}** : {chat['content']}")