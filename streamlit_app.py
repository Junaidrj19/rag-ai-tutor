import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
# Import the engine function from your backend file!
from services.chat_service import get_tutor_response
from backend import get_rag_chain

@st.cache_resource
def load_chain():
    return get_rag_chain()

chain = load_chain()

# 1. Page Configuration Setup
st.set_page_config(page_title="B.Tech AI Tutor", page_icon="🎓", layout="centered")
st.title("🎓 Your Personal B.Tech AI Tutor")
st.subheader("Ask questions from your any syllabus")

# Initialize chat history tracker in the browser session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 2. Render Existing Chat Bubble Components from Session State
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 3. Capture Live Interactions from the User
if user_input := st.chat_input("Ask me about stacks, queues, linked lists..."):
    
    # Render user's bubble immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Trigger backend pipeline with loading indicators
    with st.chat_message("assistant"):
        with st.spinner("Searching syllabus chunks..."):
            # Call your backend file's function cleanly here!
            response = get_tutor_response(
                user_input=user_input, 
                chat_history=st.session_state.chat_history
            )
            st.markdown(response)

    # Append the dialog turn back to states tracking array
    st.session_state.chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=response)
    ])