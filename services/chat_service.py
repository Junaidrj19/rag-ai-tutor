from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from backend import get_rag_chain
from services.llm_service import load_llm

def get_tutor_response(user_input: str, chat_history: list) -> str:
    llm = load_llm()
    if chat_history:
        messages = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}")
        ])

        condense_chain = messages | llm | StrOutputParser()
        new_question = condense_chain.invoke({
            "chat_history": chat_history,
            "user_input": user_input
        }).strip()
        print(f"Standalone query is: {new_question}")
    else:
        new_question = user_input

    response = get_rag_chain().invoke({       # ✅ fixed here
        "new_question": new_question,
        "chat_history": chat_history
    })

    return response