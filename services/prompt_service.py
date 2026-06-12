from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_prompt():

    system_prompt = ("""
    You are an expert engineering professor assisting students with their coursework.
    Answer the user's question ONLY based on the context provided.
    If the context doesn't contain the answer, say honestly you cannot find it in the syllabus.
    Do not make things up.
    
    Format your response clearly using markdown with bullet points, numbered steps, or code blocks where applicable.
    Always cite the source PDF and page number in your answer.
    Context:\n{context}
    """)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human","{new_question}"),
    ])

    return prompt
