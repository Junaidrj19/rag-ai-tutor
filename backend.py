from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

loader = PyPDFLoader("data/Unit-I DS Data Structures copy.pdf")
raw_documents = loader.load()

print(f"Loaded {len(raw_documents)} pages from the pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 300,
    length_function = len
)

chunks = text_splitter.split_documents(raw_documents)

print(f"Split pages into {len(chunks)} smaller text chunks.")

embedding_model = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, embedding_model)

print("Vector store built successfully")

retriever = vector_store.as_retriever(search_kwargs={"k":5})

llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.2,
        google_api_key=api_key
    )

chat_history = []

system_prompt = ("""
"You are expert engineering professor assisting students with their coursework."
"Answer the user's question ONLY based on the context provided."
"If the context doesn't contain the answer , say honestly you cannot find it in the syllabus."
"Do not make things up.\n\n"
"Format your response clearly using markdown with bullet points, numbered steps, or code blocks where applicable.\n\n"
"Context :\n{context}"
""")

prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human","{new_question}"),
])

rag_chain_lcel = (
    {
        "context": (lambda x: x["new_question"]) | retriever, 
        "new_question": lambda x: x["new_question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

def get_tutor_response(user_input: str, chat_history: list) -> str:
    if chat_history:
        messages = ChatPromptTemplate.from_messages([
        ("system","Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{user_input}")
        ])

        condense_chain = messages | llm | StrOutputParser()
        new_question = condense_chain.invoke({
            "chat_history": chat_history,
            "user_input": user_input
        }).strip()
        print(f"Standalone query is:{new_question}")
    else:
        new_question = user_input

    # 3. Run the full chain to get Gemini's response
    response = rag_chain_lcel.invoke({
        "new_question": new_question,
        "chat_history":chat_history
        })

    return response

#debug_docs = retriever.invoke(query)
#print("--- EXACT CHUNKS DISCOVERED BY RETRIEVER ---")
#for i, doc in enumerate(debug_docs):
    #print(f"\n[Chunk {i+1}]:\n{doc.page_content}")
