from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import FAISS_INDEX_PATH
import os 

def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )

def create_vector_store(chunks):

    embedding_model = get_embedding_model()

    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing vector store...")

        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

    print("Building vector store for the first time...")

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    vector_store.save_local(FAISS_INDEX_PATH)

    print("Vector store saved successfully.")

    return vector_store