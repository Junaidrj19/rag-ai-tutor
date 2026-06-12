from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

def load_llm():

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    print("API KEY:", api_key)

    if not api_key:
        raise ValueError("Gemini API key not found")


    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.2,
        google_api_key=api_key
    )

    return llm