from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from services.pdf_service import load_pdf
from services.chunk_service import chunk_documents
from services.embedding_service import create_vector_store
from services.retrieval_service import create_retriever
from services.llm_service import load_llm
from services.prompt_service import create_prompt

from functools import lru_cache

@lru_cache(maxsize=1)
def get_rag_chain():
    raw_documents = load_pdf("data")
    chunks = chunk_documents(raw_documents)
    vector_store = create_vector_store(chunks)
    retriever = create_retriever(vector_store)
    llm = load_llm()
    prompt = create_prompt()
    
    def format_docs(docs):
        print("\n" + "="*80)
        print("RETRIEVED DOCUMENTS")
        print("="*80)
        
        for i, doc in enumerate(docs):
            print(f"\nChunk {i+1}")
            print(doc.metadata)
            print(doc.page_content[:1000])
        return "\n\n".join(
            f"[Source:{doc.metadata.get('source')}, Page:{doc.metadata.get('page')}]\n{doc.page_content}"
            for doc in docs
        )

    return (
        {
            "context": (lambda x: x["new_question"]) | retriever | format_docs,
            "new_question": lambda x: x["new_question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

#debug_docs = retriever.invoke(query)
#print("--- EXACT CHUNKS DISCOVERED BY RETRIEVER ---")
#for i, doc in enumerate(debug_docs):
    #print(f"\n[Chunk {i+1}]:\n{doc.page_content}")
