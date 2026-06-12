from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(all_documents):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 150,
    length_function = len,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_documents)
    
    print(f"Split pages into {len(chunks)} smaller text chunks.")

    return chunks