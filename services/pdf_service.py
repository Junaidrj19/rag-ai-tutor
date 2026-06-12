from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from pathlib import Path
import os
from services.embedding_service import get_embedding_model
from services.ocr_service import ocr_scanned_pdf, clean_ocr_text
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

DOCS_DIR = "data"

def load_pdf(pdf_path):
    pdf_paths = list(Path(DOCS_DIR).glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDFs in '{DOCS_DIR}'! Add files and retry")

    print(f"Found {len(pdf_paths)} PDFs...")

    all_documents = []
    for pdf_path in pdf_paths:
        print(f"📖 Loading file: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        images = None 

        for idx, doc in enumerate(docs):
            page_num = idx + 1
            
            # .strip() removes trailing/leading spaces and newlines
            page_text = doc.page_content.strip() 
            text_length = len(page_text)

            # 🛑 CRITICAL DEBUG PRINT: Let's see exactly what the computer sees
            print(f"📄 Page {page_num}: Native Text Length detected = {text_length} characters.")

            # We change this to check for actual alphanumeric characters, or a slightly higher character limit
            # If a page has fewer than 40 real words/characters, it's highly likely a handwritten/scanned page
            if text_length < 150:  
                print(f"   ➡️ [OCR PATH] Routing Page {page_num} to Tesseract OCR...")
                
                try:
                    if images is None:
                        print(f"   📸 Converting {pdf_path.name} to images (this may take a moment)...")
                        images = convert_from_path(
                            str(pdf_path),
                            poppler_path="/opt/homebrew/bin"
                        )
                    
                    page_image = images[idx]
                    ocr_text = pytesseract.image_to_string(page_image)
                    cleaned_text = clean_ocr_text(ocr_text)
                    
                    # Debug print to ensure OCR actually found words
                    print(f"   ✨ OCR Success! Extracted {len(cleaned_text)} characters for Page {page_num}.")
                    
                    doc.page_content = cleaned_text
                    doc.metadata["source"] = pdf_path.name
                    doc.metadata["page"] = page_num
                    doc.metadata["extraction_method"] = "ocr"
                    all_documents.append(doc)

                except Exception as e:
                    print(f"   ❌ ERROR during OCR on Page {page_num}: {str(e)}")
                    # Fallback to whatever text it had so the pipeline doesn't completely crash
                    all_documents.append(doc)
            else:
                print(f"   ➡️ [NATIVE PATH] Keeping native text for Page {page_num}.")
                doc.metadata["source"] = pdf_path.name
                doc.metadata["page"] = page_num
                doc.metadata["extraction_method"] = "native_digital"
                all_documents.append(doc)
                
        print(f"✅ Successfully processed {pdf_path.name}. Total documents in memory: {len(all_documents)}")
        
    return all_documents
