import os
import re 
from pdf2image import convert_from_path
import pytesseract
from langchain_core.documents import Document

def ocr_scanned_pdf(file_path):
    print(f"Converting PDF to images for: {os.path.basename(file_path)}")
    pages = convert_from_path(file_path, dpi=300)
    print(f"✅ Converted to {len(pages)} pages")  # ADD THIS

    extracted_documents = []

    for page_number, page_image in enumerate(pages, start=1):
        print(f"👁️ OCR-ing Page {page_number}/{len(pages)}...")
        
        raw_text = pytesseract.image_to_string(page_image)
        print(f"   Raw OCR length: {len(raw_text)} chars")  # ADD THIS
        print(f"   Raw preview: {repr(raw_text[:100])}")   # ADD THIS

        cleaned_txt = clean_ocr_text(raw_text)
        print(f"   Cleaned length: {len(cleaned_txt)} chars")  # ADD THIS
        
        # ADD VALIDATION CHECK
        if len(cleaned_txt) < 50:
            print(f"   ⚠️ WARNING: Page {page_number} has very short text!")
        
        doc = Document(
            page_content=cleaned_txt,
            metadata={
                "source": os.path.basename(file_path),
                "page_number": page_number
            }
        )
        extracted_documents.extend([doc])

    print(f"✅ Total documents: {len(extracted_documents)}")  # ADD THIS
    return extracted_documents

def clean_ocr_text(text:str) -> str:
    text = re.sub(r'[|~°`]', '', text)          # remove noise chars
    text = re.sub(r'[ \t]+', ' ', text)          # collapse horizontal whitespace only
    text = re.sub(r'\n{3,}', '\n\n', text)       # reduce excess blank lines
    return text.strip()