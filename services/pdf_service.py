"""
pdf_service.py
==============
Loads all PDFs from the DOCS_DIR folder and returns a flat list of
LangChain Document objects — one per page — ready for chunking.

Extraction strategy per page
─────────────────────────────
1.  PyMuPDF (fitz) tries to pull the native text layer.
2.  If that layer is too short (< 150 chars) → the page is handwritten
    or scanned, so we fall back to Mistral OCR.
3.  The result is wrapped in a LangChain Document with rich metadata.

Dependencies
────────────
    pip install pymupdf mistralai langchain-core python-dotenv

Mistral OCR setup
─────────────────
    1. Sign up at https://console.mistral.ai and get an API key.
    2. Add it to your .env file:
         MISTRAL_API_KEY="your-key-here"
    OR export it in your terminal:
         export MISTRAL_API_KEY="your-key-here"
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv

import fitz                          # PyMuPDF
from mistralai import Mistral        # Mistral OCR
from langchain_core.documents import Document

load_dotenv()   # reads MISTRAL_API_KEY from your .env file if present

# ─── Configuration ────────────────────────────────────────────────────────────

DOCS_DIR = "data"          # Folder that holds your PDF files
MIN_NATIVE_CHARS = 150     # Pages with fewer chars are treated as scanned/handwritten


# ─── OCR fallback: Mistral OCR ────────────────────────────────────────────────

def _ocr_page_with_mistral(page: fitz.Page) -> str:
    """
    Renders a single PyMuPDF page to a PNG in memory, base64-encodes it,
    and sends it to Mistral's OCR model (mistral-ocr-latest) as an
    image_url message — which is how Mistral accepts inline image data.

    Parameters
    ----------
    page : fitz.Page
        A single page object from an open PyMuPDF document.

    Returns
    -------
    str
        The full text extracted by Mistral OCR, or "" on failure.

    How it works step by step
    ─────────────────────────
    1.  page.get_pixmap(dpi=300)
            Renders the page as a raster image at 300 DPI.
            300 DPI gives enough detail for cursive handwriting without
            producing an image so large the API call becomes slow.

    2.  pixmap.tobytes("png")
            Converts the rendered pixels to raw PNG bytes, entirely in RAM.
            No temp file is written to disk.

    3.  base64.b64encode(png_bytes).decode("utf-8")
            Mistral's API does not accept raw bytes — it expects images
            embedded as a base64 data URI inside an "image_url" field.
            base64 encoding converts binary -> ASCII-safe text so it can
            travel inside a JSON request body.

    4.  Mistral(api_key=...).ocr.process(...)
            Calls the dedicated Mistral OCR endpoint with:
              model         = "mistral-ocr-latest"  (their best OCR model)
              document type = "image_url"
              image_url     = "data:image/png;base64,<encoded data>"

            The "data:image/png;base64,..." format is a standard
            browser/API convention for embedding an image inline in text.
            Mistral decodes it server-side before running OCR.

    5.  response.pages[0].markdown
            Mistral returns the extracted text in Markdown format —
            headings, bullet points, and tables are all preserved.
            We use .markdown instead of plain .text because it keeps
            structure that helps the chunker split at natural boundaries.
    """
    try:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found. Add it to your .env file.")

        # Step 1 & 2: render page -> PNG bytes (in memory only)
        pixmap = page.get_pixmap(dpi=300)
        png_bytes = pixmap.tobytes("png")

        # Step 3: encode to base64 so it can travel inside a JSON body
        b64_image = base64.b64encode(png_bytes).decode("utf-8")
        image_data_uri = f"data:image/png;base64,{b64_image}"

        # Step 4: call Mistral OCR
        client = Mistral(api_key=api_key)
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": image_data_uri,
            }
        )

        # Step 5: extract markdown text from the response
        if not response.pages:
            print("      Warning: Mistral OCR returned no pages.")
            return ""

        return response.pages[0].markdown

    except Exception as e:
        print(f"      Mistral OCR failed: {e}")
        return ""


# ─── Native text extraction: PyMuPDF ──────────────────────────────────────────

def _extract_native_text(page: fitz.Page) -> str:
    """
    Extracts the digital (native) text layer from a PDF page using PyMuPDF.

    Parameters
    ----------
    page : fitz.Page
        A single page object from an open PyMuPDF document.

    Returns
    -------
    str
        The native text, stripped of leading/trailing whitespace.

    Why PyMuPDF instead of PyPDFLoader?
    ────────────────────────────────────
    PyMuPDF (fitz) is faster, more accurate, and correctly handles:
    - Multi-column layouts
    - Text inside tables
    - Unicode characters (equations, accents, etc.)
    - PDFs with embedded fonts that PyPDF sometimes mangles

    page.get_text("text") returns the plain text of the page.
    Other modes: "html", "dict", "blocks" — we use "text" for simplicity.
    """
    return page.get_text("text").strip()


# ─── Text cleaning ────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Cleans extracted text while preserving paragraph structure.

    Why not collapse ALL whitespace?
    ─────────────────────────────────
    The chunker in chunk_service.py uses "\n\n" and "\n" as split points.
    If we replace every whitespace character with a single space, those
    separators disappear and the chunker produces one massive chunk per page,
    which is too large and poorly retrievable.

    What we do instead:
    - Remove common OCR noise characters (|, ~, degree, backtick)
    - Collapse runs of spaces/tabs on the SAME line (horizontal whitespace only)
    - Reduce 3+ consecutive blank lines to exactly 2 (keeps paragraphs intact)
    - Strip leading/trailing whitespace
    """
    import re
    text = re.sub(r'[|~°`]', '', text)        # remove noise glyphs
    text = re.sub(r'[ \t]+', ' ', text)        # collapse horizontal whitespace only
    text = re.sub(r'\n{3,}', '\n\n', text)     # max 2 consecutive blank lines
    return text.strip()


# ─── Main entry point ─────────────────────────────────────────────────────────

def load_pdfs() -> list[Document]:
    """
    Loads every PDF in DOCS_DIR and returns a list of LangChain Documents.

    Each Document represents ONE PAGE and carries this metadata:
        source              filename (e.g. "EDC notes Unit 1-5.pdf")
        page                1-based page number
        extraction_method   "native_digital" or "mistral_ocr"

    The list is passed directly to chunk_service.chunk_documents().

    Step-by-step walkthrough
    ─────────────────────────
    1.  Glob for *.pdf files in DOCS_DIR.
    2.  Open each PDF with fitz.open() — this is PyMuPDF's document handle.
    3.  Iterate over pages using fitz's built-in page iterator.
    4.  Try native text extraction first (fast, free, perfect for digital PDFs).
    5.  If the page looks blank/scanned (< MIN_NATIVE_CHARS), call Mistral OCR.
    6.  Clean the text, wrap in a Document, append to results.
    """

    pdf_paths = list(Path(DOCS_DIR).glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDFs found in '{DOCS_DIR}'. Add your files and retry.")

    print(f"Found {len(pdf_paths)} PDF(s) in '{DOCS_DIR}'")

    all_documents: list[Document] = []

    for pdf_path in pdf_paths:
        print(f"\nProcessing: {pdf_path.name}")

        # Step 2: Open the PDF with PyMuPDF.
        # fitz.open() returns a Document object that behaves like a list of pages.
        # It reads the file once into memory — no repeated disk I/O per page.
        fitz_doc = fitz.open(str(pdf_path))
        print(f"   Total pages: {len(fitz_doc)}")

        for page_index, page in enumerate(fitz_doc):   # fitz pages are 0-based internally
            page_num = page_index + 1                   # human-readable 1-based number

            # Step 4: Try native text extraction first.
            native_text = _extract_native_text(page)
            char_count = len(native_text)
            print(f"   Page {page_num}: native text length = {char_count} chars")

            if char_count >= MIN_NATIVE_CHARS:
                # Digital page: native text is good.
                # This path is fast (no API call) and perfectly accurate for
                # typed/digital PDFs. PyMuPDF returns the text exactly as
                # encoded in the PDF's content stream.
                print(f"      [NATIVE] Using PyMuPDF text")
                final_text = _clean_text(native_text)
                method = "native_digital"

            else:
                # Scanned / handwritten page: use Mistral OCR.
                # char_count < MIN_NATIVE_CHARS means the PDF has no useful text
                # layer on this page. Common causes:
                #   - The page was scanned from paper (image-only PDF)
                #   - The page contains handwritten notes (no machine text layer)
                #   - The page is mostly diagrams/figures with very little text
                print(f"      [OCR] Routing to Mistral OCR...")
                ocr_text = _ocr_page_with_mistral(page)
                final_text = _clean_text(ocr_text)
                char_count_after = len(final_text)
                print(f"      OCR extracted {char_count_after} characters")

                if char_count_after < 50:
                    print(f"      WARNING: Very little text found on page {page_num}. "
                          f"Check if the image quality is sufficient.")
                method = "mistral_ocr"

            # Step 6: Wrap in a LangChain Document.
            # Document is a simple container: { page_content: str, metadata: dict }
            # The metadata travels with the chunk all the way to the retriever,
            # so the LLM can cite "source: X, page: Y" in its answer.
            doc = Document(
                page_content=final_text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_num,
                    "extraction_method": method,
                }
            )
            all_documents.append(doc)

        fitz_doc.close()   # release the file handle — good practice in long pipelines
        print(f"   Done with {pdf_path.name}")

    print(f"\nTotal pages loaded: {len(all_documents)}")
    return all_documents