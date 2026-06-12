"""
standalone_paddle_single_image.py

Minimal example: PaddleOCR on one image.
Shows all properties, outputs, and simple functions.
"""

from paddleocr import PaddleOCR
from PIL import Image
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. INITIALIZE
# ─────────────────────────────────────────────────────────────

# Create the OCR reader (downloads models on first run)
ocr = PaddleOCR(
    lang='en',           # language: 'en', 'ch', 'fr', 'en', etc.
    use_angle_cls=True,  # detect rotated text (90°, 180°, etc.)
    show_log=False,      # suppress verbose output       # CPU mode; set True if you have CUDA
)

print("PaddleOCR initialized.\n")


# ─────────────────────────────────────────────────────────────
# 2. LOAD A SINGLE IMAGE
# ─────────────────────────────────────────────────────────────

# Option A: From file path
image_path = "image_ocr.png"  # change this

# Option B: From PIL Image (useful if you already have one)
# pil_image = Image.open("your_image.png")

# Option C: From numpy array (useful for OpenCV or pdf2image)
# img_array = np.array(pil_image)


# ─────────────────────────────────────────────────────────────
# 3. RUN OCR — THE CORE FUNCTION
# ─────────────────────────────────────────────────────────────

print(f"Running OCR on: {image_path}\n")

# result is a list of lists: [[line1], [line2], ...]
# Each line is: [bbox, (text, confidence)]
result = ocr.ocr(image_path, cls=True)

# result structure:
# [
#   [
#     [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],   # bounding box (4 corners)
#     ('Detected text here', 0.9876)              # (text, confidence score)
#   ],
#   ...
# ]

print(f"Raw result type: {type(result)}")
print(f"Number of pages detected: {len(result)}")
print(f"Number of text lines on page 0: {len(result[0]) if result[0] else 0}\n")


# ─────────────────────────────────────────────────────────────
# 4. EXTRACT SIMPLE PROPERTIES
# ─────────────────────────────────────────────────────────────

def extract_text_simple(result) -> str:
    """Just the text, joined with newlines."""
    lines = []
    if result and result[0]:
        for line in result[0]:
            text = line[1][0]
            lines.append(text)
    return "\n".join(lines)

def extract_text_only(result) -> list[str]:
    """Just a list of text strings, no metadata."""
    if result and result[0]:
        return [line[1][0] for line in result[0]]
    return []



print("=" * 50)
print("SIMPLE TEXT OUTPUT")
print("=" * 50)
simple_text = extract_text_simple(result)
print(simple_text)
print()

print("=" * 50)
print("PLAIN LIST OF STRINGS")
print("=" * 50)
print(extract_text_only(result))
print

