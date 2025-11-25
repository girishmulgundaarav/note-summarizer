# image_extractor.py
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import os
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"



def extract_images_and_text(pdf_file) -> list[str]:
    """
    Extract images from a PDF and convert them to text using OCR.
    Returns a list of strings (captions/recognized text).
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    image_texts = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:  # RGB or grayscale
                img_data = pix.tobytes("png")
                pil_img = Image.open(io.BytesIO(img_data))
                text = pytesseract.image_to_string(pil_img)
                if text.strip():
                    image_texts.append(f"[Image on page {page_num+1}] {text.strip()}")
            pix = None  # free memory

    return image_texts