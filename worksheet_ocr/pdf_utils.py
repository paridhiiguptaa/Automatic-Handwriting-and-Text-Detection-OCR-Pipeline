import os
import io
import logging
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Optional

logger = logging.getLogger(__name__)

def is_pdf_file(input_source: Union[str, bytes]) -> bool:
    """Check if the input path or bytes represents a PDF document."""
    if isinstance(input_source, str):
        if os.path.isfile(input_source) and input_source.lower().endswith('.pdf'):
            return True
        # Check header bytes if file exists
        if os.path.isfile(input_source):
            try:
                with open(input_source, 'rb') as f:
                    header = f.read(4)
                    return header.startswith(b'%PDF')
            except Exception:
                return False
    elif isinstance(input_source, bytes):
        return input_source.startswith(b'%PDF')
    return False

def get_pdf_page_count(pdf_input: Union[str, bytes]) -> int:
    """Return the total number of pages in a PDF document."""
    # 1. Try pypdfium2
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_input)
        count = len(pdf)
        pdf.close()
        return count
    except Exception as e:
        logger.debug(f"pypdfium2 get_pdf_page_count failed: {e}")

    # 2. Try fitz (PyMuPDF)
    try:
        import fitz
        doc = fitz.open(pdf_input) if isinstance(pdf_input, str) else fitz.open(stream=pdf_input, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except Exception as e:
        logger.debug(f"fitz get_pdf_page_count failed: {e}")

    # 3. Try pdf2image
    try:
        from pdf2image import pdf2image
        info = pdf2image.pdfinfo_from_path(pdf_input) if isinstance(pdf_input, str) else pdf2image.pdfinfo_from_bytes(pdf_input)
        return info.get("Pages", 0)
    except Exception as e:
        logger.debug(f"pdf2image info failed: {e}")

    raise RuntimeError("Failed to read PDF page count. Ensure a compatible PDF library (pypdfium2, PyMuPDF, or pdf2image) is installed.")

def render_pdf_page(pdf_input: Union[str, bytes], page_index: int = 0, scale: float = 2.5) -> Image.Image:
    """
    Render a specific 0-indexed page of a PDF document as a PIL Image.
    scale=2.5 provides ~180-200 DPI for high quality OCR text detection.
    """
    # 1. Try pypdfium2
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_input)
        if page_index < 0 or page_index >= len(pdf):
            pdf.close()
            raise IndexError(f"Page index {page_index} out of range for PDF with {len(pdf)} pages.")
        page = pdf[page_index]
        pil_image = page.render(scale=scale).to_pil()
        pdf.close()
        return pil_image.convert("RGB")
    except Exception as e:
        logger.debug(f"pypdfium2 render_pdf_page failed: {e}")

    # 2. Try fitz (PyMuPDF)
    try:
        import fitz
        doc = fitz.open(pdf_input) if isinstance(pdf_input, str) else fitz.open(stream=pdf_input, filetype="pdf")
        if page_index < 0 or page_index >= len(doc):
            doc.close()
            raise IndexError(f"Page index {page_index} out of range for PDF with {len(doc)} pages.")
        page = doc[page_index]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        doc.close()
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        logger.debug(f"fitz render_pdf_page failed: {e}")

    # 3. Try pdf2image
    try:
        from pdf2image import convert_from_path, convert_from_bytes
        dpi = int(72 * scale)
        if isinstance(pdf_input, str):
            images = convert_from_path(pdf_input, first_page=page_index + 1, last_page=page_index + 1, dpi=dpi)
        else:
            images = convert_from_bytes(pdf_input, first_page=page_index + 1, last_page=page_index + 1, dpi=dpi)
        if images:
            return images[0].convert("RGB")
    except Exception as e:
        logger.debug(f"pdf2image render_pdf_page failed: {e}")

    raise RuntimeError("Failed to render PDF page. Ensure a compatible PDF library (pypdfium2, PyMuPDF, or pdf2image) is installed.")

def pdf_to_pil_images(pdf_input: Union[str, bytes], scale: float = 2.5) -> List[Image.Image]:
    """Convert all pages of a PDF document into a list of PIL Images."""
    count = get_pdf_page_count(pdf_input)
    images = []
    for i in range(count):
        img = render_pdf_page(pdf_input, page_index=i, scale=scale)
        images.append(img)
    return images
