import os

# Configure environment for Paddle / Torch execution
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_DISABLE_PIR"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from .config import PipelineConfig
from .preprocessing import ImagePreprocessor
from .text_detection import TextDetector
from .ocr_recognition import OCRRecognizer
from .reading_order import ReadingOrderSorter
from .layout_analysis import DocumentLayoutAnalyzer
from .document_reconstructor import DocumentReconstructor
from .pipeline import NotebookOCRPipeline
from .pdf_utils import is_pdf_file, get_pdf_page_count, render_pdf_page, pdf_to_pil_images

WorksheetOCRPipeline = NotebookOCRPipeline

__all__ = [
    "NotebookOCRPipeline",
    "WorksheetOCRPipeline",
    "PipelineConfig",
    "ImagePreprocessor",
    "TextDetector",
    "OCRRecognizer",
    "ReadingOrderSorter",
    "DocumentLayoutAnalyzer",
    "DocumentReconstructor",
    "is_pdf_file",
    "get_pdf_page_count",
    "render_pdf_page",
    "pdf_to_pil_images"
]
