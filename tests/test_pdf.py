import os
import tempfile
import unittest
import numpy as np
import cv2
from PIL import Image

from worksheet_ocr import (
    WorksheetOCRPipeline,
    PipelineConfig,
    is_pdf_file,
    get_pdf_page_count,
    render_pdf_page,
    pdf_to_pil_images
)

class TestPDFSupport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary synthetic multi-page PDF file for testing
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.pdf_path = os.path.join(cls.temp_dir.name, "test_worksheet.pdf")
        
        # Create 2 synthetic page images
        page1_img = Image.new("RGB", (600, 800), color=(255, 255, 255))
        page2_img = Image.new("RGB", (600, 800), color=(240, 240, 240))
        
        # Save as 2-page PDF using PIL
        page1_img.save(cls.pdf_path, save_all=True, append_images=[page2_img])

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_is_pdf_file(self):
        self.assertTrue(is_pdf_file(self.pdf_path))
        with open(self.pdf_path, "rb") as f:
            pdf_bytes = f.read()
        self.assertTrue(is_pdf_file(pdf_bytes))
        self.assertFalse(is_pdf_file("non_existent_file.png"))

    def test_get_pdf_page_count(self):
        count = get_pdf_page_count(self.pdf_path)
        self.assertEqual(count, 2)

    def test_render_pdf_page(self):
        img_page0 = render_pdf_page(self.pdf_path, page_index=0)
        self.assertIsInstance(img_page0, Image.Image)
        self.assertGreater(img_page0.width, 0)
        self.assertGreater(img_page0.height, 0)

        img_page1 = render_pdf_page(self.pdf_path, page_index=1)
        self.assertIsInstance(img_page1, Image.Image)

    def test_pdf_to_pil_images(self):
        images = pdf_to_pil_images(self.pdf_path)
        self.assertEqual(len(images), 2)
        self.assertIsInstance(images[0], Image.Image)

    def test_pipeline_process_pdf_single_page(self):
        pipeline = WorksheetOCRPipeline()
        result = pipeline.process_image(self.pdf_path, page_num=1)
        self.assertIn("layout_understanding", result)
        self.assertIn("subject_identification", result)
        self.assertIn("learning_analytics", result)
        self.assertIn("annotated_bgr", result)

    def test_pipeline_process_pdf_all_pages(self):
        pipeline = WorksheetOCRPipeline()
        results = pipeline.process_pdf(self.pdf_path)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIn("subject_identification", results[0])
        self.assertIn("subject_identification", results[1])

if __name__ == "__main__":
    unittest.main()
