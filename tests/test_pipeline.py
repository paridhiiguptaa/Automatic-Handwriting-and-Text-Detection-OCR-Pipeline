import os
import unittest
import numpy as np
import cv2
from worksheet_ocr.config import PipelineConfig
from worksheet_ocr.preprocessing import ImagePreprocessor
from worksheet_ocr.confidence_validation import ConfidenceValidator
from worksheet_ocr.vlm_fallback import VLMFallback
from worksheet_ocr.result_merger import ResultMerger

class TestWorksheetPipelineStages(unittest.TestCase):

    def setUp(self):
        self.config = PipelineConfig()
        # Create a synthetic worksheet crop image for testing
        self.test_img = np.full((300, 400, 3), 240, dtype=np.uint8)
        # Add black printed text representation
        cv2.putText(self.test_img, "Question 1. What color is the grass?", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        # Add pencil handwritten answer representation (gray)
        cv2.putText(self.test_img, "green", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (110, 110, 110), 2)

    def test_stage1_preprocessing(self):
        preprocessor = ImagePreprocessor(self.config)
        processed, meta = preprocessor.process(self.test_img)
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, self.test_img.shape)
        self.assertIn("skew_angle_deg", meta)

    def test_stage4_confidence_validation(self):
        validator = ConfidenceValidator(self.config)
        
        # Test High confidence printed region
        region_high = {
            "extracted_text": "Write the correct answer.",
            "confidence": 0.95,
            "region_type": "printed",
            "crop_bgr": self.test_img[10:80, 10:350]
        }
        res_high = validator.validate(region_high)
        self.assertTrue(res_high["validation"]["accepted"])
        self.assertFalse(res_high["validation"]["needs_vlm"])

        # Test Low confidence handwritten region
        region_low = {
            "extracted_text": "gree?",
            "confidence": 0.52,
            "region_type": "handwritten",
            "crop_bgr": self.test_img[100:150, 10:200]
        }
        res_low = validator.validate(region_low)
        self.assertTrue(res_low["validation"]["needs_vlm"])
        self.assertIn("Low OCR confidence (0.52 < 0.6)", res_low["validation"]["reasons"][0])

    def test_stage5_vlm_safeguard(self):
        vlm = VLMFallback(self.config)
        
        # Test low confidence crop fallback
        region = {
            "crop_bgr": np.zeros((10, 10, 3), dtype=np.uint8),
            "extracted_text": ""
        }
        res = vlm.process_region(region)
        self.assertEqual(res["text"], "Low confidence.")
        self.assertTrue(res["vlm_invoked"])

    def test_stage6_result_merger(self):
        merger = ResultMerger()
        mock_regions = [
            {
                "bounding_box": [10, 20, 100, 50],
                "region_type": "printed",
                "recognition_model_used": "PaddleOCR",
                "confidence": 0.98,
                "text": "Name:"
            },
            {
                "bounding_box": [110, 20, 250, 50],
                "region_type": "handwritten",
                "model": "TrOCR-Handwritten",
                "vlm_invoked": False,
                "confidence": 0.89,
                "text": "Vritika"
            }
        ]
        doc_json = merger.merge(mock_regions, page_num=1)
        self.assertEqual(doc_json["document_type"], "worksheet")
        self.assertEqual(len(doc_json["regions"]), 2)
        self.assertEqual(doc_json["regions"][0]["model"], "PaddleOCR")
        self.assertEqual(doc_json["regions"][1]["text"], "Vritika")

if __name__ == "__main__":
    unittest.main()
