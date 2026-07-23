import cv2
import numpy as np
import logging
from typing import List, Union, Dict, Any
from PIL import Image

from .config import PipelineConfig
from .preprocessing import ImagePreprocessor
from .text_detection import TextDetector
from .ocr_recognition import OCRRecognizer
from .confidence_validation import ConfidenceValidator
from .vlm_fallback import VLMFallback
from .result_merger import ResultMerger

logger = logging.getLogger(__name__)

class WorksheetOCRPipeline:
    """
    End-to-End CPU-Friendly Intelligent Worksheet Reading Pipeline.
    Integrates 6 Stages:
    1. Preprocessing (Deskew, Contrast CLAHE, Denoising, Pencil Stroke Preservation)
    2. Text Detection (PaddleOCR DB detector)
    3. Hybrid Recognition (PaddleOCR for Printed, TrOCR for Handwriting)
    4. Confidence Validation & Risk Flagging
    5. Selective VLM Fallback (SmolVLM2 for uncertain/complex crops)
    6. Structured JSON Result Merging
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.config.setup_environment()
        
        self.preprocessor = ImagePreprocessor(self.config)
        self.detector = TextDetector(self.config)
        self.recognizer = OCRRecognizer(self.config)
        self.validator = ConfidenceValidator(self.config)
        self.vlm_fallback = VLMFallback(self.config)
        self.merger = ResultMerger()

    def process_image(self, image_input: Union[str, Image.Image, np.ndarray], page_num: int = 1) -> Dict[str, Any]:
        """
        Process a single worksheet image through Stages 1 - 6.
        """
        logger.info(f"--- Processing Worksheet Image Page {page_num} ---")

        # Stage 1: Preprocessing
        preprocessed_bgr, prep_meta = self.preprocessor.process(image_input)
        
        # Stage 2: Text Detection
        detected_regions = self.detector.detect(preprocessed_bgr)
        logger.info(f"Stage 2: Detected {len(detected_regions)} text regions.")

        processed_regions = []
        vlm_count = 0

        for region in detected_regions:
            # Stage 3: Recognition
            region = self.recognizer.recognize_region(region)

            # Stage 4: Confidence Validation
            region = self.validator.validate(region)

            # Stage 5: VLM Fallback (ONLY if flagged)
            if region["validation"]["needs_vlm"]:
                vlm_count += 1
                logger.info(f"Region {region['id']} flagged for VLM fallback. Reasons: {region['validation']['reasons']}")
                region = self.vlm_fallback.process_region(region)

            processed_regions.append(region)

        logger.info(f"Stage 5: Invoked VLM on {vlm_count}/{len(detected_regions)} regions.")

        # Stage 6: Merge Results
        final_document = self.merger.merge(processed_regions, page_num=page_num)
        final_document["preprocessing_metadata"] = prep_meta
        
        # Attach annotated visualization image
        annotated_bgr = self.draw_visualization(preprocessed_bgr, processed_regions)
        final_document["annotated_bgr"] = annotated_bgr

        return final_document

    def process_batch(self, image_inputs: List[Union[str, Image.Image, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        Process multiple worksheet images sequentially while preserving page order.
        """
        results = []
        for idx, img_input in enumerate(image_inputs):
            page_res = self.process_image(img_input, page_num=idx + 1)
            results.append(page_res)
        return results

    def draw_visualization(self, img_bgr: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw color-coded bounding boxes on image:
        - Blue: PaddleOCR Printed Text
        - Green: TrOCR Handwritten Text
        - Purple / Orange: SmolVLM2 Fallback
        """
        vis_img = img_bgr.copy()
        
        for reg in regions:
            bbox = reg.get("bounding_box", reg.get("bbox", [0, 0, 0, 0]))
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            model = reg.get("model", reg.get("recognition_model_used", "PaddleOCR"))
            text = reg.get("text", reg.get("extracted_text", ""))
            conf = reg.get("confidence", 0.0)

            if "SmolVLM" in model or reg.get("vlm_invoked", False):
                color = (0, 165, 255) # Orange/Purple for VLM
                label = f"VLM ({conf:.2f})"
            elif "TrOCR" in model or reg.get("region_type") == "handwritten":
                color = (0, 200, 0) # Green for TrOCR
                label = f"TrOCR ({conf:.2f})"
            else:
                color = (255, 100, 0) # Blue for PaddleOCR
                label = f"Paddle ({conf:.2f})"

            # Draw bbox rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # Draw text label box
            caption = f"{label}: {text[:15]}"
            cv2.putText(vis_img, caption, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        return vis_img
