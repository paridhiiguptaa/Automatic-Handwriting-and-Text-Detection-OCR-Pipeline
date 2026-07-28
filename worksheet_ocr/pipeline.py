import cv2
import numpy as np
import logging
from typing import Dict, Any, Union

from .config import PipelineConfig
from .preprocessing import ImagePreprocessor
from .text_detection import TextDetector
from .ocr_recognition import OCRRecognizer
from .reading_order import ReadingOrderSorter
from .layout_analysis import DocumentLayoutAnalyzer
from .document_reconstructor import DocumentReconstructor

logger = logging.getLogger(__name__)

class NotebookOCRPipeline:
    """
    Master Pipeline Orchestrator for Milestone 2: Document Layout Understanding & Notebook Structure Reconstruction.
    Flow:
      Preprocessing -> Text Region Detection -> Hybrid OCR -> Reading Order Sorter -> Layout Analyzer -> Document Reconstructor.
    """

    def __init__(self, config=None):
        self.config = config or PipelineConfig()
        self.preprocessor = ImagePreprocessor(config=self.config)
        self.detector = TextDetector(config=self.config)
        self.recognizer = OCRRecognizer(config=self.config)
        self.reading_sorter = ReadingOrderSorter(config=self.config)
        self.layout_analyzer = DocumentLayoutAnalyzer(config=self.config)
        self.reconstructor = DocumentReconstructor(config=self.config)

    def process_image(self, image_input: Union[str, np.ndarray, Any], page_num: int = 1) -> Dict[str, Any]:
        """
        Processes notebook page/image input and returns full structured payload.
        """
        # 1. Preprocessing
        page_idx = max(0, page_num - 1)
        cleaned_bgr, prep_meta = self.preprocessor.process(image_input, page_index=page_idx)

        # 2. Text Detection
        raw_regions = self.detector.detect(cleaned_bgr)

        # 3. Text Recognition
        recognized_regions = []
        for reg in raw_regions:
            rec_reg = self.recognizer.recognize_region(reg)
            recognized_regions.append(rec_reg)

        # 4. Reading Order Sorting
        img_w = cleaned_bgr.shape[1]
        sorted_regions = self.reading_sorter.sort_regions(recognized_regions, image_width=img_w)

        # 5. Milestone 2: Document Layout Analysis
        layout_regions = self.layout_analyzer.analyze_layout(sorted_regions, image_width=img_w)

        # 6. Milestone 2: Document Reconstruction (Formatted Markdown & Tree)
        formatted_markdown = self.reconstructor.reconstruct_markdown(layout_regions)
        document_tree = self.reconstructor.build_layout_tree(layout_regions)

        # 7. Generate Multi-Color Structural Bounding Box Visual Overlay
        annotated_bgr = self._draw_structural_bounding_boxes(cleaned_bgr, layout_regions)

        # 8. Build JSON-safe Region List
        clean_regions = []
        conf_sum = 0.0
        role_counts = {}

        for r in layout_regions:
            c = float(r.get("confidence", 0.0))
            conf_sum += c
            role = r.get("layout_role", "paragraph")
            role_counts[role] = role_counts.get(role, 0) + 1

            clean_regions.append({
                "id": int(r.get("id", 0)),
                "reading_order": int(r.get("reading_order", 0)),
                "layout_role": role,
                "indentation_level": int(r.get("indentation_level", 0)),
                "font_height_ratio": float(r.get("font_height_ratio", 1.0)),
                "bbox": [int(v) for v in r.get("bbox", [0, 0, 0, 0])],
                "region_type": r.get("region_type", "printed"),
                "text": r.get("extracted_text", ""),
                "confidence": round(c, 4),
                "model_used": r.get("recognition_model_used", "Unknown")
            })

        avg_conf = round(conf_sum / max(1, len(clean_regions)), 4)

        result_payload = {
            "milestone": "Milestone 2: Document Layout Understanding & Notebook Structure",
            "summary": {
                "total_text_regions": len(clean_regions),
                "average_confidence": avg_conf,
                "layout_role_counts": role_counts
            },
            "preprocessing_metadata": prep_meta,
            "reconstructed_text": formatted_markdown,
            "formatted_markdown": formatted_markdown,
            "document_tree": document_tree,
            "regions": clean_regions,
            "cleaned_bgr": cleaned_bgr,
            "annotated_bgr": annotated_bgr
        }

        return result_payload

    def _draw_structural_bounding_boxes(self, img_bgr: np.ndarray, regions: list) -> np.ndarray:
        """
        Annotate regions with color-coded badges based on layout_role:
        - Blue: Headings
        - Green: Question Labels
        - Orange: Answer Labels
        - Purple: Math Expressions
        - Yellow: Bullet Items
        - Gray: Paragraphs
        """
        annotated = img_bgr.copy()
        
        # Color mapping (BGR)
        role_colors = {
            "heading_1": (255, 120, 0),      # Blue
            "heading_2": (230, 80, 0),       # Dark Blue
            "question_label": (0, 200, 0),   # Green
            "answer_label": (0, 165, 255),   # Orange
            "math_expression": (200, 0, 200),# Purple
            "bullet_item": (0, 220, 220),    # Yellow
            "paragraph": (100, 100, 100)     # Gray
        }

        for reg in regions:
            bbox = reg.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            role = reg.get("layout_role", "paragraph")
            order_id = reg.get("reading_order", reg.get("id", 1))
            color = role_colors.get(role, (120, 120, 120))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Badge Label
            label = f"#{order_id} [{role.upper()}]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 4)
            badge_y2 = y1
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + text_w + 4, badge_y2), color, -1)
            cv2.putText(annotated, label, (x1 + 2, max(10, y1 - 2)), font, font_scale, (255, 255, 255), thickness)

        return annotated
