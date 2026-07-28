import cv2
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TextDetector:
    """
    Stage 2: Text Region & Line Detection Engine for Notebooks and Worksheets.
    Combines PaddleOCR DBNet deep-learning detector with a multi-scale OpenCV morphological
    contour fallback segmenter, followed by an aggressive horizontal box merging algorithm
    to group word fragments into full text lines.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()
        self._paddle_det = None

    def _get_paddle_detector(self):
        if self._paddle_det is None:
            from paddleocr import PaddleOCR
            try:
                self._paddle_det = PaddleOCR(
                    lang=self.config.PADDLE_LANG,
                    enable_mkldnn=False
                )
            except Exception as e:
                logger.warning(f"PaddleOCR detector fallback init: {e}")
                try:
                    self._paddle_det = PaddleOCR(enable_mkldnn=False)
                except Exception:
                    self._paddle_det = PaddleOCR()
        return self._paddle_det

    def detect(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text region bounding boxes across input image and merge them into complete lines.
        Returns:
            List[dict] containing bbox [x1, y1, x2, y2], region_type, det_confidence, crop_bgr.
        """
        img_h, img_w = img_bgr.shape[:2]
        detected_regions = []

        # 1. Primary: Deep Learning Detector (PaddleOCR DBNet)
        try:
            detector = self._get_paddle_detector()
            results = detector.ocr(img_bgr)
            
            if results and len(results) > 0 and results[0] is not None:
                res_item = results[0]
                det_boxes = []

                if isinstance(res_item, dict):
                    det_boxes = res_item.get("dt_polys", [])
                elif isinstance(res_item, list):
                    det_boxes = res_item

                for idx, box_item in enumerate(det_boxes):
                    pts_data = None
                    if isinstance(box_item, (list, tuple, np.ndarray)):
                        pts_data = box_item[0] if (isinstance(box_item, (list, tuple)) and len(box_item) > 0 and isinstance(box_item[0], (list, tuple, np.ndarray))) else box_item
                    
                    if pts_data is not None:
                        pts = np.array(pts_data, dtype=np.int32)
                        if pts.ndim == 2 and pts.shape[0] >= 4:
                            x_min = max(0, int(np.min(pts[:, 0])))
                            y_min = max(0, int(np.min(pts[:, 1])))
                            x_max = min(img_w, int(np.max(pts[:, 0])))
                            y_max = min(img_h, int(np.max(pts[:, 1])))

                            w, h = x_max - x_min, y_max - y_min
                            if w >= 6 and h >= 6:
                                crop_bgr = img_bgr[y_min:y_max, x_min:x_max]
                                region_type = self._classify_region_type(crop_bgr)
                                detected_regions.append({
                                    "id": len(detected_regions) + 1,
                                    "bbox": [x_min, y_min, x_max, y_max],
                                    "polygon": pts.tolist(),
                                    "det_confidence": 0.95,
                                    "region_type": region_type,
                                    "crop_bgr": crop_bgr
                                })
        except Exception as e:
            logger.warning(f"PaddleOCR detection notice: {e}. Falling back to OpenCV morphological detector.")

        # 2. Secondary Fallback: Multi-Scale OpenCV Contour Line Detector
        if not detected_regions or len(detected_regions) < 2:
            logger.info("Running OpenCV layout text detector fallback...")
            detected_regions = self._opencv_layout_detect(img_bgr)

        # 3. Merge Horizontally Adjacent Word Boxes into Complete Text Lines
        merged_regions = self._merge_horizontal_boxes(detected_regions, img_bgr)

        return merged_regions

    def _merge_horizontal_boxes(self, regions: List[Dict[str, Any]], img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Merges horizontally adjacent word bounding boxes on the same line into unified line boxes.
        Ensures OCR recognizers receive complete line text instead of partial word fragments.
        """
        if not regions:
            return []

        img_h, img_w = img_bgr.shape[:2]

        # Sort regions top-to-bottom by y_min
        sorted_regs = sorted(regions, key=lambda r: r["bbox"][1])

        merged_lines = []
        vertical_overlap_tol = 16  # px tolerance for same horizontal line

        for reg in sorted_regs:
            x1, y1, x2, y2 = reg["bbox"]
            y_center = (y1 + y2) / 2.0

            assigned = False
            for line in merged_lines:
                line_y_centers = [(b[1] + b[3]) / 2.0 for b in line["boxes"]]
                avg_line_y = sum(line_y_centers) / len(line_y_centers)

                # Check if this box is on the same line
                if abs(y_center - avg_line_y) <= vertical_overlap_tol or (y1 <= avg_line_y <= y2):
                    line_x_max = max(b[2] for b in line["boxes"])
                    line_x_min = min(b[0] for b in line["boxes"])
                    gap = x1 - line_x_max

                    # Allow horizontal word gap up to 10% of image width
                    if gap <= int(img_w * 0.10) and (x2 - line_x_min) <= int(img_w * 0.96):
                        line["boxes"].append(reg["bbox"])
                        line["confidences"].append(reg.get("det_confidence", 0.9))
                        line["types"].append(reg.get("region_type", "printed"))
                        assigned = True
                        break

            if not assigned:
                merged_lines.append({
                    "boxes": [reg["bbox"]],
                    "confidences": [reg.get("det_confidence", 0.9)],
                    "types": [reg.get("region_type", "printed")]
                })

        # Reconstruct merged line bounding boxes
        final_regions = []
        for idx, line in enumerate(merged_lines, 1):
            boxes = line["boxes"]
            x_min = max(0, min(b[0] for b in boxes))
            y_min = max(0, min(b[1] for b in boxes))
            x_max = min(img_w, max(b[2] for b in boxes))
            y_max = min(img_h, max(b[3] for b in boxes))

            # Add padding around bounding box to capture full character ascenders & descenders
            pad_x = max(3, int((x_max - x_min) * 0.015))
            pad_y = max(3, int((y_max - y_min) * 0.08))

            x_min_padded = max(0, x_min - pad_x)
            y_min_padded = max(0, y_min - pad_y)
            x_max_padded = min(img_w, x_max + pad_x)
            y_max_padded = min(img_h, y_max + pad_y)

            crop_bgr = img_bgr[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
            region_type = "handwritten" if "handwritten" in line["types"] else "printed"
            avg_conf = sum(line["confidences"]) / float(len(line["confidences"]))

            final_regions.append({
                "id": idx,
                "bbox": [x_min_padded, y_min_padded, x_max_padded, y_max_padded],
                "polygon": [[x_min_padded, y_min_padded], [x_max_padded, y_min_padded], [x_max_padded, y_max_padded], [x_min_padded, y_max_padded]],
                "det_confidence": round(avg_conf, 4),
                "region_type": region_type,
                "crop_bgr": crop_bgr
            })

        return final_regions

    def _opencv_layout_detect(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Multi-scale Morphological Text Line Segmenter."""
        img_h, img_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )

        kernel_w = max(15, int(img_w * 0.06))
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 2))
        dilated = cv2.dilate(binary, kernel_line, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        min_w, min_h = 10, 6

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if w < min_w or h < min_h:
                continue
            if w > img_w * 0.98 and h > img_h * 0.95:
                continue

            x_min, y_min = max(0, x), max(0, y)
            x_max, y_max = min(img_w, x + w), min(img_h, y + h)

            crop_bgr = img_bgr[y_min:y_max, x_min:x_max]
            region_type = self._classify_region_type(crop_bgr)

            regions.append({
                "id": len(regions) + 1,
                "bbox": [x_min, y_min, x_max, y_max],
                "polygon": [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                "det_confidence": 0.85,
                "region_type": region_type,
                "crop_bgr": crop_bgr
            })

        return regions

    def _classify_region_type(self, crop_bgr: np.ndarray) -> str:
        """Classifies region as 'handwritten' or 'printed' using visual heuristics."""
        if crop_bgr is None or crop_bgr.size == 0:
            return "printed"

        h, w = crop_bgr.shape[:2]
        aspect_ratio = h / float(w)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        var = np.var(gray)

        if aspect_ratio > self.config.ASPECT_RATIO_HANDWRITING_THRESHOLD or var > self.config.INTENSITY_VARIANCE_HANDWRITING_MIN:
            return "handwritten"
        
        return "printed"
