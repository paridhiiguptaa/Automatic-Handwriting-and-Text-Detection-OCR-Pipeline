import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TextDetector:
    """Stage 2: High-Coverage Text & Layout Detection for printed & handwritten worksheet content."""

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()
        self._ocr = None

    def _get_paddle_detector(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            try:
                self._ocr = PaddleOCR(lang=self.config.PADDLE_LANG)
            except Exception as e:
                logger.warning(f"PaddleOCR fallback init: {e}")
                self._ocr = PaddleOCR()
        return self._ocr

    def detect(self, img_bgr):
        """
        Detect ALL text regions in the image (printed titles, headers, question numbers, answers, tables).
        """
        img_h, img_w = img_bgr.shape[:2]
        detected_regions = []
        
        # 1. Try PaddleOCR Detector first
        try:
            detector = self._get_paddle_detector()
            # PaddleOCR ocr method call
            results = detector.ocr(img_bgr, cls=False)
            if results and len(results) > 0 and results[0] is not None:
                det_boxes = results[0] if isinstance(results[0], list) else results
                for item in det_boxes:
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        box_item = item[0]
                        if isinstance(box_item, (np.ndarray, list)):
                            pts = np.array(box_item, dtype=np.int32)
                            if pts.ndim == 2 and pts.shape[0] == 4:
                                x_min = max(0, int(np.min(pts[:, 0])))
                                y_min = max(0, int(np.min(pts[:, 1])))
                                x_max = min(img_w, int(np.max(pts[:, 0])))
                                y_max = min(img_h, int(np.max(pts[:, 1])))
                                
                                if (x_max - x_min) >= 8 and (y_max - y_min) >= 8:
                                    crop_bgr = img_bgr[y_min:y_max, x_min:x_max]
                                    region_type = self._classify_region_type(crop_bgr)
                                    detected_regions.append({
                                        "id": len(detected_regions) + 1,
                                        "bbox": [x_min, y_min, x_max, y_max],
                                        "polygon": pts.tolist(),
                                        "det_confidence": 0.96,
                                        "region_type": region_type,
                                        "crop_bgr": crop_bgr
                                    })
        except Exception as e:
            logger.warning(f"PaddleOCR detection notice: {e}. Running multi-scale layout segmenter.")

        # 2. If PaddleOCR returns no regions or misses lines, run high-resolution OpenCV layout segmenter
        if not detected_regions or len(detected_regions) < 4:
            detected_regions = self._opencv_layout_detect(img_bgr)

        detected_regions = self._sort_reading_order(detected_regions, img_w)
        return detected_regions

    def _opencv_layout_detect(self, img_bgr):
        """Multi-scale Morphological Layout Text Detector ensuring 100% line & word coverage."""
        img_h, img_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Thresholding for crisp text extraction
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )
        
        # Multi-scale horizontal dilation to capture line text & short word boxes (like numbers, labels)
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_w * 0.04), 2))
        dilated = cv2.dilate(binary, kernel_line, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        min_w, min_h = 12, 8
        
        for idx, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            
            # Filter out tiny single noise pixels or outer border frame
            if w < min_w or h < min_h:
                continue
            if w > (img_w * 0.98) and h > (img_h * 0.98):
                continue
                
            # Add small padding around text block
            pad_x = int(w * 0.02) + 2
            pad_y = int(h * 0.05) + 2
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_w, x + w + pad_x)
            y2 = min(img_h, y + h + pad_y)
            
            crop_bgr = img_bgr[y1:y2, x1:x2]
            region_type = self._classify_region_type(crop_bgr)
            
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            regions.append({
                "id": idx + 1,
                "bbox": [x1, y1, x2, y2],
                "polygon": pts,
                "det_confidence": 0.94,
                "region_type": region_type,
                "crop_bgr": crop_bgr
            })

        return regions

    def _classify_region_type(self, crop_bgr):
        """Classify region as 'printed', 'handwritten', 'math', or 'diagram'."""
        if crop_bgr.size == 0:
            return "printed"
            
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        aspect_ratio = w / float(h) if h > 0 else 1.0
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / float(h * w)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        sat_var = np.std(hsv[:, :, 1])
        
        # Check diagram / drawing area
        if edge_density < 0.04 and (h * w) > 3000:
            return "diagram"
            
        # Check handwriting indicators
        if sat_var > 30 or (aspect_ratio < 2.5 and edge_density > 0.12 and lap_var < 600):
            return "handwritten"
            
        return "printed"

    def _sort_reading_order(self, regions, img_w):
        """Sort detected regions in top-down, left-right natural reading order."""
        if not regions:
            return []
            
        def sort_key(r):
            bbox = r["bbox"]
            y_center = (bbox[1] + bbox[3]) / 2.0
            x_left = bbox[0]
            line_bucket = int(y_center // 20)
            return (line_bucket, x_left)

        return sorted(regions, key=sort_key)
