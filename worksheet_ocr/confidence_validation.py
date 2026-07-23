import re
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ConfidenceValidator:
    """Stage 4: Confidence Validation & Risk Flagging to trigger selective VLM fallback."""

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def validate(self, region):
        """
        Evaluate OCR result for a text region.
        Returns tuple: (is_accepted: bool, reason: str, needs_vlm: bool)
        """
        text = region.get("extracted_text", "")
        conf = region.get("confidence", 0.0)
        region_type = region.get("region_type", "printed")
        crop_bgr = region.get("crop_bgr", None)
        
        reasons = []

        # 1. Low Confidence check
        if conf < self.config.LOW_CONFIDENCE_THRESHOLD:
            reasons.append(f"Low OCR confidence ({conf:.2f} < {self.config.LOW_CONFIDENCE_THRESHOLD})")
            
        # 2. Empty text or unreadable handwriting check
        if not text or len(text.strip()) == 0:
            reasons.append("Empty/Unreadable recognized text")

        # 3. Diagram / Visual Element check
        if region_type == "diagram":
            reasons.append("Diagram / Drawing region")

        # 4. Mathematical Notation & Symbols check
        if region_type == "math" or self._has_math_notation(text):
            reasons.append("Mathematical notation or equation")

        # 5. Overlapping writing / Strikethrough check
        if crop_bgr is not None and self._detect_strikethrough_or_overlap(crop_bgr):
            reasons.append("Overlapping writing / Strikethrough / Teacher mark detected")

        # 6. Garbage text / OCR noise check (e.g. repeated symbols, illegal chars)
        if self._is_garbage_text(text):
            reasons.append("Abnormal character sequence / OCR artifact")

        # Decision
        needs_vlm = len(reasons) > 0 or conf < self.config.HIGH_CONFIDENCE_THRESHOLD
        is_accepted = not needs_vlm

        region["validation"] = {
            "accepted": is_accepted,
            "needs_vlm": needs_vlm,
            "reasons": reasons if reasons else ["High confidence OCR match"]
        }

        return region

    def _has_math_notation(self, text):
        """Check if text contains mathematical symbols or expressions (e.g., +, -, *, x, /, =, ^, square roots)."""
        math_patterns = [
            r'[\+\-\*\/×÷=\^√]',
            r'\b\d+\s*[\+\-\*\/×÷=]\s*\d+',
            r'\b\d+[\/\.]\d+\b'
        ]
        for pat in math_patterns:
            if re.search(pat, text):
                return True
        return False

    def _detect_strikethrough_or_overlap(self, crop_bgr):
        """Detect horizontal strikethrough lines or teacher red pen overlap."""
        if crop_bgr is None or crop_bgr.size == 0:
            return False
            
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        
        # Red teacher mark detection in HSV (Teacher red ink)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2
        
        red_ratio = np.count_nonzero(red_mask) / float(crop_bgr.shape[0] * crop_bgr.shape[1])
        if red_ratio > 0.03: # Red teacher stroke present over writing
            return True

        # Horizontal line strikethrough check via morphology
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if w > 20 and h > 10:
            horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.4), 1))
            edges = cv2.Canny(gray, 50, 150)
            horiz_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horiz_kernel)
            if np.count_nonzero(horiz_lines) > (w * 0.3):
                return True

        return False

    def _is_garbage_text(self, text):
        """Detect random non-readable char sequences produced by OCR confusion."""
        if len(text) > 3 and len(set(text)) == 1:
            return True
        # Check non-ascii weird ratio
        non_alphanumeric = sum(1 for c in text if not c.isalnum() and c not in " .,-!?()[]")
        if len(text) > 0 and (non_alphanumeric / len(text)) > 0.4:
            return True
        return False
