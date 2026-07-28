import re
import cv2
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DocumentLayoutAnalyzer:
    """
    Stage 5: Document Layout Understanding Engine.
    Analyzes spatial geometry, bounding box heights, text patterns, and indentation
    to classify OCR text regions into structural document roles:
    - heading_1 / heading_2
    - question_label
    - answer_label
    - bullet_item
    - math_expression
    - paragraph
    - table_cell / diagram
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def analyze_layout(self, sorted_regions: List[Dict[str, Any]], image_width: int = 2048) -> List[Dict[str, Any]]:
        """
        Processes reading-order sorted regions and enriches each dict with:
        - `layout_role`: str ('heading_1', 'heading_2', 'question_label', 'answer_label', 'bullet_item', 'math_expression', 'paragraph')
        - `indentation_level`: int (0 = left aligned, 1 = indented answer/list, 2 = sub-indented)
        - `font_height_ratio`: float (box height / average height)
        """
        if not sorted_regions:
            return []

        # Calculate height baseline metrics for heading detection
        heights = [max(1, r["bbox"][3] - r["bbox"][1]) for r in sorted_regions]
        avg_height = sum(heights) / float(len(heights)) if heights else 20.0
        min_x = min(r["bbox"][0] for r in sorted_regions) if sorted_regions else 0

        analyzed_regions = []
        for r in sorted_regions:
            reg = dict(r)
            bbox = reg.get("bbox", [0, 0, 0, 0])
            text = reg.get("extracted_text", "").strip()
            
            x1, y1, x2, y2 = bbox
            box_h = max(1, y2 - y1)
            box_w = max(1, x2 - x1)
            h_ratio = box_h / avg_height

            # Determine Indentation Level (based on horizontal left margin offset)
            offset_x = max(0, x1 - min_x)
            if offset_x < int(image_width * 0.04):
                indent_level = 0
            elif offset_x < int(image_width * 0.12):
                indent_level = 1
            else:
                indent_level = 2

            # Classify Layout Role
            role = self._classify_role(text, h_ratio, box_w, image_width, indent_level)

            reg["layout_role"] = role
            reg["indentation_level"] = indent_level
            reg["font_height_ratio"] = round(h_ratio, 2)

            analyzed_regions.append(reg)

        return analyzed_regions

    def _classify_role(self, text: str, h_ratio: float, box_w: int, image_w: int, indent_level: int) -> str:
        if not text:
            return "paragraph"

        clean_t = text.strip()

        # 1. Question Label Pattern Detection (e.g. Q1:, Q.1, Question 1, 1., 1), a), (a))
        q_patterns = [
            r'^(?:Q|q|Question)\s*[\.\:\-\d]+',
            r'^\(?\d{1,2}\)?[\.\:\-\s]+[A-Z]',
            r'^\(?[a-zA-Z]\)[\.\:\-\s]+'
        ]
        for pat in q_patterns:
            if re.match(pat, clean_t):
                return "question_label"

        # 2. Answer Label Pattern Detection (e.g. Ans:, Answer:, Sol:, Solution:)
        a_patterns = [
            r'^(?:Ans|ANS|Answer|Sol|Solution)\s*[\.\:\-\=]'
        ]
        for pat in a_patterns:
            if re.match(pat, clean_t, re.IGNORECASE):
                return "answer_label"

        # 3. Bullet Point Pattern Detection
        if clean_t.startswith(('•', '-', '*', '➢', '▪', '→')):
            return "bullet_item"

        # 4. Heading Detection (Large height ratio or ALL CAPS title)
        if h_ratio >= 1.6 and box_w > int(image_w * 0.25):
            return "heading_1"
        if h_ratio >= 1.35 or (clean_t.isupper() and len(clean_t) > 4 and indent_level == 0):
            return "heading_2"

        # 5. Math Expression Detection
        math_symbols = ['=', '+', '-', '×', '÷', '/', '√', '^', '∫', '∑', '\\frac', '\\int', 'sin', 'cos', 'tan', '%']
        symbol_count = sum(1 for sym in math_symbols if sym in clean_t)
        digit_count = sum(1 for char in clean_t if char.isdigit())
        if (symbol_count >= 2 and digit_count >= 1) or (re.search(r'^\d+\s*[\+\-\*\/\=]\s*\d+', clean_t)):
            return "math_expression"

        return "paragraph"
