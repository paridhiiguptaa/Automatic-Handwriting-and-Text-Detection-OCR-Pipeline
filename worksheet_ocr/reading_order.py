import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ReadingOrderSorter:
    """
    Stage 4: Reconstructs natural notebook reading order (top-to-bottom, left-to-right).
    Groups text regions into visual lines using bounding box spatial geometry.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def sort_regions(self, regions: List[Dict[str, Any]], image_width: int = 2048) -> List[Dict[str, Any]]:
        """
        Sorts regions into reading order.
        Each region in the output list will contain:
        - `reading_order`: 1-based index integer
        - `id`: updated sequential ID
        """
        if not regions:
            return []

        # Calculate bounding box metrics
        processed_regions = []
        for r in regions:
            bbox = r.get("bbox", [0, 0, 0, 0])
            x_min, y_min, x_max, y_max = bbox
            h = max(1, y_max - y_min)
            w = max(1, x_max - x_min)
            y_center = y_min + (h / 2.0)
            
            reg_copy = dict(r)
            reg_copy["_y_center"] = y_center
            reg_copy["_height"] = h
            processed_regions.append(reg_copy)

        # 1. Initial sort by y_min
        processed_regions.sort(key=lambda r: r["bbox"][1])

        # 2. Group into lines based on vertical overlap
        lines = []
        tolerance = getattr(self.config, "LINE_VERTICAL_TOLERANCE_PX", 18)

        for reg in processed_regions:
            y_min = reg["bbox"][1]
            y_max = reg["bbox"][3]
            y_center = reg["_y_center"]

            assigned_line = None
            for line in lines:
                line_y_avg = sum(item["_y_center"] for item in line) / len(line)
                # Check if region vertically overlaps with line average y_center
                if abs(y_center - line_y_avg) <= tolerance or (y_min <= line_y_avg <= y_max):
                    assigned_line = line
                    break

            if assigned_line is not None:
                assigned_line.append(reg)
            else:
                lines.append([reg])

        # 3. Sort lines top-to-bottom by average y_center
        lines.sort(key=lambda line: sum(item["_y_center"] for item in line) / len(line))

        # 4. Within each line, sort left-to-right by x_min
        sorted_regions = []
        order_idx = 1
        for line in lines:
            line.sort(key=lambda item: item["bbox"][0])
            for reg in line:
                reg.pop("_y_center", None)
                reg.pop("_height", None)
                reg["reading_order"] = order_idx
                reg["id"] = order_idx
                sorted_regions.append(reg)
                order_idx += 1

        return sorted_regions

    def reconstruct_text(self, sorted_regions: List[Dict[str, Any]]) -> str:
        """
        Builds natural reading order text string from sorted regions.
        """
        if not sorted_regions:
            return ""

        lines_text = []
        current_line_idx = None
        current_line_tokens = []

        # We can group tokens into lines using reading order or spatial y proximity
        for reg in sorted_regions:
            text = reg.get("extracted_text", "").strip()
            if not text:
                continue

            current_line_tokens.append(text)

        return "\n".join(current_line_tokens)
