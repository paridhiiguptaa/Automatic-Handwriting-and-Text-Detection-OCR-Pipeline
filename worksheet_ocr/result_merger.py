import numpy as np
import logging

logger = logging.getLogger(__name__)

class ResultMerger:
    """Stage 6: Merge OCR and VLM predictions into structured JSON output preserving layout and reading order."""

    def merge(self, processed_regions, page_num=1, document_type="worksheet", intelligence_data=None):
        """
        Consolidate region outputs and AI Notebook Intelligence analysis into final JSON schema.
        """
        output_regions = []
        confidence_scores = []

        for reg in processed_regions:
            bbox = reg.get("bounding_box", reg.get("bbox", [0, 0, 0, 0]))
            region_type = reg.get("region_type", "printed")
            
            # Determine final model name used
            if reg.get("vlm_invoked", False):
                model_used = reg.get("model", "SmolVLM2")
            else:
                model_used = reg.get("recognition_model_used", "PaddleOCR")

            conf = reg.get("confidence", 0.0)
            text = reg.get("text", reg.get("extracted_text", ""))

            # Format item matching required schema
            formatted_region = {
                "bbox": [int(v) for v in bbox],
                "type": region_type,
                "model": model_used,
                "confidence": round(float(conf), 2),
                "text": text
            }

            output_regions.append(formatted_region)
            confidence_scores.append(conf)

        # Compute overall document confidence score
        overall_conf = float(np.mean(confidence_scores)) if confidence_scores else 0.0

        document_json = {
            "document_type": document_type,
            "page": page_num,
            "regions": output_regions,
            "overall_confidence": round(overall_conf, 2)
        }

        # Merge AI Notebook Intelligence Data if provided
        if intelligence_data and isinstance(intelligence_data, dict):
            document_json.update(intelligence_data)

        return document_json

