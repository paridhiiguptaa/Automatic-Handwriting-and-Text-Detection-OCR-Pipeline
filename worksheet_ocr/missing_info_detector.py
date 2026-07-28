import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MissingInfoDetector:
    """
    Missing Information Detection Module.
    Identifies missing answers, omitted units, skipped steps, unlabeled figures, and incomplete explanations.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def detect_missing_info(self, qa_pairs: List[Dict[str, Any]], layout_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Scan questions, answers, and diagrams for omitted information.
        """
        missing_items = []

        for qa in qa_pairs:
            q_id = qa.get("question_id")
            q_text = qa.get("question_text", "")
            a_text = qa.get("answer_text", "").strip()

            # 1. Completely unanswered questions
            if not a_text:
                missing_items.append({
                    "question_id": q_id,
                    "type": "unanswered_question",
                    "severity": "high",
                    "explanation": "Question was detected on the page but has no corresponding student answer."
                })
                continue

            # 2. Incomplete sentence / truncated answer
            if a_text.endswith((",", "and", "the", "because", "with", "is", "of", "to")):
                missing_items.append({
                    "question_id": q_id,
                    "type": "unfinished_sentence",
                    "severity": "medium",
                    "explanation": f"Answer appears to end abruptly or incomplete: '{a_text[-25:]}'."
                })

            # 3. Omitted derivation / intermediate steps
            if ("find" in q_text.lower() or "solve" in q_text.lower()) and len(q_text) > 20 and len(a_text.split()) < 3:
                missing_items.append({
                    "question_id": q_id,
                    "type": "skipped_calculation_steps",
                    "severity": "medium",
                    "explanation": "Answer contains only a final result. Intermediate calculation steps should be shown."
                })

            # 4. Unlabeled diagrams
            associated_diagrams = qa.get("associated_diagrams", [])
            if associated_diagrams and "label" in q_text.lower() and "label" not in a_text.lower():
                missing_items.append({
                    "question_id": q_id,
                    "type": "unlabeled_diagram",
                    "severity": "medium",
                    "explanation": "Diagram is drawn, but labels or captions were omitted."
                })

        return missing_items
