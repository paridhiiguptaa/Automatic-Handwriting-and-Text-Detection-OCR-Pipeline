import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Anomaly Detection System.
    Identifies unusual notebook patterns such as blank answers, repeated paragraphs, skipped questions, and layout bugs.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def detect_anomalies(self, qa_pairs: List[Dict[str, Any]], layout_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flag page layout and student work anomalies.
        """
        anomalies = []

        # 1. Blank space anomalies
        blank_spaces = layout_data.get("blank_spaces", [])
        for bs in blank_spaces:
            if bs.get("height_px", 0) > 150:
                anomalies.append({
                    "anomaly_type": "unusually_large_blank_space",
                    "severity": "low",
                    "details": f"Large blank region detected between y={bs.get('y_start')}px and y={bs.get('y_end')}px."
                })

        # 2. Repeated paragraph anomaly
        answers = [qa.get("answer_text", "").strip() for qa in qa_pairs if qa.get("answer_text")]
        for idx, ans in enumerate(answers):
            if len(ans) > 15 and answers.count(ans) > 1:
                anomalies.append({
                    "anomaly_type": "repeated_paragraph",
                    "severity": "medium",
                    "details": f"Identical text block repeated multiple times: '{ans[:30]}...'."
                })
                break

        # 3. Question numbering inconsistency check
        q_labels = [qa.get("question_label", "") for qa in qa_pairs if qa.get("question_label")]
        if len(q_labels) > 2:
            numbers = []
            for lbl in q_labels:
                nums = [int(n) for n in lbl.replace("Q", "").split(".") if n.isdigit()]
                if nums:
                    numbers.append(nums[0])
            if numbers and numbers != sorted(numbers):
                anomalies.append({
                    "anomaly_type": "inconsistent_question_numbering",
                    "severity": "medium",
                    "details": f"Non-sequential question order detected: {numbers}."
                })

        return anomalies
