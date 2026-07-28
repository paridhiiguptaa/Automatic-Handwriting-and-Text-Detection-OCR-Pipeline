import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RepeatedMistakeTracker:
    """
    Repeated Mistake Tracking System.
    Analyzes student's historical notebook submissions to track recurring errors over time.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()
        self.history_file = self.config.STUDENT_HISTORY_FILE
        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading student history file: {e}")
        return []

    def save_submission(self, mistakes: List[Dict[str, Any]], page_num: int = 1):
        submission_entry = {
            "page": page_num,
            "mistakes": mistakes
        }
        self.history.append(submission_entry)
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving student history file: {e}")

    def track_repeated_mistakes(self, current_mistakes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify mistakes that occur repeatedly across historical submissions or multiple questions.
        """
        # Count current category occurrences
        cat_counts = {}
        for m in current_mistakes:
            cat = m.get("category", "general")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        # Compare with historical submission mistakes
        for past_sub in self.history:
            for pm in past_sub.get("mistakes", []):
                cat = pm.get("category", "general")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        recurring_categories = [cat for cat, count in cat_counts.items() if count >= 2]
        
        return {
            "recurring_mistake_categories": recurring_categories,
            "total_historical_submissions": len(self.history) + 1,
            "category_frequency_counts": cat_counts
        }
