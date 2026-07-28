import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MistakeClassifier:
    """
    Mistake Classification Engine.
    Categorizes errors into 14 distinct pedagogical categories with severity ratings and suggested corrections.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def classify_all_mistakes(
        self,
        grammar_analysis: Dict[str, Any],
        vocab_analysis: Dict[str, Any],
        spell_analysis: Dict[str, Any],
        concept_evaluation: Dict[str, Any],
        missing_info: List[Dict[str, Any]],
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Aggregate and categorize all detected errors across the pipeline.
        """
        categorized_mistakes = []

        # 1. Grammar mistakes
        for issue in grammar_analysis.get("detected_issues", []):
            cat = issue.get("issue_type", "grammar")
            if cat not in self.config.MISTAKE_CATEGORIES:
                cat = "grammar"
            categorized_mistakes.append({
                "category": cat,
                "severity": "medium",
                "original_text": issue.get("original_text", ""),
                "suggested_correction": issue.get("corrected_text", ""),
                "explanation": issue.get("explanation", ""),
                "confidence": issue.get("confidence", 0.90)
            })

        # 2. Vocabulary mistakes
        for sug in vocab_analysis.get("suggestions", []):
            categorized_mistakes.append({
                "category": "vocabulary",
                "severity": "low",
                "original_text": sug.get("original_word", ""),
                "suggested_correction": ", ".join(sug.get("suggested_alternatives", [])),
                "explanation": sug.get("explanation", ""),
                "confidence": sug.get("confidence", 0.85)
            })

        # 3. Spelling mistakes
        for err in spell_analysis.get("spelling_errors", []):
            if not err.get("is_ocr_error", False):
                categorized_mistakes.append({
                    "category": "spelling",
                    "severity": "medium",
                    "original_text": err.get("original_word", ""),
                    "suggested_correction": err.get("corrected_word", ""),
                    "explanation": err.get("explanation", ""),
                    "confidence": err.get("confidence", 0.95)
                })

        # 4. Conceptual & Subject mistakes
        for q_eval in concept_evaluation.get("question_evaluations", []):
            for m in q_eval.get("mistakes", []):
                if isinstance(m, str):
                    if "unit" in m.lower():
                        cat = "missing_unit"
                    elif "calculation" in m.lower() or "arithmetic" in m.lower():
                        cat = "calculation_error"
                    elif "diagram" in m.lower():
                        cat = "diagram_error"
                    else:
                        cat = "conceptual_misunderstanding"
                    
                    categorized_mistakes.append({
                        "category": cat,
                        "severity": "high",
                        "original_text": q_eval.get("answer_text", ""),
                        "suggested_correction": q_eval.get("explanation", ""),
                        "explanation": m,
                        "confidence": 0.90
                    })

        # 5. Missing Info
        for mi in missing_info:
            categorized_mistakes.append({
                "category": "incomplete_answer",
                "severity": mi.get("severity", "high"),
                "original_text": f"Question {mi.get('question_id')}",
                "suggested_correction": "Provide full answer and details.",
                "explanation": mi.get("explanation", ""),
                "confidence": 0.92
            })

        return categorized_mistakes
