import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class LearningAnalyticsModule:
    """
    Student Learning Analytics Module.
    Produces comprehensive quantitative scores and qualitative summaries of student mastery.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def generate_analytics(
        self,
        grammar_data: Dict[str, Any],
        vocab_data: Dict[str, Any],
        spell_data: Dict[str, Any],
        concept_data: Dict[str, Any],
        mistakes: List[Dict[str, Any]],
        subject: str
    ) -> Dict[str, Any]:
        """
        Synthesize all module scores into unified student performance analytics.
        """
        grammar_score = grammar_data.get("grammar_score", 90.0)
        vocab_score = vocab_data.get("vocabulary_score", 85.0)
        spelling_score = spell_data.get("spelling_accuracy_score", 95.0)
        concept_score = concept_data.get("overall_conceptual_score", 85.0)
        
        # Calculate Answer Completeness
        incomplete_count = sum(1 for m in mistakes if m.get("category") == "incomplete_answer")
        completeness_score = max(0.0, 100.0 - (incomplete_count * 25.0))
        
        # Subject Mastery composite score
        subject_mastery = round(0.4 * concept_score + 0.2 * grammar_score + 0.2 * spelling_score + 0.2 * completeness_score, 1)

        # Identify strengths & weaknesses
        strengths = []
        weaknesses = []

        if concept_score >= 85:
            strengths.append(f"Strong conceptual understanding in {subject}.")
        else:
            weaknesses.append(f"Requires conceptual revision in {subject}.")

        if grammar_score >= 85:
            strengths.append("High grammatical accuracy and proper sentence structure.")
        else:
            weaknesses.append("Frequent grammar or punctuation errors.")

        if spelling_score >= 90:
            strengths.append("Accurate student spelling.")
        else:
            weaknesses.append("Spelling mistakes detected in answers.")

        if vocab_score >= 80:
            strengths.append("Good vocabulary diversity.")
        else:
            weaknesses.append("Overly repetitive or simplistic vocabulary.")

        return {
            "grammar_score": grammar_score,
            "vocabulary_score": vocab_score,
            "spelling_accuracy_score": spelling_score,
            "conceptual_understanding_score": concept_score,
            "answer_completeness_score": completeness_score,
            "subject_mastery_score": subject_mastery,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "overall_writing_quality": grammar_data.get("writing_quality", "Good")
        }
