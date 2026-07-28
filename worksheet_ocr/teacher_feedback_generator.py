import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TeacherFeedbackGenerator:
    """
    Teacher Feedback Generator.
    Produces natural, constructive, prioritized teacher comments and educational recommendations.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def generate_teacher_feedback(
        self,
        analytics: Dict[str, Any],
        mistakes: List[Dict[str, Any]],
        subject: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive teacher evaluation notes and prioritized action items.
        """
        mastery = analytics.get("subject_mastery_score", 90.0)
        
        # 1. Overall Teacher Assessment Comment
        if mastery >= 90:
            summary_comment = f"Outstanding work on this {subject} notebook! Answers are well-structured, clear, and show great attention to detail. Keep up the high standard!"
        elif mastery >= 75:
            summary_comment = f"Good effort overall on your {subject} assignment. The core concepts are clear, but pay attention to minor details like units, spelling, and complete sentence structure."
        else:
            summary_comment = f"This {subject} notebook needs careful revision. Focus on answering every question completely, showing intermediate steps, and reviewing fundamental definitions."

        # 2. Actionable suggestions prioritized by impact
        prioritized_suggestions = []

        # High Priority: Conceptual & Missing Answers
        for m in mistakes:
            cat = m.get("category", "")
            if cat == "incomplete_answer":
                prioritized_suggestions.append({
                    "priority": "High",
                    "suggestion": "Complete all unanswered questions and elaborate on brief responses.",
                    "impact": "Crucial for full points on exams."
                })
            elif cat == "conceptual_misunderstanding":
                prioritized_suggestions.append({
                    "priority": "High",
                    "suggestion": f"Re-read textbook definitions for key {subject} topics.",
                    "impact": "Builds fundamental subject comprehension."
                })
            elif cat == "missing_unit":
                prioritized_suggestions.append({
                    "priority": "High",
                    "suggestion": "Always label calculation answers with proper physical units (e.g. cm, kg, m/s).",
                    "impact": "Prevents deduction of marks on math and science tests."
                })

        # Medium Priority: Grammar & Spelling
        if analytics.get("grammar_score", 100) < 85:
            prioritized_suggestions.append({
                "priority": "Medium",
                "suggestion": "Proofread your sentences for proper capitalization, periods, and subject-verb agreement.",
                "impact": "Improves overall readability and academic writing."
            })
            
        if analytics.get("spelling_accuracy_score", 100) < 90:
            prioritized_suggestions.append({
                "priority": "Medium",
                "suggestion": "Practice spelling key vocabulary terms before writing final submissions.",
                "impact": "Eliminates preventable spelling errors."
            })

        # Low Priority: Vocabulary & Formatting
        if analytics.get("vocabulary_score", 100) < 80:
            prioritized_suggestions.append({
                "priority": "Low",
                "suggestion": "Incorporate stronger academic verbs and adjectives instead of generic words.",
                "impact": "Elevates tone and writing sophistication."
            })

        # Deduplicate suggestions by text
        seen = set()
        dedup_suggestions = []
        for sug in prioritized_suggestions:
            if sug["suggestion"] not in seen:
                seen.add(sug["suggestion"])
                dedup_suggestions.append(sug)

        if not dedup_suggestions:
            dedup_suggestions.append({
                "priority": "Low",
                "suggestion": "Maintain your clean handwriting and neat organization.",
                "impact": "Sustains excellent notebook quality."
            })

        return {
            "teacher_summary_comment": summary_comment,
            "prioritized_suggestions": dedup_suggestions[:5]
        }
