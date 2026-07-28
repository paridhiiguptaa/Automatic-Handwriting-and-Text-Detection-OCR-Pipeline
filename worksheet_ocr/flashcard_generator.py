import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FlashcardGenerator:
    """
    Personalized Flashcard Generator.
    Automatically generates targeted revision flashcards from student mistakes and recurring error patterns.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def generate_flashcards(self, mistakes: List[Dict[str, Any]], subject: str) -> List[Dict[str, Any]]:
        """
        Generate study flashcards corresponding to detected student mistakes.
        """
        flashcards = []

        for idx, m in enumerate(mistakes):
            cat = m.get("category", "")
            orig = m.get("original_text", "")
            fix = m.get("suggested_correction", "")
            exp = m.get("explanation", "")

            if cat == "spelling":
                flashcards.append({
                    "flashcard_id": len(flashcards) + 1,
                    "title": f"Spelling Revision: {orig}",
                    "question": f"How do you correctly spell '{orig}'?",
                    "answer": fix,
                    "explanation": f"Rule/Tip: {exp}",
                    "difficulty_level": "Easy",
                    "subject_tags": [subject, "Spelling"],
                    "reason_generated": f"Student misspelled '{orig}' in their answer."
                })
            elif cat == "homophone_check" or "their" in orig.lower() or "there" in orig.lower():
                flashcards.append({
                    "flashcard_id": len(flashcards) + 1,
                    "title": "Homophones: Their vs There vs They're",
                    "question": "What is the difference between 'their', 'there', and 'they're'?",
                    "answer": "'Their' = possessive (belonging to them). 'There' = place/location. 'They're' = contraction of 'they are'.",
                    "explanation": "Example: They're going over there to get their notebooks.",
                    "difficulty_level": "Medium",
                    "subject_tags": ["English", "Vocabulary"],
                    "reason_generated": "Detected homophone confusion in student response."
                })
            elif cat in ("conceptual_misunderstanding", "calculation_error"):
                flashcards.append({
                    "flashcard_id": len(flashcards) + 1,
                    "title": f"Concept Review: {subject} Question",
                    "question": f"Review Problem: {orig[:60]}...",
                    "answer": f"Correct Solution / Step: {fix}",
                    "explanation": exp,
                    "difficulty_level": "Hard",
                    "subject_tags": [subject, "Concept Review"],
                    "reason_generated": "Conceptual or calculation error made during notebook submission."
                })
            elif cat == "missing_unit":
                flashcards.append({
                    "flashcard_id": len(flashcards) + 1,
                    "title": "Measurement Units in Calculation",
                    "question": "Why must you always include units (e.g., cm², kg, m/s) in physics and math answers?",
                    "answer": "A number without units is incomplete because it does not convey the physical magnitude or scale.",
                    "explanation": "Always attach the required unit to final answers.",
                    "difficulty_level": "Medium",
                    "subject_tags": [subject, "Units"],
                    "reason_generated": "Omitted units in mathematical calculation."
                })
            elif cat == "weak_vocabulary":
                flashcards.append({
                    "flashcard_id": len(flashcards) + 1,
                    "title": f"Academic Vocabulary Enhancement: {orig}",
                    "question": f"What is a stronger academic alternative to '{orig}'?",
                    "answer": fix,
                    "explanation": exp,
                    "difficulty_level": "Easy",
                    "subject_tags": ["English", "Vocabulary"],
                    "reason_generated": "Overly simplistic word choice."
                })

            if len(flashcards) >= self.config.MAX_FLASHCARDS_PER_NOTEBOOK:
                break

        # Fallback default flashcard if no severe mistakes found
        if not flashcards:
            flashcards.append({
                "flashcard_id": 1,
                "title": f"Mastery Review: {subject}",
                "question": f"What is the key summary concept of this {subject} worksheet?",
                "answer": "Review all key definitions and review formulas regularly.",
                "explanation": "Continuous revision maintains high test performance.",
                "difficulty_level": "Easy",
                "subject_tags": [subject],
                "reason_generated": "General performance reinforcement."
            })

        return flashcards
