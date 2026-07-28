import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SubjectClassifier:
    """
    Subject Identification Module.
    Classifies notebook topic into: Mathematics, Science, English, History/Geography, or General.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def identify_subject(self, document_text: str, layout_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify document text and layout features into target subject domain.
        """
        text_lower = document_text.lower()
        
        # Keyword & pattern scoring dictionaries
        scores = {
            "Mathematics": 0,
            "Science": 0,
            "English": 0,
            "History/Geography": 0,
            "General": 1
        }
        
        # 1. Mathematics indicators
        math_keywords = [
            "calculate", "solve", "find", "equation", "triangle", "angle", "area", "perimeter",
            "fraction", "decimal", "algebra", "x =", "y =", "sin", "cos", "tan", "plus", "minus",
            "multiply", "divide", "hypotenuse", "volume", "radius", "diameter"
        ]
        for kw in math_keywords:
            if kw in text_lower:
                scores["Mathematics"] += 2
        if re.search(r'[\+\-\*\/×÷=\^√]', document_text):
            scores["Mathematics"] += 3
        if any(e.get("role") == "mathematical_equation" for e in layout_elements):
            scores["Mathematics"] += 4

        # 2. Science indicators
        science_keywords = [
            "photosynthesis", "cell", "energy", "force", "gravity", "mass", "velocity", "chemical",
            "element", "molecule", "atom", "reaction", "acid", "base", "organism", "ecosystem",
            "respiration", "temperature", "celsius", "fahrenheit", "kg", "m/s", "joules", "watts", "newton"
        ]
        for kw in science_keywords:
            if kw in text_lower:
                scores["Science"] += 2
        if any(e.get("role") == "chemical_equation" for e in layout_elements):
            scores["Science"] += 4

        # 3. English indicators
        english_keywords = [
            "grammar", "noun", "verb", "adjective", "adverb", "synonym", "antonym", "sentence",
            "paragraph", "essay", "metaphor", "simile", "comprehension", "passage", "tense", "plural"
        ]
        for kw in english_keywords:
            if kw in text_lower:
                scores["English"] += 2

        # 4. History / Geography indicators
        hist_keywords = [
            "century", "war", "king", "revolution", "empire", "emperor", "dynasty", "treaty",
            "continent", "ocean", "latitude", "longitude", "river", "capital", "population",
            "b.c.", "a.d.", "18th", "19th", "20th", "constitution", "president"
        ]
        for kw in hist_keywords:
            if kw in text_lower:
                scores["History/Geography"] += 2

        # Find max subject score
        best_subject = max(scores, key=scores.get)
        max_score = scores[best_subject]
        confidence = min(0.98, max(0.60, max_score / 10.0))
        
        return {
            "identified_subject": best_subject if max_score > 1 else self.config.DEFAULT_SUBJECT,
            "confidence": round(confidence, 2),
            "subject_scores": scores
        }
