import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ContextualSpellChecker:
    """
    Dedicated Contextual Spell Checking Module.
    Disambiguates OCR noise/artifacts from actual student spelling mistakes.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()
        
        # Standard dictionary of common academic & student spelling mistakes
        self.common_mistakes = {
            "recieve": ("receive", "i before e except after c"),
            "seperate": ("separate", "Contains 'par' in separate"),
            "definately": ("definitely", "Root word is 'finite'"),
            "occurd": ("occurred", "Double 'r' in past tense"),
            "goverment": ("government", "Includes 'n' before 'ment'"),
            "enviroment": ("environment", "Includes 'n' before 'ment'"),
            "becuase": ("because", "Typo in vowel order 'ua'"),
            "answere": ("answer", "Extra 'e' at end"),
            "exsample": ("example", "No 's' after 'x'"),
            "wating": ("waiting", "Missing 'i'"),
            "equasion": ("equation", "Spelled with 't', not 's'"),
            "triangel": ("triangle", "Ending is 'gle'"),
            "photosinthesis": ("photosynthesis", "Spelled with 'y'")
        }

    def check_spelling(self, text: str, ocr_confidence: float = 0.90) -> Dict[str, Any]:
        """
        Check text for spelling errors, classifying whether errors stem from student typos or OCR noise.
        """
        if not text or len(text.strip()) == 0:
            return {"spelling_accuracy_score": 100.0, "spelling_errors": []}

        words = [w.strip(".,!?;:()[]") for w in text.split() if w.strip()]
        spelling_errors = []

        for word in words:
            word_lower = word.lower()
            
            # Check known student spelling mistake dictionary
            if word_lower in self.common_mistakes:
                correction, rule = self.common_mistakes[word_lower]
                spelling_errors.append({
                    "original_word": word,
                    "corrected_word": correction if word.islower() else correction.capitalize(),
                    "confidence": 0.95,
                    "explanation": f"Student spelling error: {rule}",
                    "is_ocr_error": False
                })
            elif len(word) > 4 and self._is_ocr_garbage_pattern(word):
                spelling_errors.append({
                    "original_word": word,
                    "corrected_word": "[Unclear OCR text]",
                    "confidence": 0.70,
                    "explanation": "Flagged as low-confidence OCR recognition artifact rather than student mistake.",
                    "is_ocr_error": True
                })

        total_words = max(1, len(words))
        student_typos = [e for e in spelling_errors if not e["is_ocr_error"]]
        spelling_accuracy = max(0.0, min(100.0, round((1.0 - (len(student_typos) / float(total_words))) * 100.0, 1)))

        return {
            "spelling_accuracy_score": spelling_accuracy,
            "spelling_errors": spelling_errors
        }

    def _is_ocr_garbage_pattern(self, word: str) -> bool:
        # OCR artifact heuristics e.g. "l1ll1", "rn -> m", illegal punctuation inside word
        if re.search(r'\d[a-zA-Z]\d|[a-zA-Z]\d[a-zA-Z]', word):
            return True
        if len(set(word)) == 1 and len(word) > 3:
            return True
        if re.search(r'[^a-zA-Z0-9\-\']', word):
            return True
        return False
