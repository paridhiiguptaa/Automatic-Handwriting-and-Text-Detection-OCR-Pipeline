import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class VocabularyAnalyzer:
    """
    Vocabulary Analysis Engine.
    Evaluates vocabulary sophistication, repetitive word usage, homophone confusion, and academic alternatives.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def analyze_vocabulary(self, text: str) -> Dict[str, Any]:
        """
        Evaluate vocabulary usage and generate academic vocabulary suggestions.
        """
        if not text or len(text.strip()) == 0:
            return {"vocabulary_score": 100.0, "suggestions": [], "repetitive_words": []}

        words = [w.lower().strip(".,!?;:()[]") for w in text.split() if w.strip()]
        unique_words = set(words)
        
        suggestions = []

        # 1. Homophone confusion dictionary
        homophones = {
            "there": (("their", "they're"), "Verify spatial vs. possessive pronoun ('their') vs. contraction ('they are')."),
            "their": (("there", "they're"), "Verify possessive pronoun vs. location ('there')."),
            "your": (("you're",), "Verify possessive ('your') vs. contraction ('you are')."),
            "to": (("too", "two"), "Verify preposition ('to') vs. adverb/excessive ('too') vs. number ('two')."),
            "its": (("it's",), "Verify possessive ('its') vs. contraction ('it is')."),
            "effect": (("affect",), "Verify noun result ('effect') vs. verb influence ('affect').")
        }

        for word in words:
            if word in homophones:
                alts, exp = homophones[word]
                suggestions.append({
                    "type": "homophone_check",
                    "original_word": word,
                    "suggested_alternatives": list(alts),
                    "explanation": exp,
                    "confidence": 0.85
                })

        # 2. Simplistic vocabulary replacement suggestions
        weak_vocab_map = {
            "big": ("substantial / significant", "Use stronger academic descriptors for scale or impact."),
            "small": ("minor / minimal", "Use precise terminology for magnitude."),
            "good": ("effective / beneficial / exemplary", "Replace vague praise with specific qualitative adjectives."),
            "bad": ("detrimental / adverse / flawed", "Use academic terminology for negative outcomes."),
            "thing": ("concept / factor / element", "Replace generic nouns with precise terminology."),
            "stuff": ("materials / components", "Replace informal speech with formal vocabulary."),
            "get": ("obtain / acquire / derive", "Use precise active verbs."),
            "make": ("synthesize / produce / construct", "Use subject-appropriate action verbs.")
        }

        for word in words:
            if word in weak_vocab_map:
                better, exp = weak_vocab_map[word]
                suggestions.append({
                    "type": "weak_vocabulary",
                    "original_word": word,
                    "suggested_alternatives": [better],
                    "explanation": exp,
                    "confidence": 0.90
                })

        # 3. Repetitive word usage detection
        word_counts = {}
        for w in words:
            if len(w) > 3 and w not in ("this", "that", "with", "from", "have", "they", "were", "what"):
                word_counts[w] = word_counts.get(w, 0) + 1

        repetitive = [w for w, count in word_counts.items() if count >= 3]

        # Calculate Vocabulary Score
        lexical_diversity = (len(unique_words) / float(len(words))) if words else 1.0
        vocab_score = max(0.0, min(100.0, round(lexical_diversity * 100 - (len(suggestions) * 5), 1)))

        return {
            "vocabulary_score": vocab_score,
            "lexical_diversity_ratio": round(lexical_diversity, 2),
            "suggestions": suggestions[:8],
            "repetitive_words": repetitive
        }
