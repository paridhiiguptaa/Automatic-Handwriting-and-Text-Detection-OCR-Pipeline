import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GrammarAnalyzer:
    """
    Grammar & Language Analysis Module.
    Analyzes student sentences for grammar, punctuation, capitalization, tense consistency, and sentence structure.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for grammatical correctness and sentence structure quality.
        """
        if not text or len(text.strip()) == 0:
            return {"grammar_score": 100.0, "detected_issues": [], "writing_quality": "N/A"}

        issues = []
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        for sent in sentences:
            # 1. Capitalization check
            if len(sent) > 0 and sent[0].islower():
                issues.append({
                    "issue_type": "capitalization",
                    "original_text": sent,
                    "corrected_text": sent[0].upper() + sent[1:],
                    "explanation": "Sentence must begin with a capital letter.",
                    "confidence": 0.95
                })

            # 2. Missing ending punctuation check
            if len(sent) > 0 and sent[-1] not in ".!?":
                issues.append({
                    "issue_type": "punctuation",
                    "original_text": sent,
                    "corrected_text": sent + ".",
                    "explanation": "Sentence is missing a period or ending punctuation.",
                    "confidence": 0.92
                })

            # 3. Subject-Verb Agreement checks (e.g. "they is", "he do", "we was", "it contain")
            sv_patterns = [
                (r'\b(they|we|you)\s+(is|was)\b', r'\1 are/were', "Subject-verb agreement mismatch. Use 'are' or 'were' with plural subjects."),
                (r'\b(he|she|it)\s+(do|are|were)\b', r'\1 does/is/was', "Subject-verb agreement mismatch. Use singular verb forms with singular pronouns."),
                (r'\b(it|he|she)\s+(contain|show|mean|cause)\b', r'\1 contains/shows/means/causes', "Singular third-person subject requires an 's' on present tense verbs.")
            ]
            for pat, repl, desc in sv_patterns:
                m = re.search(pat, sent, re.IGNORECASE)
                if m:
                    corrected = re.sub(pat, repl, sent, flags=re.IGNORECASE)
                    issues.append({
                        "issue_type": "subject_verb_agreement",
                        "original_text": m.group(0),
                        "corrected_text": corrected,
                        "explanation": desc,
                        "confidence": 0.88
                    })

            # 4. Repeated word check (e.g. "the the", "is is")
            dup_m = re.search(r'\b([a-zA-Z]+)\s+\1\b', sent, re.IGNORECASE)
            if dup_m:
                issues.append({
                    "issue_type": "repeated_words",
                    "original_text": dup_m.group(0),
                    "corrected_text": dup_m.group(1),
                    "explanation": f"Duplicate word '{dup_m.group(1)}' found in sentence.",
                    "confidence": 0.96
                })

            # 5. Incomplete sentence fragment check
            words = sent.split()
            if len(words) < 3 and not any(w.lower() in ("yes", "no", "true", "false") for w in words):
                issues.append({
                    "issue_type": "incomplete_sentence",
                    "original_text": sent,
                    "corrected_text": sent + " (Expand into complete thought)",
                    "explanation": "Sentence fragment detected. Provide a complete sentence with subject and verb.",
                    "confidence": 0.82
                })

        # Calculate Grammar Score
        base_score = 100.0 - (len(issues) * 12.5)
        grammar_score = max(0.0, min(100.0, round(base_score, 1)))

        if grammar_score > 85:
            quality = "Excellent writing construction"
        elif grammar_score > 70:
            quality = "Good, minor grammar/punctuation corrections recommended"
        else:
            quality = "Needs grammatical refinement and sentence structure practice"

        return {
            "grammar_score": grammar_score,
            "detected_issues": issues,
            "writing_quality": quality
        }
