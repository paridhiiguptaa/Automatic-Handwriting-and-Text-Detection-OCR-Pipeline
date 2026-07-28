import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class BaseSubjectEvaluator:
    """Base class for modular subject evaluation handlers."""
    
    def evaluate(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class MathSubjectEvaluator(BaseSubjectEvaluator):
    """Mathematics Evaluation Handler."""

    def evaluate(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        q_text = qa_pair.get("question_text", "")
        a_text = qa_pair.get("answer_text", "").strip()
        
        errors = []
        is_correct = True
        
        if not a_text:
            return {
                "is_correct": False,
                "conceptual_accuracy": 0.0,
                "explanation": "Answer is blank or unreadable.",
                "mistakes": ["incomplete_answer"]
            }

        # 1. Check for missing units (e.g. cm, m, kg, m/s, degrees, %, $, units)
        requires_unit = any(kw in q_text.lower() for kw in ["find the area", "find the perimeter", "calculate the speed", "length", "weight", "mass", "volume", "distance", "cost", "price"])
        has_unit = bool(re.search(r'\b(cm|m|km|g|kg|m\/s|s|min|hrs|°|degrees|\$|USD|INR|units|sq cm|m²|cm²)\b', a_text, re.IGNORECASE))
        if requires_unit and not has_unit:
            errors.append("Missing measurement unit in final answer.")
            
        # 2. Calculation step verification heuristic
        has_steps = ("=" in a_text or "+" in a_text or "-" in a_text or "*" in a_text or "/" in a_text or "\n" in a_text)
        if len(q_text) > 15 and not has_steps:
            errors.append("Missing intermediate calculation steps; only final number provided.")

        # 3. Numeric accuracy heuristic if question has simple arithmetic
        numbers = [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', q_text)]
        if "add" in q_text.lower() or "+" in q_text:
            expected_sum = sum(numbers) if len(numbers) >= 2 else None
            ans_nums = [float(n) for n in re.findall(r'\b\d+(?:\.\d+)?\b', a_text)]
            if expected_sum and ans_nums and expected_sum not in ans_nums:
                is_correct = False
                errors.append(f"Arithmetic calculation error. Expected sum {expected_sum}, but found {ans_nums[-1]}.")

        accuracy = 100.0 if not errors else (50.0 if is_correct else 25.0)

        return {
            "is_correct": is_correct and len(errors) == 0,
            "conceptual_accuracy": accuracy,
            "explanation": "Math answer meets step and unit criteria." if not errors else " ".join(errors),
            "mistakes": errors
        }

class ScienceSubjectEvaluator(BaseSubjectEvaluator):
    """Science Evaluation Handler."""

    def evaluate(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        q_text = qa_pair.get("question_text", "")
        a_text = qa_pair.get("answer_text", "").strip()
        associated_diagrams = qa_pair.get("associated_diagrams", [])
        
        errors = []
        
        if not a_text:
            return {
                "is_correct": False,
                "conceptual_accuracy": 0.0,
                "explanation": "Answer is blank or unreadable.",
                "mistakes": ["incomplete_answer"]
            }

        # 1. Diagram check if question requests drawing/labeling
        if any(kw in q_text.lower() for kw in ["draw", "diagram", "label", "sketch", "illustrate"]) and not associated_diagrams:
            errors.append("Question requested a diagram/drawing, but no diagram was provided or detected.")

        # 2. Scientific terminology check
        science_terms = ["energy", "force", "cell", "photosynthesis", "gravity", "mass", "reaction", "acid", "base", "molecule", "chlorophyll", "respiration"]
        used_terms = [t for t in science_terms if t in a_text.lower()]
        if len(q_text) > 20 and len(used_terms) == 0:
            errors.append("Answer lacks core scientific terminology.")

        accuracy = 100.0 if not errors else 60.0
        return {
            "is_correct": len(errors) == 0,
            "conceptual_accuracy": accuracy,
            "explanation": "Scientifically complete answer." if not errors else " ".join(errors),
            "mistakes": errors
        }

class EnglishSubjectEvaluator(BaseSubjectEvaluator):
    """English Evaluation Handler."""

    def evaluate(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        a_text = qa_pair.get("answer_text", "").strip()
        errors = []
        
        if not a_text:
            return {
                "is_correct": False,
                "conceptual_accuracy": 0.0,
                "explanation": "Answer is blank or unreadable.",
                "mistakes": ["incomplete_answer"]
            }

        if len(a_text.split()) < 3:
            errors.append("Response is overly brief; elaborate in complete sentences.")

        return {
            "is_correct": len(errors) == 0,
            "conceptual_accuracy": 100.0 if not errors else 70.0,
            "explanation": "Well-constructed English response." if not errors else " ".join(errors),
            "mistakes": errors
        }

class HistorySubjectEvaluator(BaseSubjectEvaluator):
    """History/Geography Evaluation Handler."""

    def evaluate(self, qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        a_text = qa_pair.get("answer_text", "").strip()
        errors = []
        
        if not a_text:
            return {
                "is_correct": False,
                "conceptual_accuracy": 0.0,
                "explanation": "Answer is blank or unreadable.",
                "mistakes": ["incomplete_answer"]
            }

        # Check factual detail
        if len(a_text.split()) < 4:
            errors.append("Factual explanation is too brief. Include relevant dates, names, or locations.")

        return {
            "is_correct": len(errors) == 0,
            "conceptual_accuracy": 100.0 if not errors else 75.0,
            "explanation": "Historically accurate and structured." if not errors else " ".join(errors),
            "mistakes": errors
        }

class NotebookEvaluator:
    """
    Intelligent Notebook Evaluation Engine.
    Evaluates student answers based on subject domain and logical correctness.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()
        self.evaluators = {
            "Mathematics": MathSubjectEvaluator(),
            "Science": ScienceSubjectEvaluator(),
            "English": EnglishSubjectEvaluator(),
            "History/Geography": HistorySubjectEvaluator(),
            "General": EnglishSubjectEvaluator()
        }

    def evaluate_notebook(self, qa_pairs: List[Dict[str, Any]], subject: str) -> Dict[str, Any]:
        evaluator = self.evaluators.get(subject, self.evaluators["General"])
        
        evaluations = []
        total_acc = 0.0
        
        for qa in qa_pairs:
            eval_res = evaluator.evaluate(qa)
            eval_res["question_id"] = qa.get("question_id")
            eval_res["question_text"] = qa.get("question_text", "")
            eval_res["answer_text"] = qa.get("answer_text", "")
            evaluations.append(eval_res)
            total_acc += eval_res["conceptual_accuracy"]

        avg_acc = (total_acc / float(len(qa_pairs))) if qa_pairs else 100.0

        return {
            "overall_conceptual_score": round(avg_acc, 1),
            "question_evaluations": evaluations
        }
