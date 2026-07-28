import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DocumentUnderstandingLayer:
    """
    Document Understanding Layer.
    Maps layout elements into structured Question-Answer pairs, diagram links, and logical reading flow.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def process_document_structure(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Group layout elements into Question-Answer blocks and establish logical relationships.
        """
        elements = layout_data.get("layout_elements", [])
        
        qa_pairs = []
        current_qa = None
        unassociated_diagrams = []
        
        for elem in elements:
            role = elem.get("role", "paragraph")
            text = elem.get("text", "")
            elem_id = elem.get("element_id")
            
            if role in ("question_number", "question"):
                if current_qa and (current_qa["question_text"] or current_qa["answer_text"]):
                    qa_pairs.append(current_qa)
                    
                current_qa = {
                    "question_id": len(qa_pairs) + 1,
                    "question_number_elem_id": elem_id if role == "question_number" else None,
                    "question_text": text if role == "question" else "",
                    "answer_text": "",
                    "answer_elem_ids": [],
                    "sub_questions": [],
                    "associated_diagrams": [],
                    "explanations": []
                }
                if role == "question_number":
                    current_qa["question_label"] = text
            elif role == "sub_question":
                if current_qa:
                    current_qa["sub_questions"].append({
                        "elem_id": elem_id,
                        "text": text
                    })
            elif role in ("answer", "answer_section"):
                if current_qa:
                    if current_qa["answer_text"]:
                        current_qa["answer_text"] += " " + text
                    else:
                        current_qa["answer_text"] = text
                    current_qa["answer_elem_ids"].append(elem_id)
                else:
                    # Floating answer before any question number
                    current_qa = {
                        "question_id": len(qa_pairs) + 1,
                        "question_text": "General Response",
                        "answer_text": text,
                        "answer_elem_ids": [elem_id],
                        "sub_questions": [],
                        "associated_diagrams": [],
                        "explanations": []
                    }
            elif role == "diagram":
                if current_qa:
                    current_qa["associated_diagrams"].append(elem_id)
                else:
                    unassociated_diagrams.append(elem_id)
            elif role == "paragraph":
                if current_qa and current_qa["answer_text"]:
                    current_qa["explanations"].append(text)
                elif current_qa and not current_qa["question_text"]:
                    current_qa["question_text"] = text
                    
        if current_qa and (current_qa["question_text"] or current_qa["answer_text"]):
            qa_pairs.append(current_qa)

        # Logical reading order flow (e.g. sequence of element IDs)
        reading_flow = [e.get("element_id") for e in elements]

        return {
            "qa_pairs": qa_pairs,
            "reading_order_flow": reading_flow,
            "unassociated_diagram_ids": unassociated_diagrams,
            "total_questions_detected": len(qa_pairs)
        }
