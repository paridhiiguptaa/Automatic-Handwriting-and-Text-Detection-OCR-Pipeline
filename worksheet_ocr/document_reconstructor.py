import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DocumentReconstructor:
    """
    Stage 6: Document Structure Reconstructor.
    Transforms layout-analyzed OCR regions into formatted notebook Markdown
    and a hierarchical JSON document tree.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def reconstruct_markdown(self, analyzed_regions: List[Dict[str, Any]]) -> str:
        """
        Renders layout regions into structured Markdown matching original notebook flow.
        """
        if not analyzed_regions:
            return "*(No text regions detected)*"

        md_lines = []
        for reg in analyzed_regions:
            text = reg.get("extracted_text", "").strip()
            if not text:
                continue

            role = reg.get("layout_role", "paragraph")
            indent = reg.get("indentation_level", 0)
            indent_prefix = "  " * indent

            if role == "heading_1":
                md_lines.append(f"\n# 📌 {text}\n")
            elif role == "heading_2":
                md_lines.append(f"\n## 🔹 {text}\n")
            elif role == "question_label":
                md_lines.append(f"\n### ❓ {text}")
            elif role == "answer_label":
                md_lines.append(f"\n{indent_prefix}> ✏️ **{text}**")
            elif role == "bullet_item":
                clean_bullet = text.lstrip('•-*➢▪→ ').strip()
                md_lines.append(f"{indent_prefix}- {clean_bullet}")
            elif role == "math_expression":
                md_lines.append(f"\n{indent_prefix}$${text}$$\n")
            else: # Paragraph
                if indent > 0:
                    md_lines.append(f"{indent_prefix}{text}")
                else:
                    md_lines.append(text)

        return "\n".join(md_lines).strip()

    def build_layout_tree(self, analyzed_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a structured JSON hierarchy tree of the notebook page.
        """
        tree = {
            "title": "Notebook Page Document Structure",
            "total_elements": len(analyzed_regions),
            "elements": []
        }

        current_heading = None
        current_question = None

        for reg in analyzed_regions:
            text = reg.get("extracted_text", "").strip()
            if not text:
                continue

            role = reg.get("layout_role", "paragraph")
            node = {
                "id": reg.get("id"),
                "reading_order": reg.get("reading_order"),
                "role": role,
                "text": text,
                "confidence": reg.get("confidence"),
                "bbox": reg.get("bbox"),
                "indentation_level": reg.get("indentation_level", 0)
            }

            if role in ("heading_1", "heading_2"):
                current_heading = node
                current_heading["children"] = []
                tree["elements"].append(current_heading)
                current_question = None
            elif role == "question_label":
                current_question = node
                current_question["answers"] = []
                if current_heading is not None:
                    current_heading["children"].append(current_question)
                else:
                    tree["elements"].append(current_question)
            elif role == "answer_label":
                if current_question is not None:
                    current_question["answers"].append(node)
                elif current_heading is not None:
                    current_heading["children"].append(node)
                else:
                    tree["elements"].append(node)
            else:
                if current_question is not None:
                    current_question["answers"].append(node)
                elif current_heading is not None:
                    current_heading["children"].append(node)
                else:
                    tree["elements"].append(node)

        return tree
