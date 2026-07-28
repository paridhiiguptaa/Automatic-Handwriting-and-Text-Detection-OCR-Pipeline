import numpy as np
from worksheet_ocr import DocumentLayoutAnalyzer, DocumentReconstructor

def test_layout_role_classification():
    analyzer = DocumentLayoutAnalyzer()
    
    regions = [
        {"id": 1, "bbox": [50, 40, 600, 90], "extracted_text": "SCIENCE NOTEBOOK - CHAPTER 4"},
        {"id": 2, "bbox": [50, 120, 200, 150], "extracted_text": "Q1: What is Photosynthesis?"},
        {"id": 3, "bbox": [100, 160, 450, 190], "extracted_text": "Ans: Photosynthesis is light energy conversion."},
        {"id": 4, "bbox": [120, 200, 300, 220], "extracted_text": "• Requires sunlight & chlorophyll"},
        {"id": 5, "bbox": [50, 240, 250, 260], "extracted_text": "E = mc^2 + 10 = 20"}
    ]

    analyzed = analyzer.analyze_layout(regions, image_width=800)
    
    assert len(analyzed) == 5
    assert analyzed[0]["layout_role"] in ("heading_1", "heading_2")
    assert analyzed[1]["layout_role"] == "question_label"
    assert analyzed[2]["layout_role"] == "answer_label"
    assert analyzed[3]["layout_role"] == "bullet_item"
    assert analyzed[4]["layout_role"] == "math_expression"

def test_document_reconstructor():
    reconstructor = DocumentReconstructor()
    
    analyzed_regions = [
        {"id": 1, "reading_order": 1, "layout_role": "heading_1", "extracted_text": "CHAPTER 1", "indentation_level": 0},
        {"id": 2, "reading_order": 2, "layout_role": "question_label", "extracted_text": "Q1. Define Motion.", "indentation_level": 0},
        {"id": 3, "reading_order": 3, "layout_role": "answer_label", "extracted_text": "Ans: Change in position over time.", "indentation_level": 1}
    ]

    md = reconstructor.reconstruct_markdown(analyzed_regions)
    assert "# 📌 CHAPTER 1" in md
    assert "### ❓ Q1. Define Motion." in md
    assert "> ✏️ **Ans: Change in position over time.**" in md

    tree = reconstructor.build_layout_tree(analyzed_regions)
    assert tree["total_elements"] == 3
    assert "elements" in tree

if __name__ == "__main__":
    test_layout_role_classification()
    test_document_reconstructor()
    print("All Milestone 2 layout tests passed!")
