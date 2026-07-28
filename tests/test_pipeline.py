import os
import cv2
import numpy as np
from worksheet_ocr import NotebookOCRPipeline

_shared_pipeline = None

def get_shared_pipeline():
    global _shared_pipeline
    if _shared_pipeline is None:
        _shared_pipeline = NotebookOCRPipeline()
    return _shared_pipeline

def test_pipeline_integration_on_synthetic_image():
    pipeline = get_shared_pipeline()
    
    img = np.full((800, 1000, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (980, 780), (0, 0, 0), 2)
    cv2.putText(img, "SCIENCE NOTEBOOK - CHAPTER 4", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(img, "Q1: What is photosynthesis?", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Ans: Photosynthesis is light energy conversion.", (80, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    result = pipeline.process_image(img)
    
    assert "summary" in result
    assert "preprocessing_metadata" in result
    assert "reconstructed_text" in result
    assert "formatted_markdown" in result
    assert "document_tree" in result
    assert "regions" in result
    assert "cleaned_bgr" in result
    assert "annotated_bgr" in result
    assert isinstance(result["reconstructed_text"], str)

def test_pipeline_on_sample_workspace_image():
    sample_file = "WhatsApp Image 2026-07-23 at 5.11.35 AM.jpeg"
    if not os.path.exists(sample_file):
        print(f"Skipping {sample_file} (file not present).")
        return

    pipeline = get_shared_pipeline()
    result = pipeline.process_image(sample_file)

    assert result["summary"]["total_text_regions"] > 0
    assert result["annotated_bgr"] is not None
    assert len(result["regions"]) > 0

if __name__ == "__main__":
    print("--- Running Pipeline Integration Tests ---")
    test_pipeline_integration_on_synthetic_image()
    test_pipeline_on_sample_workspace_image()
    print("All pipeline tests passed successfully!")
