import numpy as np
import cv2
from worksheet_ocr import TextDetector, OCRRecognizer, ReadingOrderSorter

def test_reading_order_sorter():
    sorter = ReadingOrderSorter()
    regions = [
        {"id": 1, "bbox": [50, 200, 200, 230], "extracted_text": "Second line text"},
        {"id": 2, "bbox": [50, 50, 300, 80], "extracted_text": "First line title"},
        {"id": 3, "bbox": [50, 350, 180, 380], "extracted_text": "Third line answer"}
    ]

    sorted_regs = sorter.sort_regions(regions)
    assert len(sorted_regs) == 3
    assert sorted_regs[0]["extracted_text"] == "First line title"
    assert sorted_regs[1]["extracted_text"] == "Second line text"
    assert sorted_regs[2]["extracted_text"] == "Third line answer"
    assert sorted_regs[0]["reading_order"] == 1
    assert sorted_regs[1]["reading_order"] == 2
    assert sorted_regs[2]["reading_order"] == 3

    text_output = sorter.reconstruct_text(sorted_regs)
    assert "First line title" in text_output
    assert "Second line text" in text_output

def test_text_detector_opencv_fallback():
    detector = TextDetector()
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, "PRACTICE WORKSHEET PAGE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Q1: What is the capital of France?", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Answer: Paris", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    regions = detector._opencv_layout_detect(img)
    assert isinstance(regions, list)
    assert len(regions) >= 1
    for r in regions:
        assert "bbox" in r
        assert "region_type" in r

if __name__ == "__main__":
    test_reading_order_sorter()
    test_text_detector_opencv_fallback()
    print("All OCR & reading order tests passed!")
