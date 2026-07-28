import numpy as np
import cv2
from worksheet_ocr import ImagePreprocessor, PipelineConfig

def test_preprocessor_initialization():
    config = PipelineConfig()
    preprocessor = ImagePreprocessor(config=config)
    assert preprocessor is not None

def test_preprocessor_process_synthetic_image():
    # Create a synthetic white page image with black text and dark border
    canvas = np.zeros((600, 400, 3), dtype=np.uint8)
    # Draw white paper sheet in middle
    cv2.rectangle(canvas, (40, 40), (360, 560), (240, 240, 240), -1)
    # Draw sample text lines
    cv2.putText(canvas, "Sample Notebook Line 1", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(canvas, "Sample Notebook Line 2", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

    preprocessor = ImagePreprocessor()
    processed_bgr, meta = preprocessor.process(canvas)

    assert processed_bgr is not None
    assert isinstance(processed_bgr, np.ndarray)
    assert processed_bgr.ndim == 3
    assert "original_dimensions" in meta
    assert "processed_dimensions" in meta
    assert meta["shadow_reduction_applied"] is True

def test_shadow_reduction_and_contrast():
    img = np.full((200, 200, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (180, 180), (100, 100, 100), -1)
    
    preprocessor = ImagePreprocessor()
    shadow_free = preprocessor.remove_shadows_and_normalize(img)
    enhanced = preprocessor.enhance_contrast_lab(shadow_free)
    
    assert shadow_free.shape == img.shape
    assert enhanced.shape == img.shape

if __name__ == "__main__":
    test_preprocessor_initialization()
    test_preprocessor_process_synthetic_image()
    test_shadow_reduction_and_contrast()
    print("All preprocessing tests passed!")
