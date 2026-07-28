import os
import torch

class PipelineConfig:
    """Configuration settings for the Notebook Preprocessing & OCR Pipeline."""
    
    # Device configuration
    DEVICE = "cpu"
    NUM_CPU_THREADS = min(8, max(1, os.cpu_count() or 4))
    
    # Model selections for OCR
    PADDLE_LANG = "en"
    TROCR_MODEL_NAME = "microsoft/trocr-small-handwritten"
    FALLBACK_TROCR_MODEL_NAME = "microsoft/trocr-small-handwritten"
    
    # Preprocessing parameters
    MAX_IMAGE_DIMENSION = 2048
    PAPER_MIN_AREA_RATIO = 0.15
    CLAHE_CLIP_LIMIT = 2.5
    CLAHE_TILE_GRID_SIZE = (8, 8)
    BILATERAL_D = 5
    BILATERAL_SIGMA_COLOR = 40
    BILATERAL_SIGMA_SPACE = 40
    DESKEW_MAX_ANGLE = 15.0
    
    # Region classification heuristics
    ASPECT_RATIO_HANDWRITING_THRESHOLD = 0.18
    INTENSITY_VARIANCE_HANDWRITING_MIN = 15.0
    
    # Confidence validation thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    LOW_CONFIDENCE_THRESHOLD = 0.50
    
    # Reading Order Sorter Thresholds
    LINE_VERTICAL_TOLERANCE_PX = 18
    
    def __init__(self):
        self.setup_environment()

    @classmethod
    def setup_environment(cls):
        """Configure PyTorch and environment flags for stable CPU execution."""
        torch.set_num_threads(cls.NUM_CPU_THREADS)
        os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
        os.environ["OMP_NUM_THREADS"] = str(cls.NUM_CPU_THREADS)
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["PADDLE_DISABLE_PIR"] = "1"
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Execute environment setup on module import
PipelineConfig.setup_environment()
