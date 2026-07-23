import os
import torch

class PipelineConfig:
    """Configuration settings for the Worksheet OCR Pipeline."""
    
    # Device configuration
    DEVICE = "cpu"
    NUM_CPU_THREADS = min(8, max(1, os.cpu_count() or 4))
    
    # Model selections
    PADDLE_LANG = "en"
    TROCR_MODEL_NAME = "microsoft/trocr-small-handwritten"
    VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM2-256M-Instruct"
    
    # Confidence validation thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.82
    LOW_CONFIDENCE_THRESHOLD = 0.60
    
    # Image preprocessing parameters
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    BILATERAL_D = 5
    BILATERAL_SIGMA_COLOR = 50
    BILATERAL_SIGMA_SPACE = 50
    DESKEW_ANGLE_LIMIT = 15.0
    
    # Text region routing heuristics
    ASPECT_RATIO_HANDWRITING_THRESHOLD = 0.2
    EDGE_DENSITY_HANDWRITING_MAX = 0.35
    
    # Batch sizes for CPU efficiency
    TROCR_BATCH_SIZE = 4
    PADDLE_REC_BATCH_SIZE = 8
    
    @classmethod
    def setup_environment(cls):
        """Configure PyTorch and Paddle execution flags for stable CPU inference."""
        torch.set_num_threads(cls.NUM_CPU_THREADS)
        os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
        os.environ["OMP_NUM_THREADS"] = str(cls.NUM_CPU_THREADS)
        # Disable oneDNN PIR instruction bug on CPU
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["PADDLE_DISABLE_PIR"] = "1"
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
