import cv2
import numpy as np
import torch
from PIL import Image
import logging
import warnings
from typing import Dict, Any, Tuple

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class OCRRecognizer:
    """
    Stage 3: Hybrid Printed & Handwritten OCR Recognizer.
    Engine 1: PaddleOCR - High accuracy for printed header titles, instructions, numbers, fonts.
    Engine 2: Hugging Face TrOCR (microsoft/trocr-small-handwritten) - Ultra-lightweight handwritten OCR model.
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

        self._paddle_rec = None
        self._trocr_processor = None
        self._trocr_model = None

    def _get_paddle_recognizer(self):
        if self._paddle_rec is None:
            from paddleocr import PaddleOCR
            try:
                self._paddle_rec = PaddleOCR(
                    lang=self.config.PADDLE_LANG,
                    enable_mkldnn=False
                )
            except Exception as e:
                logger.warning(f"PaddleOCR recognizer init fallback: {e}")
                try:
                    self._paddle_rec = PaddleOCR(enable_mkldnn=False)
                except Exception:
                    self._paddle_rec = PaddleOCR()
        return self._paddle_rec

    def _get_trocr(self):
        if self._trocr_model is None:
            import transformers
            transformers.logging.set_verbosity_error()
            from transformers import RobertaTokenizer, ViTImageProcessor, TrOCRProcessor, VisionEncoderDecoderModel

            model_name = self.config.TROCR_MODEL_NAME
            logger.info(f"Loading Hugging Face TrOCR model: {model_name}...")
            
            try:
                img_proc = ViTImageProcessor.from_pretrained(model_name)
                tokenizer = RobertaTokenizer.from_pretrained(model_name)
                self._trocr_processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tokenizer)
                self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
                self._trocr_model.eval()
                self._trocr_model.to(self.config.DEVICE)
            except Exception as e:
                logger.warning(f"TrOCR direct loading fallback: {e}")
                try:
                    self._trocr_processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
                    self._trocr_model = VisionEncoderDecoderModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32
                    )
                    self._trocr_model.eval()
                    self._trocr_model.to(self.config.DEVICE)
                except Exception as ex:
                    logger.error(f"Failed to load TrOCR model {model_name}: {ex}")

        return self._trocr_processor, self._trocr_model

    def recognize_region(self, region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs optimal OCR engine on a region crop based on region_type.
        Updates region dict with:
        - `extracted_text`: str
        - `confidence`: float (0.0 to 1.0)
        - `recognition_model_used`: str
        """
        crop_bgr = region.get("crop_bgr")
        region_type = region.get("region_type", "printed")

        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
            region["extracted_text"] = ""
            region["confidence"] = 0.0
            region["recognition_model_used"] = "None"
            return region

        # Dispatch based on region classification
        if region_type == "handwritten":
            text, conf, model_used = self._recognize_trocr(crop_bgr)
            if not text.strip() or conf < self.config.LOW_CONFIDENCE_THRESHOLD:
                p_text, p_conf, p_model = self._recognize_paddle(crop_bgr)
                if p_text.strip() and p_conf > conf:
                    text, conf, model_used = p_text, p_conf, f"PaddleOCR-HandwritingFallback ({p_model})"
        else: # Printed
            text, conf, model_used = self._recognize_paddle(crop_bgr)
            if not text.strip() or conf < self.config.LOW_CONFIDENCE_THRESHOLD:
                t_text, t_conf, t_model = self._recognize_trocr(crop_bgr)
                if t_text.strip() and t_conf > conf:
                    text, conf, model_used = t_text, t_conf, f"TrOCR-PrintedFallback ({t_model})"

        region["extracted_text"] = text.strip()
        region["confidence"] = round(float(conf), 4)
        region["recognition_model_used"] = model_used

        return region

    def _recognize_paddle(self, crop_bgr: np.ndarray) -> Tuple[str, float, str]:
        """Runs PaddleOCR recognition engine on image crop."""
        try:
            paddle_engine = self._get_paddle_recognizer()
            res = paddle_engine.ocr(crop_bgr)

            if res and len(res) > 0:
                res_item = res[0]
                if isinstance(res_item, dict):
                    texts = res_item.get("rec_texts", [])
                    scores = res_item.get("rec_scores", [])
                    if texts and scores:
                        return texts[0], float(scores[0]), "PaddleOCR-v4"
                elif isinstance(res_item, list) and len(res_item) > 0:
                    first_item = res_item[0]
                    if isinstance(first_item, (tuple, list)) and len(first_item) >= 2:
                        text, conf = first_item[0], float(first_item[1])
                        return text, conf, "PaddleOCR-v4"
                    elif isinstance(first_item, tuple) and isinstance(first_item[0], str):
                        return first_item[0], float(first_item[1]), "PaddleOCR-v4"
                elif isinstance(res_item, tuple) and len(res_item) >= 2:
                    return str(res_item[0]), float(res_item[1]), "PaddleOCR-v4"
        except Exception as e:
            logger.debug(f"PaddleOCR rec exception: {e}")

        return "", 0.0, "PaddleOCR-Failed"

    def _recognize_trocr(self, crop_bgr: np.ndarray) -> Tuple[str, float, str]:
        """Runs Hugging Face TrOCR handwritten model on image crop."""
        try:
            processor, model = self._get_trocr()
            if processor is None or model is None:
                return "", 0.0, "TrOCR-Unavailable"

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)

            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(self.config.DEVICE)

            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=64)

            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            conf = 0.88 if len(text.strip()) > 0 else 0.0

            return text, conf, f"HF-{self.config.TROCR_MODEL_NAME}"
        except Exception as e:
            logger.debug(f"TrOCR exception: {e}")

        return "", 0.0, "TrOCR-Failed"
