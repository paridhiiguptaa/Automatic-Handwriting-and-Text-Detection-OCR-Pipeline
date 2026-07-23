import cv2
import numpy as np
import torch
from PIL import Image
import logging
import warnings

# Suppress minor HF warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class OCRRecognizer:
    """Stage 3: Hybrid OCR Recognition using PaddleOCR for printed text & TrOCR for handwritten text."""

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
                self._paddle_rec = PaddleOCR(lang=self.config.PADDLE_LANG)
            except Exception as e:
                logger.warning(f"PaddleOCR rec init fallback: {e}")
                self._paddle_rec = PaddleOCR()
        return self._paddle_rec

    def _get_trocr(self):
        if self._trocr_model is None:
            import transformers
            transformers.logging.set_verbosity_error()
            from transformers import RobertaTokenizer, ViTImageProcessor, TrOCRProcessor, VisionEncoderDecoderModel
            logger.info(f"Loading TrOCR model: {self.config.TROCR_MODEL_NAME}")
            
            try:
                img_proc = ViTImageProcessor.from_pretrained(self.config.TROCR_MODEL_NAME)
                tokenizer = RobertaTokenizer.from_pretrained(self.config.TROCR_MODEL_NAME)
                self._trocr_processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tokenizer)
                self._trocr_model = VisionEncoderDecoderModel.from_pretrained(self.config.TROCR_MODEL_NAME)
                self._trocr_model.eval()
                self._trocr_model.to(self.config.DEVICE)
            except Exception as e:
                logger.warning(f"TrOCR direct loading fallback: {e}")
                try:
                    self._trocr_processor = TrOCRProcessor.from_pretrained(self.config.TROCR_MODEL_NAME)
                    self._trocr_model = VisionEncoderDecoderModel.from_pretrained(self.config.TROCR_MODEL_NAME)
                    self._trocr_model.eval()
                    self._trocr_model.to(self.config.DEVICE)
                except Exception as ex:
                    logger.error(f"Failed to load TrOCR model: {ex}")

        return self._trocr_processor, self._trocr_model

    def recognize_region(self, region):
        """
        Recognize a single region crop based on its region_type.
        Returns updated region dict with:
        - extracted_text
        - confidence
        - bounding_box
        - recognition_model_used
        """
        crop_bgr = region["crop_bgr"]
        region_type = region.get("region_type", "printed")
        
        if crop_bgr is None or crop_bgr.size == 0:
            region["extracted_text"] = ""
            region["confidence"] = 0.0
            region["recognition_model_used"] = "None"
            return region

        if region_type == "printed":
            text, conf, model_used = self._recognize_paddle(crop_bgr)
            if not text or conf < 0.2:
                text_t, conf_t, _ = self._recognize_trocr(crop_bgr)
                if text_t:
                    text, conf, model_used = text_t, conf_t, "TrOCR-PrintedFallback"
        elif region_type in ("handwritten", "math"):
            text, conf, model_used = self._recognize_trocr(crop_bgr)
        else:
            text, conf, model_used = self._recognize_paddle(crop_bgr)
            if not text or conf < 0.2:
                text_t, conf_t, _ = self._recognize_trocr(crop_bgr)
                if text_t:
                    text, conf, model_used = text_t, conf_t, "TrOCR-PrintedFallback"

        region["extracted_text"] = text
        region["confidence"] = float(conf)
        region["recognition_model_used"] = model_used
        region["bounding_box"] = region["bbox"]
        
        return region

    def _recognize_paddle(self, crop_bgr):
        """PaddleOCR Recognition for printed text with fallback."""
        try:
            paddle_rec = self._get_paddle_recognizer()
            res = paddle_rec.ocr(crop_bgr, det=False, rec=True)
            if res and len(res) > 0 and len(res[0]) > 0:
                item = res[0][0]
                text, conf = item[0], item[1]
                return text.strip(), float(conf), "PaddleOCR"
        except Exception as e:
            logger.debug(f"PaddleOCR rec exception: {e}")
            
        return "", 0.0, "PaddleOCR"

    def _recognize_trocr(self, crop_bgr):
        """TrOCR Recognition for handwritten text."""
        try:
            processor, model = self._get_trocr()
            if processor is None or model is None:
                return "", 0.0, "TrOCR-Handwritten"

            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.config.DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=64
                )
                
            sequences = outputs.sequences
            text = processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()
            
            scores = outputs.scores
            if scores:
                probs = [torch.softmax(score, dim=-1).max().item() for score in scores]
                conf = sum(probs) / len(probs) if len(probs) > 0 else 0.85
            else:
                conf = 0.85
                
            return text, float(conf), "TrOCR-Handwritten"
            
        except Exception as e:
            logger.error(f"TrOCR recognition error: {e}")
            return "", 0.0, "TrOCR-Handwritten"
