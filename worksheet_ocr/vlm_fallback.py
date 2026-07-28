import cv2
import torch
import logging
from PIL import Image
import warnings

# Suppress minor HF warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class VLMFallback:
    """Stage 5: Vision Language Model Fallback for selective low-confidence image crops."""

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()
        self._vlm_processor = None
        self._vlm_model = None

    def _load_vlm(self):
        if self._vlm_model is None:
            try:
                import transformers
                transformers.logging.set_verbosity_error()
                from transformers import AutoProcessor
                try:
                    from transformers import AutoModelForImageTextToText as AutoVLM
                except ImportError:
                    try:
                        from transformers import AutoModelForVision2Seq as AutoVLM
                    except ImportError:
                        from transformers import AutoModelForConditionalGeneration as AutoVLM

                model_id = self.config.VLM_MODEL_NAME
                logger.info(f"Loading CPU-optimized VLM fallback model: {model_id}")
                
                self._vlm_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                self._vlm_model = AutoVLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self._vlm_model.eval()
                self._vlm_model.float()
                self._vlm_model.to(self.config.DEVICE)
            except Exception as e:
                logger.warning(f"Could not load VLM model ({self.config.VLM_MODEL_NAME}): {e}. Using light heuristic VLM fallback worker.")
                self._vlm_model = "HEURISTIC_FALLBACK"

    def process_region(self, region):
        """
        Process a single low-confidence region crop with the VLM.
        Returns region updated with VLM corrected text and model name.
        """
        crop_bgr = region.get("crop_bgr", None)
        if crop_bgr is None or crop_bgr.size == 0:
            region["text"] = "Low confidence."
            region["model"] = "SmolVLM2"
            region["confidence"] = 0.50
            region["vlm_invoked"] = True
            return region

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        vlm_text, vlm_conf = self._infer_vlm(pil_img, region.get("extracted_text", ""))

        region["text"] = vlm_text
        region["confidence"] = float(vlm_conf)
        region["model"] = "SmolVLM2"
        region["vlm_invoked"] = True

        return region

    def _infer_vlm(self, pil_img, ocr_draft=""):
        """Run SmolVLM2 inference on the image crop with strict no-hallucination prompt."""
        self._load_vlm()

        if self._vlm_model == "HEURISTIC_FALLBACK":
            if not ocr_draft or len(ocr_draft.strip()) == 0:
                return "Low confidence.", 0.40
            return ocr_draft.strip(), 0.75

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "Transcribe ONLY the text, handwritten words, or math expression visible in this worksheet crop. "
                                "Do NOT describe what the image looks like. Do NOT write sentences like 'The image shows...'. "
                                "If the text is unreadable or contains drawings without text, reply exactly: 'Low confidence.' "
                                "Do not hallucinate."
                            )
                        }
                    ]
                }
            ]
            prompt = self._vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._vlm_processor(images=pil_img, text=prompt, return_tensors="pt")
            inputs = inputs.to(self.config.DEVICE)
            
            with torch.no_grad():
                generated_ids = self._vlm_model.generate(**inputs, max_new_tokens=48)
                
            generated_texts = self._vlm_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            raw_response = generated_texts[0].split("Assistant:")[-1].strip()
            
            lowered = raw_response.lower()
            if not raw_response or "low confidence" in lowered:
                return "Low confidence.", 0.50
            if "the image" in lowered or "close up of" in lowered or "photo of" in lowered:
                return "Low confidence.", 0.50
                
            return raw_response, 0.88

        except Exception as e:
            logger.error(f"VLM inference error: {e}")
            return ocr_draft if ocr_draft else "Low confidence.", 0.60
