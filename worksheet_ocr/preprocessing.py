import cv2
import numpy as np
from PIL import Image, ImageOps
import io
import logging
from .pdf_utils import render_pdf_page

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Stage 1: Preprocessing Pipeline for Educational Notebook Images & Worksheets.
    Features:
    1. EXIF Orientation & PDF Rendering
    2. Paper Page Contour Detection & 4-Point Perspective Warp Auto-Crop
    3. Shadow Removal & Illumination Normalization (LAB Space)
    4. Deskewing via Minimum Bounding Box Angle Detection
    5. Selective CLAHE Contrast Enhancement (LAB Space)
    6. Pencil-Safe Edge-Preserving Denoising
    """

    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def process(self, image_input, page_index: int = 0):
        """
        Main entry point for image preprocessing.
        Returns:
            processed_bgr (np.ndarray): Cleaned, normalized, perspective-corrected image.
            metadata (dict): Preprocessing parameters, crop status, skew angle, dimensions.
        """
        img_bgr = self._load_as_bgr(image_input, page_index=page_index)
        img_bgr = self._correct_exif_orientation(img_bgr, image_input)
        
        orig_h, orig_w = img_bgr.shape[:2]

        # 1. Page Contour Auto-Crop & Perspective Correction
        paper_cropped, paper_found = self.crop_paper_sheet(img_bgr)
        
        # 2. Shadow Attenuation & Illumination Normalization
        shadow_free = self.remove_shadows_and_normalize(paper_cropped)

        # 3. Deskewing
        deskewed, skew_angle = self.deskew(shadow_free)

        # 4. CLAHE Contrast Enhancement
        enhanced = self.enhance_contrast_lab(deskewed)

        # 5. Pencil-Safe Denoising
        denoised = self.denoise_pencil_safe(enhanced)

        proc_h, proc_w = denoised.shape[:2]

        metadata = {
            "original_dimensions": [orig_w, orig_h],
            "processed_dimensions": [proc_w, proc_h],
            "paper_page_detected": paper_found,
            "perspective_corrected": paper_found,
            "shadow_reduction_applied": True,
            "skew_angle_degrees": round(float(skew_angle), 2)
        }

        return denoised, metadata

    def _load_as_bgr(self, image_input, page_index: int = 0):
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input.copy()
        
        if isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image_input, (str, io.BytesIO)):
            if isinstance(image_input, str) and image_input.lower().endswith(".pdf"):
                pil_img = render_pdf_page(image_input, page_index=page_index)
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is not None:
                    return img
            
            # Fallback PIL load
            pil_img = Image.open(image_input)
            rgb = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _correct_exif_orientation(self, img_bgr, image_input):
        try:
            if isinstance(image_input, str) and not image_input.lower().endswith(".pdf"):
                pil_img = Image.open(image_input)
                pil_oriented = ImageOps.exif_transpose(pil_img)
                rgb = np.array(pil_oriented.convert("RGB"))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
        return img_bgr

    def crop_paper_sheet(self, img_bgr):
        """Locates paper boundary and warps perspective to crop background table clutter."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_bgr, False

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        img_area = h * w

        paper_pts = None
        for c in contours[:5]:
            area = cv2.contourArea(c)
            if area < (img_area * self.config.PAPER_MIN_AREA_RATIO):
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                paper_pts = approx.reshape(4, 2)
                break
            else:
                hull = cv2.convexHull(c)
                hull_peri = cv2.arcLength(hull, True)
                hull_approx = cv2.approxPolyDP(hull, 0.03 * hull_peri, True)
                if len(hull_approx) == 4:
                    paper_pts = hull_approx.reshape(4, 2)
                    break

        if paper_pts is None:
            largest_c = contours[0]
            if cv2.contourArea(largest_c) > (img_area * 0.30):
                x, y, bw, bh = cv2.boundingRect(largest_c)
                pad_x, pad_y = int(bw * 0.01), int(bh * 0.01)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
                return img_bgr[y1:y2, x1:x2], True
            return img_bgr, False

        warped = self._four_point_transform(img_bgr, paper_pts)
        return warped, True

    def _four_point_transform(self, image, pts):
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def remove_shadows_and_normalize(self, img_bgr):
        """Attenuate uneven notebook lighting and shadows via LAB background division."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        bg = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        bg = cv2.GaussianBlur(bg, (21, 21), 0)

        bg_float = np.maximum(bg.astype(np.float32), 1.0)
        l_norm = (l.astype(np.float32) / bg_float) * 255.0
        l_norm = np.clip(l_norm, 0, 255).astype(np.uint8)

        normalized_lab = cv2.merge([l_norm, a, b])
        return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    def deskew(self, img_bgr):
        """Rotates image to compensate for camera tilt."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10:
            return img_bgr, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > self.config.DESKEW_MAX_ANGLE or abs(angle) < 0.3:
            return img_bgr, 0.0

        h, w = img_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, angle

    def enhance_contrast_lab(self, img_bgr):
        """Enhances local contrast on Lightness channel using CLAHE to keep pencil text clear."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_TILE_GRID_SIZE
        )
        l_clahe = clahe.apply(l)

        enhanced_lab = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def denoise_pencil_safe(self, img_bgr):
        """Bilateral filter removes paper grain noise while preserving sharp pencil stroke edges."""
        return cv2.bilateralFilter(
            img_bgr,
            d=self.config.BILATERAL_D,
            sigmaColor=self.config.BILATERAL_SIGMA_COLOR,
            sigmaSpace=self.config.BILATERAL_SIGMA_SPACE
        )
