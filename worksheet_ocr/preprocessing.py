import cv2
import numpy as np
from PIL import Image, ImageOps
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Stage 1: Image Preprocessing pipeline to crop paper sheet, deskew, enhance contrast, and preserve pencil marks."""
    
    def __init__(self, config=None):
        from .config import PipelineConfig
        self.config = config or PipelineConfig()

    def process(self, image_input):
        """
        Preprocess input image.
        1. Correct orientation (EXIF)
        2. Automatic Paper Page Crop (removes surrounding dark/table background)
        3. Deskew
        4. CLAHE contrast enhancement
        5. Denoising while preserving pencil strokes
        """
        # Load image to BGR numpy array
        img_bgr = self._load_as_bgr(image_input)
        
        # 1. Correct Orientation (EXIF check)
        img_bgr = self._correct_exif_orientation(img_bgr, image_input)
        
        # 2. Crop Paper Sheet Page (removes extra background)
        paper_cropped, paper_found = self.crop_paper_sheet(img_bgr)
        
        # 3. Deskew image
        deskewed_img, angle = self.deskew(paper_cropped)
        
        # 4. Enhance Contrast preserving pencil strokes (CLAHE in LAB color space)
        enhanced_img = self.enhance_contrast_lab(deskewed_img)
        
        # 5. Selective Denoising preserving fine handwriting lines
        denoised_img = self.denoise_pencil_safe(enhanced_img)
        
        metadata = {
            "paper_page_detected": paper_found,
            "skew_angle_deg": round(angle, 2),
            "original_shape": img_bgr.shape[:2],
            "processed_shape": denoised_img.shape[:2]
        }
        
        return denoised_img, metadata

    def crop_paper_sheet(self, img_bgr):
        """
        Detect outer paper contour and perform perspective warp transformation 
        to crop strictly to the worksheet page, removing table/background clutter.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Blur and edge detection for document boundary
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Morphological gradient to emphasize paper edges against dark/table backgrounds
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        
        # Thresholding to locate bright paper area
        thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_bgr, False
            
        # Find largest contour by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        img_area = h * w
        
        paper_contour = None
        for c in contours[:5]:
            area = cv2.contourArea(c)
            # Must occupy at least 20% of image area to be the main paper sheet
            if area < (img_area * 0.20):
                continue
                
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                paper_contour = approx
                break
            elif len(approx) > 4:
                # Convex hull approach for quadrilateral paper sheet approximation
                hull = cv2.convexHull(c)
                hull_peri = cv2.arcLength(hull, True)
                hull_approx = cv2.approxPolyDP(hull, 0.03 * hull_peri, True)
                if len(hull_approx) == 4:
                    paper_contour = hull_approx
                    break
                    
        if paper_contour is None:
            # Fallback: check bounding box of largest bright paper contour
            largest_c = contours[0]
            if cv2.contourArea(largest_c) > (img_area * 0.30):
                x, y, bw, bh = cv2.boundingRect(largest_c)
                # Crop strictly to bounding rectangle if aspect ratio is reasonable
                if bw > (w * 0.4) and bh > (h * 0.4):
                    pad = 5
                    crop_x1, crop_y1 = max(0, x - pad), max(0, y - pad)
                    crop_x2, crop_y2 = min(w, x + bw + pad), min(h, y + bh + pad)
                    return img_bgr[crop_y1:crop_y2, crop_x1:crop_x2], True
            return img_bgr, False
            
        # Perspective transform for 4-corner paper sheet
        pts = paper_contour.reshape(4, 2)
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Compute width and height of transformed paper page
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_bgr, M, (max_width, max_height), flags=cv2.INTER_CUBIC)
        
        return warped, True

    def _order_points(self, pts):
        """Order 4 contour points: [top-left, top-right, bottom-right, bottom-left]."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)] # Bottom-right has largest sum
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right has smallest difference
        rect[3] = pts[np.argmax(diff)] # Bottom-left has largest difference
        return rect

    def _load_as_bgr(self, image_input):
        if isinstance(image_input, str):
            pil_img = Image.open(image_input)
            pil_img = ImageOps.exif_transpose(pil_img)
            img_np = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, Image.Image):
            pil_img = ImageOps.exif_transpose(image_input)
            img_np = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            return image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _correct_exif_orientation(self, img_bgr, image_input):
        return img_bgr

    def deskew(self, img_bgr):
        """Deskew image using minimum area rectangle of text contours."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        angles = []
        h, w = img_bgr.shape[:2]
        min_area = (h * w) * 0.001
        
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            rect = cv2.minAreaRect(c)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            
            if abs(angle) <= self.config.DESKEW_ANGLE_LIMIT:
                angles.append(angle)
                
        if not angles:
            return img_bgr, 0.0
            
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.3:
            return img_bgr, 0.0
            
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, median_angle

    def enhance_contrast_lab(self, img_bgr):
        """Enhance contrast using CLAHE on L channel in LAB color space."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_TILE_GRID_SIZE
        )
        l_enhanced = clahe.apply(l)
        
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def denoise_pencil_safe(self, img_bgr):
        """Bilateral filtering to smooth background noise while keeping sharp pencil/pen edges."""
        denoised = cv2.bilateralFilter(
            img_bgr,
            d=self.config.BILATERAL_D,
            sigmaColor=self.config.BILATERAL_SIGMA_COLOR,
            sigmaSpace=self.config.BILATERAL_SIGMA_SPACE
        )
        return denoised
