import os
import glob
import json
import cv2
import logging
from worksheet_ocr import WorksheetOCRPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RunPipeline")

def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Locate all sample worksheet images
    image_paths = sorted(glob.glob(os.path.join(workspace_dir, "WhatsApp Image *.jpeg")))
    if not image_paths:
        image_paths = sorted(glob.glob(os.path.join(workspace_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(workspace_dir, "*.png")))
        
    print(f"Found {len(image_paths)} sample worksheet images to process.")
    for p in image_paths:
        print(f"  - {os.path.basename(p)}")

    pipeline = WorksheetOCRPipeline()
    
    output_dir = os.path.join(workspace_dir, "ocr_results")
    os.makedirs(output_dir, exist_ok=True)

    for idx, img_path in enumerate(image_paths):
        page_num = idx + 1
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n=======================================================")
        print(f"Processing Page {page_num}: {base_name}")
        print(f"=======================================================")
        
        result = pipeline.process_image(img_path, page_num=page_num)
        
        # Extract annotated image for output saving
        annotated_bgr = result.pop("annotated_bgr", None)
        
        # Save JSON output
        json_path = os.path.join(output_dir, f"{base_name}_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f"Saved JSON result -> {json_path}")
        print("\nStructured JSON Output Preview:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Save annotated image
        if annotated_bgr is not None:
            vis_path = os.path.join(output_dir, f"{base_name}_annotated.jpeg")
            cv2.imwrite(vis_path, annotated_bgr)
            print(f"Saved annotated image -> {vis_path}")

if __name__ == "__main__":
    main()
