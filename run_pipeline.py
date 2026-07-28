import os
import glob
import json
import cv2
import logging
from worksheet_ocr import WorksheetOCRPipeline, is_pdf_file, get_pdf_page_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RunPipeline")

def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Locate all sample worksheet images and PDFs
    input_paths = sorted(glob.glob(os.path.join(workspace_dir, "WhatsApp Image *.jpeg")))
    if not input_paths:
        input_paths = sorted(glob.glob(os.path.join(workspace_dir, "*.jpg"))) + \
                      sorted(glob.glob(os.path.join(workspace_dir, "*.jpeg"))) + \
                      sorted(glob.glob(os.path.join(workspace_dir, "*.png")))
                      
    pdf_paths = sorted(glob.glob(os.path.join(workspace_dir, "*.pdf")))
    all_files = input_paths + pdf_paths
        
    print(f"Found {len(all_files)} sample document/image file(s) to process.")
    for p in all_files:
        is_pdf = is_pdf_file(p)
        kind = "PDF Document" if is_pdf else "Worksheet Image"
        print(f"  - [{kind}] {os.path.basename(p)}")

    pipeline = WorksheetOCRPipeline()
    
    output_dir = os.path.join(workspace_dir, "ocr_results")
    os.makedirs(output_dir, exist_ok=True)

    global_page_count = 0
    for file_path in all_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        if is_pdf_file(file_path):
            total_pages = get_pdf_page_count(file_path)
            print(f"\n=======================================================")
            print(f"Processing PDF File: {base_name} ({total_pages} page(s))")
            print(f"=======================================================")
            
            for p in range(1, total_pages + 1):
                global_page_count += 1
                print(f"--- Processing PDF Page {p}/{total_pages} ---")
                result = pipeline.process_image(file_path, page_num=p)
                save_pipeline_result(result, output_dir, f"{base_name}_page_{p}")
        else:
            global_page_count += 1
            print(f"\n=======================================================")
            print(f"Processing Image Page {global_page_count}: {base_name}")
            print(f"=======================================================")
            result = pipeline.process_image(file_path, page_num=global_page_count)
            save_pipeline_result(result, output_dir, base_name)

def save_pipeline_result(result, output_dir, file_prefix):
    annotated_bgr = result.pop("annotated_bgr", None)
    
    # Save JSON output
    json_path = os.path.join(output_dir, f"{file_prefix}_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print(f"Saved JSON result -> {json_path}")
    
    # Display AI Notebook Intelligence Summary
    subj = result.get("subject_identification", {}).get("identified_subject", "General")
    analytics = result.get("learning_analytics", {})
    feedback = result.get("teacher_feedback", {})
    flashcards = result.get("personalized_flashcards", [])
    
    print("--- AI Notebook Intelligence Summary ---")
    print(f"Subject Identified: {subj}")
    print(f"Subject Mastery Score: {analytics.get('subject_mastery_score', 0)}%")
    print(f"Grammar Score: {analytics.get('grammar_score', 0)}% | Vocabulary Score: {analytics.get('vocabulary_score', 0)}%")
    print(f"Teacher Summary Comment: {feedback.get('teacher_summary_comment', '')}")
    print(f"Generated {len(flashcards)} personalized revision flashcards.")

    # Save annotated image
    if annotated_bgr is not None:
        vis_path = os.path.join(output_dir, f"{file_prefix}_annotated.jpeg")
        cv2.imwrite(vis_path, annotated_bgr)
        print(f"Saved annotated image -> {vis_path}")

if __name__ == "__main__":
    main()
