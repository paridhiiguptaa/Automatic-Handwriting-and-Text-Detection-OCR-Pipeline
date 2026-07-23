import os
import json
import cv2
import gradio as gr
from worksheet_ocr import WorksheetOCRPipeline

# Global pipeline instance
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = WorksheetOCRPipeline()
    return pipeline

def process_worksheet(image):
    if image is None:
        return None, None, "Please upload a worksheet image."
        
    ocr_pipe = get_pipeline()
    
    # Process image
    result = ocr_pipe.process_image(image, page_num=1)
    
    annotated_bgr = result.pop("annotated_bgr", None)
    annotated_rgb = None
    if annotated_bgr is not None:
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    
    # Generate human-readable summary
    total_regions = len(result.get("regions", []))
    printed_count = sum(1 for r in result.get("regions", []) if r.get("model") == "PaddleOCR")
    handwritten_count = sum(1 for r in result.get("regions", []) if "TrOCR" in str(r.get("model")))
    vlm_count = sum(1 for r in result.get("regions", []) if "SmolVLM" in str(r.get("model")))
    overall_conf = result.get("overall_confidence", 0.0)
    
    summary_text = (
        f"### 📊 Worksheet OCR Processing Summary\n"
        f"- **Overall Confidence**: `{overall_conf * 100:.1f}%`\n"
        f"- **Total Text Regions Detected**: `{total_regions}`\n"
        f"- **Printed Text (PaddleOCR)**: `{printed_count}`\n"
        f"- **Handwritten Text (TrOCR)**: `{handwritten_count}`\n"
        f"- **VLM Fallback Regions (SmolVLM2)**: `{vlm_count}`\n\n"
        f"*Preprocessing*: Skew angle corrected `{result.get('preprocessing_metadata', {}).get('skew_angle_deg', 0)}°`"
    )
    
    return annotated_rgb, summary_text, json_str

def build_gradio_app():
    with gr.Blocks(title="CPU-Friendly Intelligent Worksheet OCR Pipeline") as demo:
        gr.Markdown(
            """
            # 📚 CPU-Friendly Intelligent Worksheet OCR Pipeline
            ### High-Accuracy OCR for Children's Homework & Assignments using PaddleOCR, TrOCR & Selective VLM Fallback
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="numpy", label="Upload Worksheet Image")
                btn_run = gr.Button("🚀 Run OCR Pipeline", variant="primary")
                
                gr.Markdown(
                    """
                    **Pipeline Legend**:
                    - 🔵 **Blue Box**: Printed Text (PaddleOCR)
                    - 🟢 **Green Box**: Children's Handwriting (TrOCR)
                    - 🟠 **Orange Box**: Low-Confidence / Diagram / Math (SmolVLM2 Fallback)
                    """
                )
                
            with gr.Column(scale=1):
                output_annotated = gr.Image(label="Annotated Bounding Boxes & Predictions")
                output_summary = gr.Markdown(label="Pipeline Performance Summary")
                
        with gr.Row():
            output_json = gr.Code(label="Structured JSON Output", language="json")
            
        btn_run.click(
            fn=process_worksheet,
            inputs=[input_image],
            outputs=[output_annotated, output_summary, output_json]
        )
        
    return demo

if __name__ == "__main__":
    app = build_gradio_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
