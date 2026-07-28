import os
# Configure environment flags before importing paddle or torch
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_DISABLE_PIR"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import json
import cv2
import numpy as np
import gradio as gr
from worksheet_ocr import NotebookOCRPipeline, is_pdf_file, get_pdf_page_count, render_pdf_page

# Global pipeline instance
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = NotebookOCRPipeline()
    return pipeline

def handle_file_change(file_obj):
    if file_obj is None:
        return gr.update(maximum=2, value=1, visible=False), "📁 **Status**: No file uploaded."

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    if is_pdf_file(file_path):
        try:
            count = get_pdf_page_count(file_path)
            max_pages = max(2, count)
            slider_visible = count > 1
            return gr.update(maximum=max_pages, value=1, visible=slider_visible), f"📄 **PDF Document Detected**: Total **{count}** page(s). Select page to analyze below."
        except Exception as e:
            return gr.update(maximum=2, value=1, visible=False), f"⚠️ Error reading PDF file: {str(e)}"
    else:
        return gr.update(maximum=2, value=1, visible=False), "📷 **Image Uploaded**: Ready for document layout & structure analysis."

def process_notebook_layout(file_obj, direct_image, page_num):
    input_source = None
    if file_obj is not None:
        input_source = file_obj.name if hasattr(file_obj, "name") else file_obj
    elif direct_image is not None:
        input_source = direct_image
    else:
        empty_msg = "⚠️ Please upload a notebook image or PDF document to process."
        return (
            "0", "0.0%", "0", "0",
            None, None, None, empty_msg,
            [], "{}"
        )

    ocr_pipe = get_pipeline()

    try:
        req_page = int(page_num) if page_num else 1
        result = ocr_pipe.process_image(input_source, page_num=req_page)
    except Exception as e:
        err_msg = f"### ❌ Error Processing Notebook Image\n`{str(e)}`"
        return (
            "0", "0.0%", "0", "0",
            None, None, None, err_msg,
            [], "{}"
        )

    # 1. Summary Metrics
    summary = result.get("summary", {})
    total_regions = str(summary.get("total_text_regions", 0))
    avg_conf_pct = f"{summary.get('average_confidence', 0.0) * 100:.1f}%"
    role_counts = summary.get("layout_role_counts", {})
    printed_cnt = str(sum(1 for r in result.get("regions", []) if r.get("region_type") == "printed"))
    handwritten_cnt = str(sum(1 for r in result.get("regions", []) if r.get("region_type") == "handwritten"))

    # 2. Extract Images
    cleaned_bgr = result.get("cleaned_bgr")
    annotated_bgr = result.get("annotated_bgr")

    cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB) if cleaned_bgr is not None else None
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB) if annotated_bgr is not None else None

    # Original Image rendering
    if isinstance(input_source, str) and input_source.lower().endswith(".pdf"):
        orig_pil = render_pdf_page(input_source, page_index=req_page-1)
        orig_rgb = cv2.cvtColor(cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
    elif isinstance(input_source, np.ndarray):
        orig_rgb = cv2.cvtColor(input_source, cv2.COLOR_BGR2RGB) if input_source.ndim == 3 else input_source
    elif isinstance(input_source, str):
        bgr = cv2.imread(input_source)
        orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
    else:
        orig_rgb = cleaned_rgb

    # 3. Formatted Notebook Markdown Preview
    formatted_markdown = result.get("formatted_markdown", result.get("reconstructed_text", ""))

    # 4. Build Detailed Layout Dataframe Table
    regions = result.get("regions", [])
    table_data = []
    for r in regions:
        bbox_str = f"[{r['bbox'][0]}, {r['bbox'][1]}, {r['bbox'][2]}, {r['bbox'][3]}]"
        conf_pct = f"{float(r['confidence']) * 100:.1f}%"
        table_data.append([
            r["reading_order"],
            r.get("layout_role", "paragraph").upper(),
            f"Level {r.get('indentation_level', 0)}",
            r["text"] if r["text"] else "*(blank)*",
            conf_pct,
            bbox_str,
            r["model_used"]
        ])

    # 5. Clean JSON Payload with Document Tree
    json_payload = {
        "milestone": result.get("milestone"),
        "summary": summary,
        "preprocessing_metadata": result.get("preprocessing_metadata"),
        "document_tree": result.get("document_tree"),
        "formatted_markdown": formatted_markdown,
        "regions": regions
    }
    json_str = json.dumps(json_payload, indent=2, ensure_ascii=False)

    return (
        total_regions, avg_conf_pct, printed_cnt, handwritten_cnt,
        orig_rgb, cleaned_rgb, annotated_rgb, formatted_markdown,
        table_data, json_str
    )

def build_gradio_app():
    with gr.Blocks(title="Notebook Layout Understanding & Structure (Milestone 2)", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📐 Notebook Layout Understanding & Document Structure Engine
            ### Milestone 2: Structural Role Classification, Hierarchy Indentation & Notebook Layout Reconstruction
            *Detects headings, question numbers, answer blocks, bullet lists, and math expressions to reconstruct formatted notebook structure.*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="📄 Upload Notebook Image or PDF Document",
                    file_types=[".png", ".jpg", ".jpeg", ".bmp", ".webp", ".pdf"]
                )

                status_box = gr.Markdown("📁 **Upload Status**: Upload a notebook image or PDF document.")

                page_slider = gr.Slider(
                    minimum=1,
                    maximum=2,
                    value=1,
                    step=1,
                    label="📖 Select PDF Page Number",
                    visible=False
                )

                direct_image = gr.Image(
                    label="📷 Direct Image Upload / Webcam Capture",
                    type="numpy"
                )

                btn_run = gr.Button("🚀 Run Layout & Notebook Structure Analysis", variant="primary")

                gr.Markdown(
                    """
                    **Structural Color Legend**:
                    - 🔵 **Blue**: Heading 1 / Heading 2
                    - 🟢 **Green**: Question Number / Question Label
                    - 🟠 **Orange**: Answer Section / Student Response
                    - 🟣 **Purple**: Mathematical Expression
                    - 🟡 **Yellow**: Bullet List Item
                    - 🩶 **Gray**: Paragraph Body Text
                    """
                )

            with gr.Column(scale=2):
                # Summary Metric Cards
                with gr.Row():
                    metric_regions = gr.Textbox(label="Total Regions Detected", value="0", interactive=False)
                    metric_conf = gr.Textbox(label="Average Confidence", value="0.0%", interactive=False)
                    metric_printed = gr.Textbox(label="Printed Regions", value="0", interactive=False)
                    metric_handwritten = gr.Textbox(label="Handwritten Regions", value="0", interactive=False)

                with gr.Tabs():
                    with gr.TabItem("📐 Structural Overlay & Notebook Markdown Preview"):
                        with gr.Row():
                            img_annotated = gr.Image(label="Structural Role Bounding Boxes")
                            md_preview = gr.Markdown(label="Reconstructed Notebook Markdown Preview")

                    with gr.TabItem("🖼️ Image Preprocessing Inspection"):
                        with gr.Row():
                            img_orig = gr.Image(label="1. Original Uploaded Image")
                            img_clean = gr.Image(label="2. Preprocessed Cleaned Image (Paper Crop & CLAHE)")

                    with gr.TabItem("📊 Layout Hierarchy & Region Table"):
                        df_regions = gr.Dataframe(
                            headers=["Order #", "Layout Role", "Indentation", "Extracted Text", "Confidence", "Bounding Box", "OCR Model"],
                            datatype=["number", "str", "str", "str", "str", "str", "str"],
                            label="Notebook Layout Structure Table"
                        )

                    with gr.TabItem("🔍 Document Tree & Metadata (JSON)"):
                        output_json = gr.Code(label="Structured Document Payload", language="json")

        file_input.change(
            fn=handle_file_change,
            inputs=[file_input],
            outputs=[page_slider, status_box]
        )

        btn_run.click(
            fn=process_notebook_layout,
            inputs=[file_input, direct_image, page_slider],
            outputs=[
                metric_regions, metric_conf, metric_printed, metric_handwritten,
                img_orig, img_clean, img_annotated, md_preview,
                df_regions, output_json
            ]
        )

    return demo

if __name__ == "__main__":
    app = build_gradio_app()
    app.launch(server_name="127.0.0.1", share=False)
