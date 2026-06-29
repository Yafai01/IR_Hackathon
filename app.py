import os
import time
import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch

from inference.pipeline import IRPipeline

# =====================================================
# LOAD MODEL
# =====================================================

try:
    pipeline = IRPipeline(
        "checkpoints/best_srresnet.pth",
        "checkpoints/best_colorizer.pth"
    )
except Exception as e:
    print(f"Failed to load pipeline: {e}")
    pipeline = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# =====================================================
# RUN PIPELINE
# =====================================================

def run_pipeline(folder, brightness, progress=gr.Progress()):
    if pipeline is None:
        return (None, None, None, None, "❌ Pipeline failed to load.", "-", "-", "-", None)

    if not folder or not os.path.exists(folder):
        return (None, None, None, None, "❌ Invalid or empty folder path provided.", "-", "-", "-", None)

    try:
        progress(0.1, desc="Initializing...")
        start = time.time()

        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", "prediction.png")

        progress(0.4, desc="Running Inference...")
        
        # We also need to get the images to return to Gradio.
        # Instead of reading from output_path, we can just let pipeline save them,
        # and then read them back.
        pipeline.run(folder, output_path, brightness)
        
        elapsed = round(time.time() - start, 2)
        progress(0.8, desc="Loading outputs...")

        thermal_path = "outputs/thermal_input.png"
        sr_path = "outputs/sr_output.png"
        gt_path = "outputs/ground_truth.png"

        thermal_img = Image.open(thermal_path) if os.path.exists(thermal_path) else None
        sr_img = Image.open(sr_path) if os.path.exists(sr_path) else None
        pred_img = Image.open(output_path) if os.path.exists(output_path) else None
        gt_img = Image.open(gt_path) if os.path.exists(gt_path) else None

        res_text = f"{pred_img.width}x{pred_img.height}" if pred_img else "Unknown"

        status = f"✅ Finished successfully in {elapsed}s."
        inf_time = f"{elapsed} s"
        hw_used = f"{DEVICE_NAME} ({DEVICE.type.upper()})"

        progress(1.0, desc="Done")

        return (
            thermal_img,
            sr_img,
            pred_img,
            gt_img,
            status,
            inf_time,
            hw_used,
            res_text,
            output_path
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (None, None, None, None, f"❌ Error: {str(e)}", "-", "-", "-", None)


def clear_inputs():
    return ("", 2.0, None, None, None, None, "Ready", "-", "-", "-", None)


# =====================================================
# GRADIO UI
# =====================================================

TITLE = """
# 🌍 Thermal Satellite Image Colorization AI
### Super Resolution + Colorization using Deep Learning
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(TITLE)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Settings")
            sample_folder = gr.Textbox(
                label="Sample Folder Path",
                placeholder="sample_input/sample_00187"
            )
            
            brightness_slider = gr.Slider(
                label="Brightness Multiplier",
                minimum=0.5,
                maximum=5.0,
                value=2.0,
                step=0.1
            )

            with gr.Row():
                run_btn = gr.Button("🚀 Run Pipeline", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")

            gr.Markdown("### System Info")
            with gr.Group():
                status_panel = gr.Textbox(label="Status", value="Ready", interactive=False)
                inf_time = gr.Textbox(label="Inference Time", value="-", interactive=False)
                hw_used = gr.Textbox(label="Hardware Used", value="-", interactive=False)
                img_res = gr.Textbox(label="Output Resolution", value="-", interactive=False)

            download = gr.File(label="Download Prediction")

        with gr.Column(scale=2):
            gr.Markdown("### Visualization")
            with gr.Row():
                thermal_img = gr.Image(label="Thermal Input (False Color)", height=300)
                sr_img = gr.Image(label="Super Resolution Output", height=300)

            with gr.Row():
                pred_img = gr.Image(label="Predicted RGB", height=300)
                gt_img = gr.Image(label="Ground Truth RGB", height=300)

    # Interactions
    run_btn.click(
        fn=run_pipeline,
        inputs=[sample_folder, brightness_slider],
        outputs=[
            thermal_img,
            sr_img,
            pred_img,
            gt_img,
            status_panel,
            inf_time,
            hw_used,
            img_res,
            download
        ]
    )

    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[
            sample_folder,
            brightness_slider,
            thermal_img,
            sr_img,
            pred_img,
            gt_img,
            status_panel,
            inf_time,
            hw_used,
            img_res,
            download
        ]
    )


# =====================================================
# LAUNCH
# =====================================================

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )