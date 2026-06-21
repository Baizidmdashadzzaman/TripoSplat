import torch
import ipywidgets as widgets
from IPython.display import display
from PIL import Image
import plotly.graph_objects as go
import numpy as np

# Import the TripoSplat pipeline and the generate function
# Adjust import paths if necessary
from triposplat import TripoSplatPipeline
from run_gradio import generate

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# Initialize pipeline (if needed)
# PIPE = TripoSplatPipeline(device=device)  # Uncomment if you need explicit init

def on_upload_change(change):
    """Callback when an image is uploaded."""
    for name, file_info in upload.value.items():
        # Load and display the uploaded image
        img = Image.open(file_info["content"]).convert("RGB")
        display(img)
        # Run the generation (using default parameters; adjust as desired)
        prepared, view_html, download_path, info = generate(
            image_in=img,
            seed_in=42,
            steps_in=20,
            cfg_in=3.0,
            num_g_in="262144",
            fmt_in="ply",
        )
        # Show the viewer iframe or info
        if view_html:
            display(view_html)
        else:
            print("No viewer generated.")
        print(info)

# File upload widget
upload = widgets.FileUpload(accept="image/*", multiple=False)
upload.observe(on_upload_change, names="value")
print("Upload an image to run TripoSplat inference:")
display(upload)
