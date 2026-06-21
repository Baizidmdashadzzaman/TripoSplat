import json
from pathlib import Path

def create_notebook():
    # Read files
    model_code = Path("model.py").read_text(encoding="utf-8")
    triposplat_code = Path("triposplat.py").read_text(encoding="utf-8")
    run_gradio_code = Path("run_gradio.py").read_text(encoding="utf-8")

    # Modify run_gradio_code to use share=True for Kaggle/Colab access
    run_gradio_code = run_gradio_code.replace(
        'server_port=7860,',
        'server_port=7860,\n        share=True,'
    )

    # Build notebook structure
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# TripoSplat Kaggle / Colab Notebook\n",
                "\n",
                "This notebook allows you to run TripoSplat on Kaggle (or Google Colab) using a GPU (e.g., T4) or CPU fallback.\n",
                "\n",
                "### Prerequisites\n",
                "Ensure that GPU accelerator is enabled in your Kaggle notebook settings (Settings -> Accelerator -> GPU T4 x2 or GPU T4)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Install dependencies\n",
                "!pip install -q safetensors pillow tqdm torchvision huggingface_hub gradio\n",
                "\n",
                "# 2. Clone repository to get the static files (HTML viewer template and example inputs)\n",
                "import os\n",
                "if not os.path.exists(\"static\"):\n",
                "    print(\"Cloning repository for static files...\")\n",
                "    !git clone https://github.com/Baizidmdashadzzaman/TripoSplat.git temp_repo\n",
                "    !mv temp_repo/static .\n",
                "    !rm -rf temp_repo\n",
                "\n",
                "# 3. Download weights from Hugging Face\n",
                "from huggingface_hub import snapshot_download\n",
                "\n",
                "print(\"Downloading weights from VAST-AI/TripoSplat...\")\n",
                "snapshot_download(repo_id='VAST-AI/TripoSplat', local_dir='ckpts/')\n",
                "print(\"Download finished!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%writefile model.py\n" + model_code
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%writefile triposplat.py\n" + triposplat_code
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Run Example CLI Inference\n",
                "# Import components and run the pipeline\n",
                "import torch\n",
                "from triposplat import TripoSplatPipeline\n",
                "\n",
                "# Check if GPU is available\n",
                "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
                "print(f\"Using device: {device}\")\n",
                "\n",
                "pipe = TripoSplatPipeline(\n",
                "    ckpt_path              = \"ckpts/diffusion_models/triposplat_fp16.safetensors\",\n",
                "    decoder_path           = \"ckpts/vae/triposplat_vae_decoder_fp16.safetensors\",\n",
                "    dinov3_path            = \"ckpts/clip_vision/dino_v3_vit_h.safetensors\",\n",
                "    flux2_vae_encoder_path = \"ckpts/vae/flux2-vae.safetensors\",\n",
                "    rmbg_path              = \"ckpts/background_removal/birefnet.safetensors\",\n",
                "    device                 = device,\n",
                ")\n",
                "\n",
                "INPUT = \"static/example_inputs/building_stone_house.webp\"\n",
                "gaussian, prepared = pipe.run(INPUT, num_gaussians=262144, show_progress=True)\n",
                "\n",
                "prepared.save(\"preprocessed_image.webp\")\n",
                "gaussian.save_ply(\"output.ply\")\n",
                "gaussian.save_splat(\"output.splat\")\n",
                "print(\"Generation finished! Files saved: output.ply, output.splat, preprocessed_image.webp\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Run Gradio Web UI\n",
                "\n",
                "Run the cell below to launch the interactive Gradio web application. Because we passed `share=True`, Gradio will output a public `.gradio.live` link. You can open that link in your browser to interact with the model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%writefile run_gradio.py\n" + run_gradio_code
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Launch the Gradio Web App\n",
                "# This will output a public link (e.g. https://xxxx.gradio.live) to access the interface\n",
                "import torch\n",
                "from triposplat import TripoSplatPipeline\n",
                "\n",
                "# Dynamically detect device\n",
                "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
                "print(f\"Starting Gradio server on device: {device}\")\n",
                "\n",
                "# We run run_gradio.py directly via system call\n",
                "# But first, edit the pipeline initialization in run_gradio.py to match our current device\n",
                "with open(\"run_gradio.py\", \"r\", encoding=\"utf-8\") as f:\n",
                "    content = f.read()\n",
                "\n",
                "content = content.replace('device                 = \"cuda\",', f'device                 = \"{device}\",')\n",
                "\n",
                "with open(\"run_gradio.py\", \"w\", encoding=\"utf-8\") as f:\n",
                "    f.write(content)\n",
                "\n",
                "!python run_gradio.py"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Save to file
    out_path = Path("triposplat_kaggle.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Notebook created at: {out_path.resolve()}")

if __name__ == "__main__":
    create_notebook()
