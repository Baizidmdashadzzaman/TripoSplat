# Implementation Plan – Update `triposplat_kaggle.ipynb`

## Goal Description
Create a single, self‑contained Jupyter notebook (`triposplat_kaggle.ipynb`) that embeds **all source code** from the repository (`model.py`, `triposplat.py`, `run_gradio.py`, `create_notebook.py`, `inline_upload.py`) as separate notebook cells. Replace any `Path(__file__)` usage with `Path.cwd()`. Add an inline image‑upload widget, UI sliders for parameters (steps, guidance scale, number of gaussians), and a download button for the generated mesh. The notebook should run top‑to‑bottom without writing external files.

## User Review Required
> **IMPORTANT** This operation will **overwrite** the existing `triposplat_kaggle.ipynb`. Make a backup if you wish to keep the current version.

## Open Questions (answered)
- **Include all source files?** Yes – embed the full contents of `model.py`, `triposplat.py`, `run_gradio.py` (excluding the `if __name__ == "__main__":` launch block), `create_notebook.py`, and `inline_upload.py`.
- **Import order?** Keep logical order: imports → device selector → model definitions → VAE/decoder → utility functions → UI widgets.
- **Path handling?** Replace `Path(__file__)` with `Path.cwd()` to ensure file paths resolve inside the notebook.
- **Device selector widget?** Include a simple dropdown (`CPU`/`CUDA`) that sets the device for the model.

## Proposed Changes
---
### [MODIFY] `implementation_plan.md`
- Updated the plan to target `triposplat_kaggle.ipynb` directly.

### [MODIFY] `triposplat_kaggle.ipynb`
1. **Cell 1 – Title & Overview** (Markdown) – notebook purpose and usage.
2. **Cell 2 – Imports** (Code) – `torch`, `ipywidgets`, `IPython.display`, `PIL.Image`, `pathlib.Path`, plus all other libraries used in the repo.
3. **Cell 3 – Device Selection Widget** (Code) – dropdown widget that sets a global `DEVICE` variable.
4. **Cell 4‑N – Model Code** – paste the entire content of `model.py`.
5. **Cell N+1‑M – Triposplat Core** – paste `triposplat.py`.
6. **Cell M+1‑P – VAE & Decoder** – paste `run_gradio.py` without the launch block, keeping the `generate` function and any static HTML helpers.
7. **Cell P+1‑Q – Notebook Helpers** – paste `create_notebook.py` utilities (if useful).
8. **Cell Q – Inline Upload UI** – adapt `inline_upload.py` logic to call `generate` and display the 3‑D viewer.
9. **Cell Q+1 – Parameter Sliders** – `ipywidgets.IntSlider` for `steps`, `guidance_scale`, `num_gaussians` that feed into `generate`.
10. **Cell Q+2 – Execution Example** – demonstrate uploading an image, adjusting sliders, and rendering the mesh.
11. **Cell Q+3 – Save/Download Helper** – button to download the generated `.ply` file.

## Verification Plan
### Automated Tests
- Execute the notebook via `jupyter nbconvert --to notebook --execute triposplat_kaggle.ipynb` and ensure it completes without errors.
- Confirm that the final cell creates a downloadable `.ply` file.

### Manual Verification
- Open the notebook in Jupyter, run all cells, upload a sample image, adjust sliders, and verify that the 3‑D viewer appears and the mesh can be downloaded.
---
