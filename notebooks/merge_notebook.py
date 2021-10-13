import nbformat
import os

os.chdir(os.path.realpath(os.path.dirname(__file__)))

notebook_paths = [
    "training_on_cloud.ipynb",
    "building_dataset.ipynb",
    "model_for_inference.ipynb"
]
output_path = "all_in_one.ipynb"

nbformat_version = 4

output_notebook = None
for file in notebook_paths:
    print(f"Reading {file}")
    notebook = nbformat.read(file, nbformat_version)
    if not output_notebook:
        output_notebook = nbformat.v4.new_notebook(metadata=notebook.metadata)
    # Concatenating the notebooks
    output_notebook.cells = output_notebook.cells + notebook.cells

# Saving the new notebook 
print(f"Writing to {output_path}")
nbformat.write(output_notebook, output_path)
