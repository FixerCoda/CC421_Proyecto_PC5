import json

with open('notebooks/colab.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    print(f"Cell {i}: {cell['cell_type']}, source_len: {len(source)}, preview: {source[:60]}")

