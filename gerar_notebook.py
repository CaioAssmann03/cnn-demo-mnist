"""Converte cnn_demo.py (células '# %%') em cnn_demo.ipynb (Jupyter/Colab)."""
import re
import nbformat as nbf

with open("cnn_demo.py", encoding="utf-8") as f:
    texto = f.read()

blocos = re.split(r"(?m)^# %%(?: \[markdown\])?\s*$", texto)
marcas = re.findall(r"(?m)^# %%( \[markdown\])?\s*$", texto)

nb = nbf.v4.new_notebook()
for marca, bloco in zip(marcas, blocos[1:]):
    bloco = bloco.strip("\n")
    if marca:  # célula de texto: remove o prefixo "# " dos comentários
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in bloco.splitlines())
        nb.cells.append(nbf.v4.new_markdown_cell(md))
    else:
        nb.cells.append(nbf.v4.new_code_cell(bloco))

nbf.write(nb, "cnn_demo.ipynb")
print("Gerado cnn_demo.ipynb com", len(nb.cells), "células")
