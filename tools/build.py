"""Convert nb_src/*.py (cell-marker format) into notebooks/*.ipynb.
Markers: '#%% md' starts a markdown cell, '#%%' starts a code cell.
Tokens {{SETUP}}, {{DIGITS}}, {{CORPUS}} are expanded from snippets/."""
import sys, pathlib, nbformat
HERE = pathlib.Path(__file__).parent
SRC = HERE.parent / "nb_src"
OUT = HERE.parent / "notebooks"
SNIP = {p.stem.upper(): p.read_text().rstrip() for p in (HERE / "snippets").glob("*.py")}

def convert(path):
    cells, kind, buf = [], None, []
    def flush():
        if kind is None: return
        text = "\n".join(buf).strip("\n")
        for k, v in SNIP.items():
            text = text.replace("{{%s}}" % k, v)
        if text.strip():
            cells.append(nbformat.v4.new_markdown_cell(text) if kind == "md" else nbformat.v4.new_code_cell(text))
    for line in path.read_text().splitlines():
        s = line.strip()
        if s == "#%% md" or s == "#%%":
            flush(); kind = "md" if s.endswith("md") else "code"; buf = []
        else:
            buf.append(line)
    flush()
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.write(nb, OUT / (path.stem + ".ipynb"))

OUT.mkdir(parents=True, exist_ok=True)
targets = sys.argv[1:] or [p.stem for p in sorted(SRC.glob("*.py"))]
for t in targets:
    convert(SRC / f"{t}.py"); print("built", t)
