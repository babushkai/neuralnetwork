"""Convert nb_src/*.py (cell-marker format) into notebooks/*.ipynb.
Markers: '#%% md' starts a markdown cell, '#%%' starts a code cell.
Tokens like {{SETUP}} are expanded from snippets/. Outputs of unchanged code cells are preserved."""
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
    out = OUT / (path.stem + ".ipynb")
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
    if out.exists():                                     # reuse unchanged cells as-is (keeps outputs, ids, metadata)
        old = nbformat.read(out, as_version=4)
        prev = {}
        for c in old.cells: prev.setdefault((c.cell_type, c.source), c)
        cells = [prev.get((c.cell_type, c.source), c) for c in cells]
        meta = {**meta, **old.metadata}
    nb = nbformat.v4.new_notebook(cells=cells, metadata=meta)
    nbformat.write(nb, out)

OUT.mkdir(parents=True, exist_ok=True)
targets = sys.argv[1:] or [p.stem for p in sorted(SRC.glob("*.py"))]
for t in targets:
    convert(SRC / f"{t}.py"); print("built", t)
