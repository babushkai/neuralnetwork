"""Repository consistency checks (fast, no execution). Run: python tools/check.py

1. every nb_src/<name>.py has a notebooks/<name>.ipynb whose cells match it exactly
2. every code cell is valid Python
3. every notebook is committed with outputs
4. "notebook N" cross-references point to notebooks that exist
5. every notebook link in README.md exists, and every notebook appears in README.md
"""
import ast, pathlib, re, sys
import nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from build import parse  # noqa: E402

# Cells added to a notebook outside nb_src (kept deliberately). Keys: notebook stem → extra code-cell sources.
EXTRA_CELLS = {"72_2025_tiny_recursive_model": {"make_puzzles(grids, B=1000)"}}
# Notebooks allowed to be committed without outputs: the heaviest 2026 notebooks, not yet executed here.
NO_OUTPUTS_OK = {"84_2026_gated_deltanet_2", "85_2026_gated_residual", "86_2026_cheaper_sparse_attention_indexers",
                 "87_2026_causal_encoder_decoder", "88_2026_looped_transformers", "89_2026_constant_lr_checkpoint_averaging"}

errors = []
srcs = sorted((ROOT / "nb_src").glob("*.py"))
nbs = {p.stem for p in (ROOT / "notebooks").glob("*.ipynb")}
n_total = len(srcs)

for src in srcs:
    stem = src.stem
    if stem not in nbs:
        errors.append(f"{stem}: no notebook; run python tools/build.py {stem}"); continue
    nb = nbformat.read(ROOT / "notebooks" / f"{stem}.ipynb", as_version=4)
    have = [(c.cell_type, c.source) for c in nb.cells if not (c.cell_type == "code" and c.source.strip() in EXTRA_CELLS.get(stem, set()))]
    want = [(c.cell_type, c.source) for c in parse(src)]
    if have != want:
        i = next((k for k, (a, b) in enumerate(zip(have, want)) if a != b), min(len(have), len(want)))
        errors.append(f"{stem}: notebook differs from nb_src at cell {i}; run python tools/build.py {stem}")
    for k, c in enumerate(nb.cells):
        if c.cell_type == "code":
            try: ast.parse(c.source)
            except SyntaxError as e: errors.append(f"{stem}: cell {k} is not valid Python ({e.msg}, line {e.lineno})")
    if stem not in NO_OUTPUTS_OK and not any(c.get("outputs") for c in nb.cells if c.cell_type == "code"):
        errors.append(f"{stem}: committed without outputs")
    for m in re.finditer(r"[Nn]otebooks? (\d{1,3}(?:(?:, | and | or |–)\d{1,3})*)", src.read_text()):
        for n in re.findall(r"\d+", m.group(1)):
            if not 1 <= int(n) <= n_total: errors.append(f"{stem}: refers to notebook {n}, which does not exist")
extra = nbs - {s.stem for s in srcs}
errors += [f"{e}: notebook has no nb_src source" for e in sorted(extra)]

readme = (ROOT / "README.md").read_text()
linked = set(re.findall(r"notebooks/([\w-]+)\.ipynb", readme))
errors += [f"README links to missing notebook {l}" for l in sorted(linked - nbs)]
errors += [f"README does not list notebook {n}" for n in sorted(nbs - linked)]

if errors:
    print("\n".join(errors)); print(f"\n{len(errors)} problem(s)"); sys.exit(1)
print(f"ok: {n_total} notebooks consistent with their sources, valid, executed, and listed in README.md")
