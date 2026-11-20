"""Print the text outputs of executed notebooks (quick sanity check)."""
import sys, nbformat
for path in sys.argv[1:]:
    print(f"===== {path}")
    for c in nbformat.read(path, 4).cells:
        for o in c.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain", "")
            if t and "Figure" not in t: print(t.rstrip()[-600:])
