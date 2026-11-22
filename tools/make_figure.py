"""Build assets/hero.png from plots saved inside executed notebooks.

Usage: python tools/make_figure.py            (uses the PANELS list below)
       python tools/make_figure.py --list NB  (print the index of every image output in notebook NB, e.g. 29)
"""
import base64, io, sys, pathlib
import nbformat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = pathlib.Path(__file__).resolve().parent.parent
# (notebook number, image index within that notebook, caption)
PANELS = [
    ("06", 0, "1982 · Hopfield network: recall from corrupted patterns"),
    ("14", 2, "2001 · Echo state network: free-running chaotic prediction"),
    ("27", 1, "2014 · Neural Turing Machine: learned write/read pattern"),
    ("29", 1, "2015 · ResNet: gradient reaching the first layers"),
    ("36", 0, "2016 · YOLO: single-pass detection on unseen scenes"),
    ("46", 0, "2020 · In-context learning vs. least squares"),
    ("56", 0, "2022 · Latent diffusion with classifier-free guidance"),
    ("82", 2, "2026 · Attention residuals: attention over depth"),
]

def images(num):
    path = next((ROOT / "notebooks").glob(f"{num}_*.ipynb"))
    out = []
    for c in nbformat.read(path, 4).cells:
        for o in c.get("outputs", []):
            png = o.get("data", {}).get("image/png")
            if png: out.append(mpimg.imread(io.BytesIO(base64.b64decode(png)), format="png"))
    return path.stem, out

if __name__ == "__main__":
    if sys.argv[1:2] == ["--list"]:
        stem, ims = images(sys.argv[2])
        for i, im in enumerate(ims): print(stem, i, im.shape)
        sys.exit()
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), facecolor="white")
    for ax, (num, idx, caption) in zip(axes.flat, PANELS):
        stem, ims = images(num)
        ax.imshow(ims[idx]); ax.set_title(caption, fontsize=12, pad=8); ax.axis("off")
    fig.text(0.5, 0.01, "All panels are outputs of notebooks in this repository, produced on a laptop.", ha="center", fontsize=11, color="#555")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out = ROOT / "assets" / "hero.png"
    fig.savefig(out, dpi=80, facecolor="white"); print("wrote", out)
