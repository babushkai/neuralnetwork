"""A small layered-diagram renderer used by tools/diagrams.py (matplotlib only).

A diagram is a list of rows, drawn bottom (inputs) to top (outputs). Each row is a list of node strings:

    "label"                plain node; its id is the label itself
    "label #id"            node with a short id, for use in edges
    "label !kind"          kind: in, op, mem, gate, out, loss, norm, circle, note (defaults: first row in, last row out, else op)
    "_"                    an invisible spacer

Labels may contain "\\n" and matplotlib mathtext ($...$).

Edges are a ";"-separated string. With edges=None, every node is joined to every node of the next row.
    a>b         straight arrow                 a.>b   dashed (training signal, no gradient, copy)
    a)>b        arrow routed around the right  a(>b   ... around the left      (prefix "." for dashed)
    a@          self-loop (recurrence)
    a-b         line without an arrowhead
    autoI-J     join rows I..J fully, as with edges=None
    any edge may end with ":label"
"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

PAL = {  # fill, edge
    "in":     ("#eef1f4", "#5f6b76"),
    "op":     ("#e6f0fb", "#2f6db5"),
    "mem":    ("#fff2df", "#c97a12"),
    "gate":   ("#f3eafa", "#7b44a8"),
    "out":    ("#e7f5e8", "#2e8540"),
    "loss":   ("#fdecec", "#c0392b"),
    "norm":   ("#f6f6f6", "#8a8a8a"),
    "circle": ("#ffffff", "#444444"),
    "note":   ("#ffffff", "#ffffff"),
}
FS, GAP_X, GAP_Y, PAD_X, PAD_Y, DPI = 10, 0.32, 0.46, 0.13, 0.09, 110


class Node:
    def __init__(self, spec, kind):
        m = re.match(r"^(.*?)((?:\s+[#!]\S+)*)\s*$", spec, re.S)
        self.label, tags = m.group(1).replace(" // ", "\n"), m.group(2).split()
        self.id, self.kind = self.label, kind
        for t in tags:
            if t[0] == "#": self.id = t[1:]
            else: self.kind = t[1:]
        self.spacer = self.label == "_"


def render(rows, edges=None, title="", caption="", path=None):
    fig = plt.figure(dpi=DPI)
    r = fig.canvas.get_renderer()
    grid, nodes = [], {}
    for i, row in enumerate(rows):
        default = "in" if i == 0 else "out" if i == len(rows) - 1 else "op"
        line = []
        for s in row:
            n = Node(s, default)
            if n.spacer:
                n.w, n.h = 0.6, 0.3
            else:
                t = fig.text(0, 0, n.label, fontsize=FS if n.kind != "note" else FS - 1, ha="center", va="center")
                bb = t.get_window_extent(r); t.remove()
                n.w, n.h = bb.width / DPI + 2 * PAD_X, bb.height / DPI + 2 * PAD_Y
                if n.kind == "circle": n.w = n.h = max(n.w, n.h, 0.32)
            line.append(n); nodes.setdefault(n.id, n)
        grid.append(line)
    # layout: rows centred, bottom to top
    y = 0.0
    for i, line in enumerate(grid):
        hmax = max(n.h for n in line)
        if i: y += hprev / 2 + GAP_Y + hmax / 2
        total = sum(n.w for n in line) + GAP_X * (len(line) - 1)
        x = -total / 2
        for n in line:
            n.x, n.y = x + n.w / 2, y; x += n.w + GAP_X
        hprev = hmax
    # edges
    E = []
    if edges is None: edges = f"auto0-{len(grid) - 1}"
    for tok in [t.strip() for t in edges.split(";") if t.strip()]:
        m = re.match(r"^auto(\d+)-(\d+)$", tok)
        if m:
            for i in range(int(m.group(1)), int(m.group(2))):
                E += [(a.id, b.id, ">", False, "") for a in grid[i] for b in grid[i + 1] if not a.spacer and not b.spacer]
            continue
        m = re.match(r"^(.+?)(\.?)(\)>|\(>|>|@|-)(.*?)(?::(.*))?$", tok, re.S)
        if not m: raise ValueError(f"bad edge {tok!r}")
        a, dashed, kind, b, lab = m.group(1).strip(), bool(m.group(2)), m.group(3), m.group(4).strip(), (m.group(5) or "").strip()
        if kind == "-" and not b: raise ValueError(f"bad edge {tok!r}")
        for v in (a, b) if b else (a,):
            if v not in nodes: raise KeyError(f"edge {tok!r}: unknown node {v!r}; ids: {list(nodes)}")
        E.append((a, b, kind, dashed, lab))

    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); fig.patch.set_facecolor("white")
    X = [v for n in nodes.values() if not n.spacer for v in (n.x - n.w / 2, n.x + n.w / 2)]
    Y = [v for n in nodes.values() if not n.spacer for v in (n.y - n.h / 2, n.y + n.h / 2)]

    def measure(txt, fs, **kw):
        t = fig.text(0, 0, txt, fontsize=fs, **kw); bb = t.get_window_extent(r); t.remove(); return bb.width / DPI, bb.height / DPI

    for n in nodes.values():
        if n.spacer: continue
        fc, ec = PAL.get(n.kind, PAL["op"])
        if n.kind == "circle":
            ax.add_patch(Circle((n.x, n.y), n.w / 2, fc=fc, ec=ec, lw=1.2, zorder=2))
        elif n.kind != "note":
            ax.add_patch(FancyBboxPatch((n.x - n.w / 2, n.y - n.h / 2), n.w, n.h, boxstyle="round,pad=0,rounding_size=0.07",
                                        fc=fc, ec=ec, lw=1.2, zorder=2))
        ax.text(n.x, n.y, n.label, fontsize=FS if n.kind != "note" else FS - 1, ha="center", va="center", zorder=3,
                color="#222" if n.kind != "note" else "#555", style="normal" if n.kind != "note" else "italic")

    def port(n, toward):
        """Point on n's border in the direction of point `toward`."""
        dx, dy = toward[0] - n.x, toward[1] - n.y
        if n.kind == "circle":
            d = (dx * dx + dy * dy) ** 0.5 or 1; return n.x + dx / d * n.w / 2, n.y + dy / d * n.w / 2
        sx = (n.w / 2) / abs(dx) if dx else 1e9; sy = (n.h / 2) / abs(dy) if dy else 1e9
        s = min(sx, sy); return n.x + dx * s, n.y + dy * s

    style = lambda d: dict(color="#444" if not d else "#888", lw=1.1, ls="--" if d else "-")
    def arrow(p, q, dashed, head=True):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>" if head else "-", mutation_scale=11, shrinkA=0, shrinkB=0, **style(dashed)), zorder=1)
    def label(x, y, s, ha="center"):
        s = s.replace(" // ", "\n")
        w, _ = measure(s, FS - 2)
        X.extend([x - w / 2, x + w / 2] if ha == "center" else [x, x + w] if ha == "left" else [x - w, x])
        ax.text(x, y, s, fontsize=FS - 2, ha=ha, va="center", color="#555", style="italic", zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))

    lanes = []
    for a, b, kind, dashed, lab in E:
        A = nodes[a]
        if kind == "@":
            x0 = A.x + A.w / 2; off = 0.32
            pts = [(x0, A.y + A.h / 4), (x0 + off, A.y + A.h / 4), (x0 + off, A.y - A.h / 4)]
            ax.plot(*zip(*pts), **style(dashed), zorder=1); arrow(pts[-1], (x0, A.y - A.h / 4), dashed); X.append(x0 + off)
            if lab: label(x0 + off + 0.05, A.y, lab, ha="left")
            continue
        B = nodes[b]
        if kind in (")>", "(>"):
            sgn = 1 if kind == ")>" else -1
            ex = lambda n: n.x + sgn * n.w / 2
            lo, hi = sorted([A.y, B.y])
            between = [n for n in nodes.values() if lo - 1e-6 <= n.y <= hi + 1e-6 and not n.spacer]
            far = (max if sgn > 0 else min)(ex(n) for n in between) + sgn * 0.25
            while any(sd == sgn and l0 <= hi + 1e-6 and lo <= h0 + 1e-6 and abs(fx - far) < 0.12 for sd, l0, h0, fx in lanes):
                far += sgn * 0.2
            lanes.append((sgn, lo, hi, far)); X.append(far)
            p0, p3 = (ex(A), A.y), (ex(B), B.y)
            ax.plot([p0[0], far, far], [p0[1], p0[1], p3[1]], **style(dashed), zorder=1); arrow((far, p3[1]), p3, dashed)
            if lab: label(far + sgn * 0.06, (A.y + B.y) / 2, lab, ha="left" if sgn > 0 else "right")
            continue
        p, q = port(A, (B.x, B.y)), port(B, (A.x, A.y))
        arrow(p, q, dashed, head=kind != "-")
        if lab: label((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, lab)

    xmin, xmax, ybot, ytop = min(X), max(X), min(Y), max(Y)
    cx = (min(n.x - n.w / 2 for n in nodes.values()) + max(n.x + n.w / 2 for n in nodes.values())) / 2
    if title:
        w, h = measure(title, FS + 2, weight="bold"); ytop += 0.18 + h / 2
        ax.text(cx, ytop, title, fontsize=FS + 2, ha="center", va="center", weight="bold", color="#222")
        xmin, xmax, ytop = min(xmin, cx - w / 2), max(xmax, cx + w / 2), ytop + h / 2
    caption = caption.replace(" // ", "\n")
    if caption:
        w, h = measure(caption, FS - 1); ybot -= 0.16 + h / 2
        ax.text(cx, ybot, caption, fontsize=FS - 1, ha="center", va="center", color="#555")
        xmin, xmax, ybot = min(xmin, cx - w / 2), max(xmax, cx + w / 2), ybot - h / 2
    m = 0.18
    fig.set_size_inches(xmax - xmin + 2 * m, ytop - ybot + 2 * m)
    ax.set_xlim(xmin - m, xmax + m); ax.set_ylim(ybot - m, ytop + m)
    if path: fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
