"""The "Dorito chip": the distance-5 layout with the smaller codes it contains outlined.

The layout of distance d - 1 is the upper-left corner of the layout of distance d (gen_layout.py
checks this), so a distance-5 chip also runs the distance-3 code and the paper's distance-2 code on
a subset of its qubits and couplers. This script draws the distance-5 layout and outlines the
triangles that hold the distance-2 and distance-3 codes.

Writes figures/dorito-chip.tex and dorito-chip.tex (standalone wrapper); build with
`latexmk -pdf dorito-chip.tex`.
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_layout as g  # noqa: E402

D_CHIP = 5
SUBCODES = ((3, "HCol", "d = 3"), (2, "Zcol", "d = 2, the paper's code"))   # larger one drawn first


def outline(L: int, m: float) -> str:
    """The triangle x <= y <= L (picture coordinates, y down) pushed out by m, apex cut flat."""
    r2 = math.sqrt(2)
    pts = [(-m, -m), (-m, L + m), (L + m * (1 + r2), L + m), (m * (r2 - 1), -m)]
    return " -- ".join("(%s,%s)" % (g.f(x), g.f(-y)) for x, y in pts) + " -- cycle"


def main() -> None:
    lay = g.build(D_CHIP)
    body = g.tikz_folded(lay, 0.95, (r"\scriptsize ", r"\tiny "), lw=0.75)
    lines = body.split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("\\begin{tikzpicture}"))
    head, begin, rest = "\n".join(lines[:i]), lines[i], "\n".join(lines[i + 1:])
    draw = []
    for d, color, label in SUBCODES:
        L = 2 * d - 2
        m = 0.42 if d == 2 else 0.5
        draw.append(r"\fill[%s,fill opacity=0.07,rounded corners=6pt] %s;" % (color, outline(L, m)))
        draw.append(r"\draw[%s,line width=1.3pt,dash pattern=on 5pt off 2.5pt,rounded corners=6pt] %s;"
                    % (color, outline(L, m)))
        # label in the empty half above the fold, beside the far corner of this code's triangle
        draw.append(r"\node[text=%s,font=\small,anchor=west] at (%s,%s) {%s};"
                    % (color, g.f(L + 0.8), g.f(-(L - 0.8)), label))
    L5 = 2 * D_CHIP - 2
    draw.append(r"\node[font=\small,anchor=west,text=black!70] at (%s,%s) {d = 5};"
                % (g.f(L5 + 0.55), g.f(-L5 - 0.1)))
    tikz = "\n".join([head, begin] + draw + [rest])
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    with open(os.path.join(HERE, "figures", "dorito-chip.tex"), "w") as fh:
        fh.write(tikz)
    with open(os.path.join(HERE, "dorito-chip.tex"), "w") as fh:
        fh.write(r"""\documentclass[border=6pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\input{layout-style}
\definecolor{HCol}{HTML}{D35400}
\definecolor{Zcol}{HTML}{FF1744}
\begin{document}
\input{figures/dorito-chip}
\end{document}
""")
    print("wrote figures/dorito-chip.tex and dorito-chip.tex")


if __name__ == "__main__":
    main()
