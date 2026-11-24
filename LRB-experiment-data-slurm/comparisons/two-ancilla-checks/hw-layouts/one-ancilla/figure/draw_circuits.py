"""Write syndrome-extraction circuits as quantikz, in the paper's notation.

Rounds: (a) the LRB generator's one-ancilla round, (b) the same 12 gates packed into
5 gate moments (rounds.py PROD5), (c) SQ14 (rounds.py SQ14). Moments come from the
generator's own ASAP scheduler (LRBCircuitGenerator.schedule_moments), so the
columns are the gate moments the noise model sees. Inside one moment, gates whose
vertical lines would overlap go into adjacent sub-columns; dashed lines separate
moments. Notation as the paper (version-sim appendix): a dot joined to a box X
(X^{-1}) is a SUM gate (its inverse) with the control on the dot.
Usage: draw_circuits.py  -> writes circuits.tex next to this file.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))   # two-ancilla-checks
sys.path.insert(0, str(HERE.parent))                 # one-ancilla
import noise_bench as NB  # noqa: E402
import protocol_search as PS  # noqa: E402
import rounds as R  # noqa: E402

NAMES = ["q_0", "q_1", "q_2", "q_3", "q_4", "a_1", "a_2", "a_3", "a_4"]
READ = {5: r"\textcolor{ZText}{$G_1$}", 6: r"\textcolor{ZText}{$G_2$}",
        7: r"\textcolor{XText}{$G_3$}", 8: r"\textcolor{XText}{$G_4$}"}


def gate_moments(circ):
    out = []
    for collapse, ops in PS.GEN.schedule_moments(circ):
        if collapse:
            continue
        out.append([(o.gate_name, tuple(PS.GEN._operation_qudits(o))) for o in ops])
    return out


def subcolumns(moment):
    """Split one moment into columns whose vertical spans do not overlap."""
    cols = []
    for g in moment:
        lo, hi = min(g[1]), max(g[1])
        for col in cols:
            if all(hi < min(h[1]) or lo > max(h[1]) for h in col):
                col.append(g)
                break
        else:
            cols.append([g])
    return cols


def box(name, style=None):
    label = {"CNOT": "X", "CNOT_INV": "X^{-1}", "H": "H", "H_INV": "H^{-1}"}[name]
    opt = f"[style={{{style}}}]" if style else ""
    return rf"\gate{opt}{{{label}}}"


def style_of(c, t):
    if c < 5 and t < 5:
        return "fill=WinTint,draw=black!60"          # data CNOT inside the round
    if c >= 5 and t >= 5:
        return "fill=Ztint,draw=ZRim" if c in (5, 6) else "fill=Xtint,draw=XRim"   # transfer
    return None


def quantikz(moments):
    rows = [[rf"\lstick{{${NAMES[w]}$}}" if w < 5 else rf"\lstick{{${NAMES[w]}\colon\ket{{0}}$}}"]
            for w in range(9)]
    for mi, moment in enumerate(moments):
        cols = subcolumns(moment)
        for ci, col in enumerate(cols):
            cell = [r"\qw"] * 9
            for name, w in col:
                if len(w) == 1:
                    cell[w[0]] = box(name)
                else:
                    c, t = w
                    cell[c] = rf"\ctrl{{{t - c}}}"
                    cell[t] = box(name, style_of(c, t))
            if ci == len(cols) - 1 and mi < len(moments) - 1:
                cell[0] += r" \slice[style={black!35}]{}"
            if ci == 0:
                cell[0] = cell[0].replace(r"\qw", r"\qw", 1)
            for w in range(9):
                rows[w].append(cell[w])
    for w in range(9):
        rows[w].append(r"\meter{} \rstick{" + READ[w] + "}" if w >= 5 else r"\qw")
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return ("\\begin{quantikz}[row sep={0.62cm,between origins}, column sep=0.22cm]\n"
            + body + "\n\\end{quantikz}")


def stats(moments):
    two = sum(1 for m in moments for g in m if len(g[1]) == 2)
    return two, len(moments)


if __name__ == "__main__":
    prod, _ = NB.production_round("folded_qutrit", "si1000")
    rounds = [
        ("a", "Current one-ancilla round, as in the LRB runs", gate_moments(prod)),
        ("b", "Same layout and gates, packed into 5 gate moments", gate_moments(PS.to_circuit(
            [op[:2] if op[0] == "m" else op for op in R.PROD5], 9))),
        ("c", "SQ14 (square lattice)", gate_moments(PS.to_circuit(
            [op[:2] if op[0] == "m" else op for op in R.SQ14], 9))),
    ]
    parts = []
    for letter, title, moms in rounds:
        n2, nm = stats(moms)
        parts.append(rf"{{\small\bfseries({letter})}}\hspace{{0.5em}}{{\small {title}: "
                     rf"{n2} two-qutrit gates, {nm} gate moments}}\\[4pt]" + "\n" + quantikz(moms))
    doc = r"""\documentclass[border=6pt,varwidth=40cm]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{quantikz}
\definecolor{Zcol}{HTML}{FF1744}
\definecolor{Xcol}{HTML}{00B7FF}
\colorlet{Xtint}{Xcol!26}
\colorlet{Ztint}{Zcol!17}
\colorlet{XRim}{Xcol}
\colorlet{ZRim}{Zcol}
\colorlet{XText}{Xcol!80!black}
\colorlet{ZText}{Zcol!85!black}
\colorlet{WinTint}{black!14}
\begin{document}
""" + "\n\n\\vspace{14pt}\n".join(parts) + r"""

\vspace{10pt}
{\footnotesize A dot joined to a box $X$ ($X^{-1}$) is a SUM gate (its inverse), control on the dot.
Dashed lines separate gate moments of the LRB generator's scheduler; every round starts with
one reset layer and ends with one readout layer.
Grey boxes: data CNOTs applied and undone inside the round.
Red / blue boxes: ancilla--ancilla gates that move partial $Z$ / $X$ sums between $a_1, a_2$ / $a_3, a_4$.}
\end{document}
"""
    (HERE / "circuits.tex").write_text(doc)
    for letter, title, moms in rounds:
        print(letter, title, stats(moms))
