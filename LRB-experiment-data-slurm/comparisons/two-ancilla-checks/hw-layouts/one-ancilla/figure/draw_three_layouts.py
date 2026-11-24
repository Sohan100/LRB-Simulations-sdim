"""Write three-layouts.tex: the three layouts of the [[5,1,2]]_3 folded code, their encoders
and their check circuits, and check every drawn circuit in sdim (three-layouts-verify.out).

Review document, not a paper figure. The layout panels are copied (read only) from the
figure session's paper-rework/a1-layout-figure/ancilla-layouts-B.tex into
three-layouts-panels.tex.

Layouts and their circuits (internal profile names only where they point at code):
  (a) terminal check on IQM Garnet: no ancillas and no check rounds; encoder of
      BRIEF-layoutB.md Sec. 2 (H^{-1} on q0, q2, q3; CNOT q0->q1, CNOT q2->q4; CNOT q3->q0,
      CNOT_INV q1->q2); check circuit = the production terminal X readout of all five data
      qutrits (terminal_const0_measurement of profile folded_qutrit);
  (b) one-ancilla layout: the five-moment round (profile folded_qutrit_prod5b) and, for
      reference, the seven-moment round of the published runs (profile folded_qutrit);
      encoder hw-layouts/encoders/one_ancilla_encoder_pick.out, first depth-5 line (8 gates
      through a1 and a2, which end in |0>), with H^{-1} as the one-qutrit layer (H^{-1}|0> =
      H|0>), read from that file;
  (c) two-ancilla layout: round and encoder of profile folded_qutrit_layout_b.
Every circuit is scheduled by the LRB generator's own scheduler
(LRBCircuitGenerator.schedule_moments), so each dashed column group is one gate moment the
noise model sees; gates of one moment whose vertical lines would overlap are drawn in
adjacent sub-columns.

Checks (three-layouts-verify.out; the script stops on the first failure):
  - every two-qutrit gate of every encoder and round is on a coupler of its layout
    (for (a): the Garnet couplers between the five data transmons);
  - gate and moment counts equal the numbers printed under the panels;
  - noiseless sdim, 300 shots: (a) encoder then terminal X readout; (b), (c) encoder, 3
    rounds, terminal X readout. Every check readout is 0, G3 and G4 from the X readout are
    0, the logical X0 X3 is 0 (the state is |+_L>), and the ancillas of encoder (b) read 0.
Run with the one-ancilla-prod5b worktree's code (the only checkout with all the profiles):
    cd <worktree> && PYTHONNOUSERSITE=1 .venv-sdim140/bin/python <this file>
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = Path("/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/"
                "one-ancilla-prod5b")
sys.path.insert(0, str(WORKTREE / "src"))

from sdim.circuit import Circuit  # noqa: E402
from sdim.program import Program  # noqa: E402
from lrb.circuit_generator import LRBCircuitGenerator  # noqa: E402
from lrb.code_simulation_profiles import CodeSimulationProfileRegistry as Reg  # noqa: E402

PICK = HERE.parents[1] / "encoders" / "one_ancilla_encoder_pick.out"
SHOTS = 300

ZT, XT = r"\textcolor{ZText}", r"\textcolor{XText}"
G = {1: ZT + "{$G_1$}", 2: ZT + "{$G_2$}", 3: XT + "{$G_3$}", 4: XT + "{$G_4$}"}
GM = {1: ZT + "{G_1}", 2: ZT + "{G_2}", 3: XT + "{G_3}", 4: XT + "{G_4}"}   # math mode


def label_text(v):
    return G[v] if isinstance(v, int) else f"${v}$"


def label_math(v):
    return GM[v] if isinstance(v, int) else v


DATA = ["q_0", "q_1", "q_2", "q_3", "q_4"]
ONE_ANCILLA = {"names": DATA + ["a_1", "a_2", "a_3", "a_4"],
               "reads": {5: [1], 6: [2], 7: [3], 8: [4]},
               "ztype": {5, 6}}
TWO_ANCILLA = {"names": DATA + ["N", "E"],
               "reads": {5: [2, 4], 6: [1, 3]},
               "ztype": set()}
TERMINAL_READS = {q: [f"x_{q}"] for q in range(5)}

COUPLERS = {   # as drawn in three-layouts-panels.tex
    # Garnet couplers between the data transmons QB1, QB4, QB5, QB2, QB6 = q0..q4:
    # QB1-QB4, QB1-QB2, QB4-QB5, QB2-QB5, QB5-QB6
    "a": {(0, 1), (0, 3), (1, 2), (3, 2), (2, 4)},
    "b": {(8, 3), (8, 1), (8, 4), (5, 3), (5, 1), (5, 0), (6, 4), (6, 1), (6, 2),
          (7, 0), (7, 1), (7, 2), (2, 3)},
    "c": {(0, 1), (1, 2), (1, 4), (2, 3), (5, 4), (5, 2), (6, 2), (0, 3)},
}
LABELS = {"CNOT": "X", "CNOT_INV": "X^{-1}", "H": "H", "H_INV": "H^{-1}"}
GARNET_ENCODER = [("H_INV", 0), ("H_INV", 2), ("H_INV", 3), ("CNOT", 0, 1), ("CNOT", 2, 4),
                  ("CNOT", 3, 0), ("CNOT_INV", 1, 2)]


def generator(code_name: str) -> LRBCircuitGenerator:
    return LRBCircuitGenerator(code_definition=Reg.resolve_code_profile(code_name).code_definition,
                               noise_model="si1000")


GEN_ONE, GEN_ONE_PUB, GEN_TWO = (generator(n) for n in
                                 ("folded_qutrit_prod5b", "folded_qutrit", "folded_qutrit_layout_b"))


def from_ops(ops, nwires=9) -> Circuit:
    c = Circuit(dimension=3, num_qudits=nwires)
    for op in ops:
        c.add_gate(op[0], *op[1:])
    return c


def one_ancilla_encoder() -> Circuit:
    for line in PICK.read_text().splitlines():
        m = re.match(r"\s*depth 5\s+\|\+> on (\(.*?\))\s+ancillas end in \|0>\s+gates (\[.*\])"
                     r"\s+overlap with paper \|\+_L>\|0,0>: d=3 ([0-9.]+), d=2 ([0-9.]+)", line)
        if m and float(m.group(3)) > 1 - 1e-9 and float(m.group(4)) > 1 - 1e-9:
            plus, gates = ast.literal_eval(m.group(1)), ast.literal_eval(m.group(2))
            return from_ops([("H_INV", q) for q in plus] + [tuple(g) for g in gates])
    raise SystemExit(f"no exact depth-5 encoder in {PICK}")


def ops_of(gen, circ):
    return [(o.gate_name, *gen._operation_qudits(o)) for o in circ.operations]


# ---------------------------------------------------------------- drawing
def gate_style(c, t, ztype_ancillas, round_):
    if not round_:
        return None
    if c < 5 and t < 5:
        return "fill=WinTint,draw=black!60"
    if c >= 5 and t >= 5:
        return "fill=Ztint,draw=ZRim" if c in ztype_ancillas else "fill=Xtint,draw=XRim"
    return None


def subcolumns(gates):
    cols = []
    for g in gates:
        lo, hi = min(g[1:]), max(g[1:])
        for col in cols:
            if all(hi < min(h[1:]) or lo > max(h[1:]) for h in col):
                col.append(g)
                break
        else:
            cols.append([g])
    return cols


def quantikz(gen, circ, names, reads=None, ztype=frozenset(), round_=True, encoder_anc=()):
    """Draw circ moment by moment. reads: wire -> stabilizer numbers in readout order."""
    reads = reads or {}
    n = len(names)
    moments = gen.schedule_moments(circ)
    if moments and moments[0][0] and all(o.gate_name == "RESET" for o in moments[0][1]):
        moments = moments[1:]                       # the reset layer is the |0> on the left
    groups = []                                    # (kind, list of columns of cells)
    seen = {w: 0 for w in range(n)}
    for collapse, ops in moments:
        gates = [(o.gate_name, *gen._operation_qudits(o)) for o in ops]
        if collapse:
            assert all(g[0] == "M" for g in gates), gates
            cell = [r"\qw"] * n
            for _, w in gates:
                cell[w] = r"\meter{}"
            groups.append(("read", [cell], gates))
            continue
        cols = []
        for col in subcolumns(gates):
            cell = [r"\qw"] * n
            for g in col:
                if len(g) == 2:
                    cell[g[1]] = rf"\gate{{{LABELS[g[0]]}}}"
                else:
                    name, c, t = g
                    cell[c] = rf"\ctrl{{{t - c}}}"
                    style = gate_style(c, t, ztype, round_)
                    opt = f"[style={{{style}}}]" if style else ""
                    cell[t] = rf"\gate{opt}{{{LABELS[name]}}}"
            cols.append(cell)
        groups.append(("gate", cols, gates))
    rows = [[rf"\lstick{{${names[w]}$}}" if round_ and w < 5 else
             rf"\lstick{{${names[w]}\colon\ket{{0}}$}}"] for w in range(n)]
    final_reads = set()
    for gi, (kind, cols, gates) in enumerate(groups):
        last = gi == len(groups) - 1
        if kind == "read":
            for _, w in gates:
                k = seen[w]
                seen[w] += 1
                if last:
                    final_reads.add((w, label_text(reads[w][k])))
                else:      # mid-round readout: label inline after the meter, wire continues
                    if len(cols) == 1:
                        cols.append([r"\qw"] * n)
                    cols[1][w] = rf"\push{{\,{label_math(reads[w][k])}\,}}"
        for ci, cell in enumerate(cols):
            before_final_read = gi == len(groups) - 2 and groups[-1][0] == "read"
            if ci == len(cols) - 1 and not last and not before_final_read:
                cell[0] += r" \slice[style={black!35}]{}"
            for w in range(n):
                rows[w].append(cell[w])
    labels = dict(final_reads)
    for w in range(n):
        if w in labels:
            rows[w].append(rf"\rstick{{{labels[w]}}}")
        elif w in encoder_anc:
            rows[w].append(r"\qw \rstick{$\ket{0}$}")
        else:
            rows[w].append(r"\qw")
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return ("\\begin{quantikz}[row sep={0.62cm,between origins}, column sep=0.22cm]\n"
            + body + "\n\\end{quantikz}")


# ---------------------------------------------------------------- checks
LOG = []


def say(msg):
    LOG.append(msg)
    print(msg, flush=True)


def counts(gen, circ):
    moments = gen.schedule_moments(circ)
    two = sum(1 for c, ops in moments for o in ops if len(gen._operation_qudits(o)) == 2)
    gate_moments = sum(1 for c, _ in moments if not c)
    return two, gate_moments


def depth2(gen, circ):
    return sum(1 for c, ops in gen.schedule_moments(circ)
               if any(len(gen._operation_qudits(o)) == 2 for o in ops))


def on_couplers(panel, gen, circ, what):
    bad = [g for g in ops_of(gen, circ) if len(g) == 3
           and (g[1], g[2]) not in COUPLERS[panel] and (g[2], g[1]) not in COUPLERS[panel]]
    assert not bad, (panel, what, bad)
    say(f"  ({panel}) {what}: every two-qutrit gate is on a coupler of the layout")


def simulate(encoder, round_circ, nwires, z_wires_postselect=(), rounds=3):
    """Encoder, rounds, X-basis data readout. Returns (kept, failures)."""
    c = Circuit(dimension=3, num_qudits=nwires) + encoder
    for _ in range(rounds):
        c = c + round_circ
    for q in range(5):
        c.add_gate("H_INV", q)
        c.add_gate("M", q)
    counter, reads = {}, []          # (wire, record index, round, is_encoder)
    rnd = -1
    for op in c.operations:
        if op.gate_name in ("M", "RESET"):
            w = int(op.qudit_index)
            if op.gate_name == "M":
                reads.append((w, counter.get(w, 0)))
            counter[w] = counter.get(w, 0) + 1
    m = Program(c).simulate(shots=SHOTS)[0]
    val = lambda w, k, s: int(m[w][k][s].measurement_value) % 3   # noqa: E731
    data_reads = reads[-5:]
    check_reads = reads[:-5]
    per_round = len([1 for op in round_circ.operations if op.gate_name == "M"])
    enc_reads = len(check_reads) - rounds * per_round
    kept = fails = 0
    for s in range(SHOTS):
        vals = [val(w, k, s) for w, k in check_reads]
        first = check_reads[enc_reads:enc_reads + per_round]
        if z_wires_postselect and any(val(w, k, s) for w, k in first if w in z_wires_postselect):
            continue
        kept += 1
        a = [val(w, k, s) for w, k in data_reads]
        ok = (all(v == 0 for v in vals)
              and (a[0] + a[1] - a[2]) % 3 == 0          # G3 = X0 X1 X2^-1
              and (-a[1] + a[3] - a[4]) % 3 == 0         # G4 = X1^-1 X3 X4^-1
              and (a[0] + a[3]) % 3 == 0)                # logical X0 X3: |+_L>
        fails += not ok
    return kept, fails


def check_all():
    say("draw_three_layouts.py checks (noiseless sdim, %d shots)" % SHOTS)
    enc_a = from_ops(GARNET_ENCODER, nwires=5)
    term = Reg.resolve_code_profile("folded_qutrit").code_definition.terminal_const0_measurement()
    enc_b = one_ancilla_encoder()
    enc_b_meas = Circuit(dimension=3, num_qudits=9) + enc_b
    enc_b_meas.add_gate("M", 5)
    enc_b_meas.add_gate("M", 6)
    round_b5, round_b7 = GEN_ONE._check_round_circuit(), GEN_ONE_PUB._check_round_circuit()
    enc_c = Reg.resolve_code_profile("folded_qutrit_layout_b").code_definition.logical_plus_initial_state()
    round_c = GEN_TWO._check_round_circuit()
    paper_enc = Reg.resolve_code_profile("folded_qutrit").code_definition.logical_plus_initial_state()
    # couplers
    on_couplers("a", GEN_ONE, enc_a, "encoder")
    on_couplers("b", GEN_ONE, enc_b, "encoder")
    on_couplers("b", GEN_ONE, round_b5, "five-moment round")
    on_couplers("b", GEN_ONE, round_b7, "seven-moment round of the published runs")
    on_couplers("c", GEN_TWO, enc_c, "encoder")
    on_couplers("c", GEN_TWO, round_c, "round")
    pairs = lambda circ: sorted({tuple(sorted(g[1:])) for g in ops_of(GEN_ONE, circ) if len(g) == 3})  # noqa: E731
    same = pairs(enc_a) == pairs(paper_enc)
    say(f"  (a) encoder couplers {pairs(enc_a)}; the paper's encoder uses "
        f"{pairs(paper_enc)}: {'the same' if same else 'different'}")
    # counts
    expect = {"(a) encoder": ((GEN_ONE, enc_a), (4, 3)),
              "(b) encoder": ((GEN_ONE, enc_b), (8, 5)),
              "(b) five-moment round": ((GEN_ONE, round_b5), (12, 5)),
              "(b) seven-moment round of the published runs": ((GEN_ONE_PUB, round_b7), (12, 7)),
              "(c) encoder": ((GEN_TWO, enc_c), (4, 4)),
              "(c) round": ((GEN_TWO, round_c), (24, 19))}
    for what, ((gen, circ), want) in expect.items():
        got = counts(gen, circ)
        assert got == want, (what, got, want)
        say(f"  {what}: {got[0]} two-qutrit gates, {got[1]} gate moments (one-qutrit layer included)")
    assert depth2(GEN_ONE, enc_a) == 2
    say(f"  (a) encoder: {depth2(GEN_ONE, enc_a)} two-qutrit gate moments")
    dd = sum(1 for g in ops_of(GEN_TWO, round_c) if len(g) == 3 and g[1] < 5 and g[2] < 5)
    assert dd == 16, dd
    say(f"  (c) round: {dd} of its two-qutrit gates act between data qutrits")
    # sdim
    empty5 = Circuit(dimension=3, num_qudits=5)
    runs = [("(a) encoder + terminal X readout", enc_a, empty5, 5, 0),
            ("(b) encoder + five-moment rounds", enc_b_meas, round_b5, 9, 3),
            ("(b) encoder + seven-moment rounds", enc_b_meas, round_b7, 9, 3),
            ("(c) encoder + rounds", enc_c, round_c, 7, 3)]
    for what, enc, rnd, nw, k in runs:
        kept, fails = simulate(enc, rnd, nw, rounds=k)
        assert fails == 0 and kept == SHOTS, (what, kept, fails)
        say(f"  {what}: {kept - fails}/{kept} shots pass every check")
    say("ALL CHECKS PASS")
    return enc_a, term, enc_b, round_b5, round_b7, enc_c, round_c, same


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[landscape,letterpaper,margin=1.4cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{adjustbox}
\usepackage{quantikz}
\usetikzlibrary{calc,arrows.meta,decorations.markings}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}

\definecolor{Zcol}{HTML}{FF1744}
\definecolor{Xcol}{HTML}{00B7FF}
\definecolor{SwapCol}{HTML}{8E44AD}
\colorlet{Xtint}{Xcol!26}
\colorlet{Ztint}{Zcol!17}
\colorlet{XRim}{Xcol}
\colorlet{ZRim}{Zcol}
\colorlet{XText}{Xcol!80!black}
\colorlet{ZText}{Zcol!85!black}
\colorlet{GateCol}{SwapCol}
\colorlet{WinCol}{black!62}
\colorlet{EncCol}{black!75}
\colorlet{WinTint}{black!14}

\def\u{1.40}
\def\rq{0.17}
\def\ss{0.40}
\def\lab{0.37}

\tikzset{
  every node/.style={font=\footnotesize},
  data/.style={circle,inner sep=0pt,minimum size=2*\rq cm,line width=0.7pt,
               draw=black,fill=white,font=\scriptsize},
  site/.style={rectangle,rounded corners=1.2pt,inner sep=0pt,minimum size=\ss cm,
               line width=0.7pt,font=\footnotesize},
  xsite/.style={site,draw=XRim,fill=Xtint},
  zsite/.style={site,draw=ZRim,fill=Ztint},
  used/.style={line width=0.9pt,draw=black},
  gate/.style={line width=1.2pt,draw=GateCol,line cap=round},
  midarrow/.style={postaction={decorate},decoration={markings,
      mark=at position 0.56 with {\arrow{Stealth[length=4.2pt,width=4.6pt]}}}},
  window/.style={line width=1.2pt,draw=WinCol,line cap=round,midarrow},
  ztransfer/.style={line width=1.0pt,draw=ZRim,dash pattern=on 3pt off 1.6pt},
  xtransfer/.style={line width=1.0pt,draw=XRim,dash pattern=on 3pt off 1.6pt},
  enc/.style={line width=0.9pt,draw=EncCol,dash pattern=on 2.4pt off 1.8pt,line cap=butt},
  encedge/.style={enc,dash phase=0.82pt},
  readout/.style={font=\footnotesize,inner sep=0pt},
  title/.style={font=\footnotesize,anchor=base west,inner sep=0pt},
  stats/.style={font=\scriptsize,anchor=north,inner sep=0pt,text=black!75},
}
\newcommand{\DataNodes}{\foreach \i in {0,...,4} \node[data] at (q\i) {\i};}
\newcommand{\Readout}[3]{\node[readout,anchor=#2+180] at ($(#1)+(#2:\lab)$) {#3};}
\newcommand{\Panel}[3]{\node[title] at (#1) {{\small\bfseries(#2)}\hspace{0.45em}#3};}
\newcommand{\SH}{\bar S,\,\bar H}
\newcommand{\KeyLine}[1]{\tikz[baseline=-0.5ex]\draw[#1] (0,0) -- (0.50,0);}
\newcommand{\DualSite}[2]{%
  \begin{scope}[shift={(#1)}]
    \def\h{0.5*\ss}
    \pgfmathsetmacro{\cl}{0.5*\ss+0.04}
    \def\sq{(-\h,-\h) rectangle (\h,\h)}
    \begin{scope}
      \clip (-\cl,\cl) -- (-\cl,-\cl) -- (\cl,-\cl) -- cycle;
      \fill[Ztint,rounded corners=1.2pt] \sq;
      \draw[ZRim,line width=0.7pt,rounded corners=1.2pt] \sq;
    \end{scope}
    \begin{scope}
      \clip (-\cl,\cl) -- (\cl,\cl) -- (\cl,-\cl) -- cycle;
      \fill[Xtint,rounded corners=1.2pt] \sq;
      \draw[XRim,line width=0.7pt,rounded corners=1.2pt] \sq;
    \end{scope}
    \node[font=\footnotesize,inner sep=0pt] at (0,0) {#2};
  \end{scope}}
\newcommand{\KeyDual}{\tikz[baseline=-0.5ex]{\def\ss{0.30}\DualSite{0,0}{}}}

\begin{document}
"""


def circ_block(tag, title, body):
    return (rf"{{\small\bfseries {tag}}}\hspace{{0.5em}}{{\small {title}}}\par\nopagebreak" + "\n"
            + r"\adjustbox{max width=\linewidth}{" + "\n" + body + "\n}\\par\\vspace{12pt}\n")


def main():
    enc_a, term, enc_b, round_b5, round_b7, enc_c, round_c, same = check_all()
    (HERE / "three-layouts-verify.out").write_text("\n".join(LOG) + "\n")
    one, two = ONE_ANCILLA, TWO_ANCILLA
    parts = [PREAMBLE]
    parts.append(r"""{\large\bfseries Three layouts of the $[[5,1,2]]_3$ code: couplers, encoders and check circuits}\par
{\small Review document, 2026-11-19, not a paper figure. Layout panels copied from the figure session's
\texttt{ancilla-layouts-B.tex}; circuits generated by \texttt{draw\_three\_layouts.py} from the production code,
checked in \texttt{three-layouts-verify.out}. Stabilizers: \textcolor{ZText}{$G_1 = Z_0 Z_1^{-1} Z_3^{-1}$},
\textcolor{ZText}{$G_2 = Z_1 Z_2 Z_4^{-1}$}, \textcolor{XText}{$G_3 = X_0 X_1 X_2^{-1}$},
\textcolor{XText}{$G_4 = X_1^{-1} X_3 X_4^{-1}$}.}\par\vspace{10pt}
\begin{center}
\input{three-layouts-panels.tex}
\end{center}
\vspace{6pt}
{\small
\begin{itemize}\setlength{\itemsep}{2pt}
\item (a) No ancillas and no check rounds. The sequence runs on five of Garnet's transmons,
  $q_0, \dots, q_4$ = QB1, QB4, QB5, QB2, QB6. At the end every data qutrit is read in $X$, and shots
  with $G_3 \neq 0$ or $G_4 \neq 0$ are discarded. The encoder uses QB1--QB4, QB5--QB6, QB2--QB1,
  QB4--QB5; $\bar S$ and $\bar H$ use QB5--QB2.
\item (b) Every ancilla is a nearest neighbour of the three data qutrits of its check. The only
  data--data coupler is $q_2$--$q_3$, used by $\bar S$ and $\bar H$ only.
\item (c) $N$ and $E$ each read one $Z$-type and one $X$-type check per round. 16 of the 24
  two-qutrit gates act between data qutrits and are undone inside the round.
  $q_2$--$q_3$ carries the logical gates and 4 gates of each round.
\end{itemize}}
\clearpage
""")
    parts.append(r"{\large\bfseries Encoding circuits: $\ket{0}^{\otimes 5} \to \ket{\bar +}$}\par\vspace{6pt}" + "\n")
    parts.append(r"\noindent\begin{minipage}[t]{0.34\linewidth}" "\n")
    parts.append(circ_block("(a)", r"Terminal check on Garnet: 4 two-qutrit gates in "
                            rf"{depth2(GEN_ONE, enc_a)} two-qutrit gate moments",
                            quantikz(GEN_ONE, enc_a, DATA, round_=False)))
    parts.append(r"\end{minipage}\hfill\begin{minipage}[t]{0.62\linewidth}" "\n")
    parts.append(circ_block("(b)", r"One-ancilla layout: 8 two-qutrit gates through $a_1$ and $a_2$, "
                            rf"which end in $\ket{{0}}$; {depth2(GEN_ONE, enc_b)} two-qutrit gate moments",
                            quantikz(GEN_ONE, enc_b, one["names"][:7], round_=False, encoder_anc=(5, 6))))
    parts.append(r"\end{minipage}\par\vspace{4pt}" "\n")
    parts.append(circ_block("(c)", r"Two-ancilla layout: 4 two-qutrit gates on data couplers "
                            rf"($q_0$--$q_3$ is used only here); {depth2(GEN_TWO, enc_c)} two-qutrit gate moments",
                            quantikz(GEN_TWO, enc_c, DATA, round_=False)))
    same_text = (r"These are exactly the couplers of the paper's 4-gate encoding circuit; this order "
                 r"needs two two-qutrit moments." if same else
                 r"The paper's 4-gate encoding circuit uses other couplers.")
    parts.append(r"""{\small
\begin{itemize}\setlength{\itemsep}{2pt}
\item (a) """ + same_text + r"""
\item (b) The paper's encoding circuit needs the data couplers $q_0$--$q_1$, $q_0$--$q_3$, $q_1$--$q_2$,
  $q_2$--$q_4$. The one-ancilla layout has none of them, so its encoder routes through two ancillas.
\item (c) Uses the layout's data couplers plus $q_0$--$q_3$, which is there only for the encoder.
\item Every LRB-D simulation starts from an ideal $\ket{\bar +}$, so no encoder changes $p^*$.
\end{itemize}}
\clearpage
""")
    parts.append(r"{\large\bfseries Check circuits}\par" "\n"
                 r"{\small Each column group between dashed lines is one gate moment of the LRB generator's "
                 r"scheduler. Every round also has one reset layer (the $\ket{0}$ on the left) and its readout "
                 r"layers. A dot joined to a box $X$ ($X^{-1}$) is a SUM gate (its inverse), control on the "
                 r"dot. Grey boxes: gates between data qutrits, undone inside the round.}\par\vspace{8pt}" + "\n")
    parts.append(r"\noindent\begin{minipage}[t]{0.26\linewidth}" "\n")
    parts.append(circ_block("(a)", r"Terminal check: no rounds; once, at the end of the sequence",
                            quantikz(GEN_ONE, term, DATA, TERMINAL_READS)))
    parts.append(r"\end{minipage}\hfill\begin{minipage}[t]{0.70\linewidth}" "\n"
                 r"{\small\vspace{14pt}Keep the shot if \textcolor{XText}{$G_3 = x_0 + x_1 - x_2$} and "
                 r"\textcolor{XText}{$G_4 = -x_1 + x_3 - x_4$} are $0 \pmod 3$. The logical outcome is "
                 r"$x_0 + x_3$. The $Z$-type checks are never measured.}" "\n"
                 r"\end{minipage}\par\vspace{8pt}" "\n")
    parts.append(circ_block("(b)", r"One-ancilla layout, the five-moment round: 12 two-qutrit gates, 5 gate moments",
                            quantikz(GEN_ONE, round_b5, one["names"], one["reads"], one["ztype"])))
    parts.append(circ_block("(b$'$)", r"One-ancilla layout, the seven-moment round of the published runs: "
                            r"the same 12 gates in 7 gate moments",
                            quantikz(GEN_ONE_PUB, round_b7, one["names"], one["reads"], one["ztype"])))
    parts.append(r"\clearpage" "\n")
    parts.append(circ_block("(c)", r"Two-ancilla layout: 24 two-qutrit gates, 19 gate moments; "
                            r"$N$ reads $G_2$ then $G_4$, $E$ reads $G_1$ then $G_3$ (mid-round readouts, "
                            r"no reset in between)",
                            quantikz(GEN_TWO, round_c, two["names"], two["reads"], two["ztype"])))
    parts.append(r"""{\small The five-moment and seven-moment rounds are compared in LRB-D in
\texttt{comparisons/sdim140-prod5b/README.md}.}
\end{document}
""")
    (HERE / "three-layouts.tex").write_text("".join(parts))
    print(HERE / "three-layouts.tex")


if __name__ == "__main__":
    main()
