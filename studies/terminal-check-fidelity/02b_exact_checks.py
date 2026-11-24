"""Consistency checks of the exact machinery (tcsteps, tcfit). Prints PASS/FAIL lines.

1. Step factors x frames reproduce the directly built depth-m circuits of
   the production sequences (max abs difference of outcome probabilities).
2. The frame transfer matrix reproduces a brute-force average over random
   sequences, mirror and compiled closing (difference in units of the
   brute-force standard error).
3. No single fault is an accepted logical error: the accepted infidelity of
   one pair scales as p^2 down to p = 1e-6.
4. For noise that is independent between the error blocks {0}, {1}, {4},
   {2,3}, the rates of the logical modes and of the stabilizer modes have
   the same sum (the first-order cancellation), for both dimensions.
5. Pair = 4 x single: one mirror pair has four times the logical infidelity
   of one Clifford.
"""
from __future__ import annotations

import numpy as np

import tclib as T
import tcexact as E
import tcfit as F
import tcsteps as S


def line(ok, text):
    print(("PASS  " if ok else "FAIL  ") + text, flush=True)


def main():
    rng = np.random.default_rng(5)
    for code in ("qubit", "qutrit"):
        for noise in ("si1000", "sd6"):
            sm = S.StepModel(code, noise)
            d = sm.d
            md = F.modes(d)
            p = 5e-3
            # 1
            words = E.production_words(code, noise)
            tab = E.sequence_table(code, noise, words, list(range(23)), workers=16, name="prod_dense")
            P = tab.dist(p)
            widx = np.array([[sm.cat.index(w) for w in seq] for seq in words])
            chi = sm.sequence_chars(p, widx, list(range(23)))
            err = np.abs(np.clip(T.dist_from_chars(chi, d), 0, None) - P).max()
            line(err < 1e-12, f"{code} {noise}: step factors vs direct circuits, max |dP| = {err:.1e}")
            # 2
            for protocol in ("mirror", "compiled"):
                depths = [0, 1, 2, 4, 8, 16]
                ex = sm.exact_chars(p, depths, protocol)
                bf = sm.sequence_chars(p, sm.draw(3000, 16, rng), depths, protocol)
                z = (bf.mean(0) - ex) / (bf.std(0) / np.sqrt(bf.shape[0]) + 1e-300)
                zmax = np.abs(z[1:, 1:]).max()
                line(zmax < 4.5, f"{code} {noise} {protocol}: transfer matrix vs 3000 random sequences, max |z| = {zmax:.2f}")
            # 3
            e = [sm.ground_truth(q)["pair_stat"] / q ** 2 for q in (1e-6, 1e-5, 1e-4)]
            line(abs(e[0] / e[1] - 1) < 1e-3 and abs(e[1] / e[2] - 1) < 1e-2,
                 f"{code} {noise}: eps_pair / p^2 = {e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f} at p = 1e-6, 1e-5, 1e-4")
            # 5
            gt = sm.ground_truth(p)
            ratio = gt["pair_stat"] / gt["single_stat"]
            line(abs(ratio - 4) < 0.1, f"{code} {noise}: pair / single = {ratio:.3f} at p = {p:g}")
    # 4
    for d in (2, 3):
        md = F.modes(d)
        rows = md.expand(md.M_block.T)                    # block x K
        diff = rows[:, md.L].sum(1) - rows[:, md.S].sum(1)
        line(np.all(diff == 0), f"d = {d}: sum of logical-mode rates = sum of stabilizer-mode rates for every block "
             f"({len(md.L)} logical modes, {len(md.S)} stabilizer modes, {md.M_block.shape[1]} block rates)")


if __name__ == "__main__":
    main()
