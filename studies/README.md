# Side studies

Each folder here holds one side study of the LRB-D simulations. Each has a `README.md` with its question, its status and how to run it, and a `REPORT.md` with the full analysis. The production pipeline lives in `src/lrb/` and `scripts/`. The studies build on it but do not change it.

| Study | Question | Answer in one line | Folder |
|---|---|---|---|
| Non-Markovian decay | Why do some LRB-D curves bend away from a single exponential, and how can we still quote a logical gate fidelity? | Under SD6 the unchecked mirror inverse adds an m² term; one compiled recovery Clifford fixes the shape at low p, and the report recommends quoting r_L and d_L per Clifford with m = 0 in every fit. | [`non-markovian/`](non-markovian/) |
| Larger local dimension | How does the pseudo-threshold change as q grows, and up to which postselection interval k is the decay a single exponential? | p* rises with q but more slowly at each step (0.0123, 0.0313, 0.0596, 0.0764 for q = 2, 3, 5, 7 under SI1000 in the five-moment round), random acceptance caps it near 0.1, and a single exponential holds for k ≤ 2 under SI1000. | [`qudit-scaling/`](qudit-scaling/) |
| Terminal-check fidelity | Can terminal-check (const-0) data give a logical gate fidelity we can compare with physical RB? | Yes, from a joint fit of the Fourier modes of the full terminal record; a decay fit of the postselected curve overstates the infidelity 6 to 11 times. | [`terminal-check-fidelity/`](terminal-check-fidelity/) |
| Correction decoders, qutrit | Can LRB correct errors instead of rejecting shots on the [[5,1,2]]_3 code? | Not at distance 2: full MLD correction is 3.1 times worse than physical RB at p = 1e-3, and selective rules and protocol changes help more than the decoder. | [`correction-decoders/qutrit/`](correction-decoders/qutrit/) |
| Correction decoders, qubit | The same question for the qubit [[5,1,2]] code. | Not at distance 2: MLD leaves a first-order error 20 to 40 times a bare qubit's; exact MLD with soft-output acceptance is the right rule, and genuine correction needs distance 3. | [`correction-decoders/qubit/`](correction-decoders/qubit/) |

## How the studies relate to the package

- The non-Markovian, qudit-scaling and terminal-check studies import `lrb` from `src/` and run against the current package. After the move we reran table scripts of the qudit-scaling and terminal-check studies and got the same output. The non-Markovian scripts import cleanly, but most of them need the production run data to run.
- The qubit decoder study is integrated. Its package is `src/lrb_qubit/`, its tests are `tests/test_lrb_qubit_model.py` and `tests/test_qubit512_*.py`, and they need stim, PyMatching and ldpc.
- The qutrit decoder study is archived. Its code targets commit `c21386f6`, and its README says how to run it there.

## What is not here

We kept the reports, the scripts, the figures the reports show, and the result tables. Raw sampled shots, caches, logs and superseded drafts stay out of git. Each study's README lists what we left out and how to regenerate it. Scripts that read production runs need the run folders under `LRB-experiment-data-slurm/`, whose raw data are not in git.
