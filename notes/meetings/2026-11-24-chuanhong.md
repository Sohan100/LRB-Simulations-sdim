# Meeting with Chuanhong: qutrit vs. qubit error correction, paper message, layout, repo

Notes received 2026-11-24 (Granola). Attendees: Sohan, Chuanhong. The transcript is in Granola;
the link is kept out of the repository.

## Qutrit vs. qubit error correction: core comparison

- Qutrit pseudo-threshold is higher than the qubit equivalent under circuit-level noise
  - Means qutrit gates can be worse in absolute fidelity and still achieve logical error protection
  - Qubit pseudo-threshold roughly 1% in the examples discussed; qutrit pseudo-threshold around 3%
- Break-even framing: for qutrit error correction to outperform qubit, qutrit gate fidelity must
  clear its own (lower) pseudo-threshold
  - Example: 98% qutrit gate fidelity may be sufficient for error protection where 98% qubit gate
    fidelity is not enough for the qubit code
  - Concrete scenario discussed: qubit at 97%, qutrit at 98%; qutrit passes its pseudo-threshold,
    qubit does not
- Anticipated reviewer pushback: qubit gates routinely achieve 2-9 or 3-9 fidelity; qutrit gates lag
  behind
  - If a QPU runs qubits at 99% and qutrits at 98%, both above their respective break-even points,
    no clear native advantage to choosing qutrit error correction
  - At that point, the easier gate wins by default
- Qutrit state-space advantage noted but considered a weak standalone argument
  - Qutrits naturally suited to spin-1 and SU(3) simulations (e.g., neutrino flavor modeling)
  - However, qubit encodings can replicate this, so not decisive
- Stabilizer-to-magic-state avenue flagged as an open research question
  - A high-fidelity qutrit stabilizer state, if teleported to a qubit, becomes a magic state
  - The omega and omega-squared phases picked up in the qutrit stabilizer setting become magic for
    qubits
  - Could enable more resource-efficient magic state preparation, but speculative and not yet
    demonstrated
- Heterogeneous QPU fidelity question raised: if 10 of 100 qubits fall below threshold, does error
  correction break?
  - No clean yes/no answer; depends on the encoding and where the low-fidelity sites sit
  - Consensus: every physical site must meet the minimum threshold; even one site above the
    crossover point degrades logical error rates

## Paper messaging: three candidate arguments

- Need one strong, concrete message to target a high-impact journal; current analysis alone is not
  sufficient
- Three candidate angles identified:
  1. Concrete application where qutrit error correction outperforms qubit, even with worse physical
     gates
  2. Proposed hardware layout is more practical and novel than prior proposals
  3. First systematic noise analysis framework for qutrit error correction
- On angle 1: no clear application identified during the meeting
  - Suggestion to cross-reference experimental desiderata from the qutrit hardware side with
    simulation results
  - Chuanhong proposed testing robustness against gate drift, e.g., Gaussian-distributed fidelity
    spread across a QPU
  - No conclusion reached; flagged as the most important open question for the paper
- On angle 3: noise model contribution may be underappreciated
  - Sohan scaled SD6 depolarizing noise non-biasedly across all 81 qutrit parameters
  - No existing qutrit device means bias assumptions are unjustifiable; unbiased treatment is the
    defensible baseline
  - Community actively debating how to define qutrit noise channels (referenced a recent conference
    discussion)
  - Likely the first systematic treatment of circuit-level noise for qutrit error correction
  - Chuanhong confirmed this could stand as a strong message if framed as a methodological
    contribution

## Hardware layout

- "Dorito chip" triangular layout: distance-5 code, all couplings nearest-neighbor or
  next-nearest-neighbor
  - Supports distance-3 and distance-2 sub-codes on the same chip
  - Long-range gates are only next-nearest-neighbor, keeping connectivity practical
  - Described as best-case for single ancillas since everything is effectively nearest-neighbor
- Compared to Musa (Moussa) et al.: assumes a square lattice surface code with background ancillas
  for folding
  - Background ancillas required to implement the fold are not feasibly realizable
  - Sohan's layout achievable with current superconducting qubit/qutrit technology
- Scope explicitly limited to superconducting platforms; neutral atom platforms excluded
  - Neutral atoms allow arbitrary connectivity, which removes the layout challenge entirely
- Also referenced IQM Garnet chip as a real-world layout analog discussed previously with Noah
- Dorito chip layout PDF not located during the meeting; Sohan to regenerate from physical notes
  - Leaning toward excluding from the paper given volume of existing material
  - Decision deferred pending Chuanhong's review of the regenerated figure
- Overleaf screenshot shared during meeting: "Nearest Neighbour Folded Surface Code" paper, page 7
  of 19
  - Shows lattice diagrams of H to H-prime transformation, Layer 2, with H, H-inverse, and SWAP
    legend
  - This was the folded surface code PDF, confirmed not to be the Dorito chip layout

## Collaboration and repo setup

- Plan to consolidate everything into the existing LRBD GitHub repo:
  - Meeting notes (via Granola, potentially auto-piped through Claude)
  - Simulation code
  - Paper LaTeX source files
  - To-do list and async discussion threads
- Overleaf not prioritized for the repo; raw LaTeX files are sufficient
- Chuanhong's lab already uses an AI-backed repo as a "digital brain" for projects
  - Stores discussions, papers, plots, manuscripts, and to-do lists
  - Enables async idea capture: one person posts, the other responds later with full context
- Sohan to set up the structure; Chuanhong will contribute once it is live
- Ongoing async communication via Slack

## Next steps

- Regenerate and share the Dorito chip layout figure (Sohan): recreate from physical notes and send
  to Chuanhong to decide whether to include in the paper.
- Set up LRBD repo with meeting notes, code, and LaTeX (Sohan): add meeting notes, paper, and code
  folders; Chuanhong will contribute once structure is live.
- Identify one strong qutrit advantage to anchor the paper's main message: focus candidates are an
  application-specific advantage, layout novelty, or the noise analysis framework. Most urgent open
  question before targeting a high-impact journal.

## Follow-up (2026-11-24, Sohan)

- The Dorito chip is the folded-surface-code layout regenerated in
  `layouts/folded-surface-code/` (figure `dorito-chip.pdf`, write-up `layout-note.pdf`).
- The repository now has `manuscript/` (LaTeX), `notes/` (this folder) and the code; see the
  top-level README.
