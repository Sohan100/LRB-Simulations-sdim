# L6 citation check: superconducting qutrit hardware and IQM

Independent check of the 33 citations behind `lit-L6-scqutrit.md`, done on 2026-10-28.

Label for everything below: **from literature**. Each verdict rests on pages I fetched in
this task, not on memory. Where I say a phrase or number is "present", I found it in the
fetched text.

## Result

- 26 confirmed, 7 corrected, 0 not found.
- No entry is fabricated. Every arXiv ID resolves to the cited paper and every cited DOI
  resolves to the cited article.
- The corrections are bibliographic:
  - two published titles differ from the cited ones (Wang2025, Renger2025);
  - one journal version was missed entirely (Renger2025);
  - three venues were incomplete or unknown (Goss2024 and Nguyen2024 lack article numbers,
    Haupt2025 had no journal);
  - one author count is wrong (Vigneau2025 has 16 authors, not 17);
  - one year follows the version read instead of first submission (Vora2026).
- No statement in the notes contradicts an abstract page I read. Three wording or support
  problems are listed under "Flags on the notes file".

## Method

1. arXiv abstract page `https://arxiv.org/abs/<id>` for all 27 arXiv entries, all HTTP 200.
   Title, authors, DOI and dates come from the page's `citation_*` meta tags. Journal
   reference and version history come from the page body.
2. Crossref record `https://api.crossref.org/works/<doi>` for all 21 cited DOIs plus the
   Renger2025 DOI found here. This gives the published title, authors, journal, volume,
   article number and publication date.
3. Publisher landing page through `https://doi.org/<doi>` for Goss2024, Nguyen2024
   (nature.com), Wang2025, Haupt2025, Vigneau2025, Renger2025, Morvan2021
   (journals.aps.org), and the Goss2022 Author Correction (nature.com). All agreed with
   Crossref.
4. For arXiv-only entries, Crossref title and author searches plus web searches for a
   journal version.
5. The six web documents, fetched directly.
6. Spot checks of quoted text in full texts: arXiv HTML of Blok2021, Goss2024, Goss2026,
   Haupt2025 and Vigneau2025; arXiv PDF of Abdurakhimov2024, text extracted with pypdf from
   the worktree's `.venv-pdf/`.

Fetched-column codes: A = arXiv abstract page, C = Crossref record, P = publisher landing
page, S = Crossref or web search for a journal version, F = full-text spot check,
W = web page.

## Verdicts

| Key | Verdict | Title as verified | Authors | Year | arXiv | DOI | Venue as verified | Fetched | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Blok2021 | confirmed | Quantum Information Scrambling on a Superconducting Qutrit Processor | M. S. Blok, V. V. Ramasesh, T. Schuster, K. O'Brien, J. M. Kreikebaum, D. Dahlen, A. Morvan, B. Yoshida, N. Y. Yao, I. Siddiqi | 2021 | 2003.03307 | 10.1103/PhysRevX.11.021010 | Phys. Rev. X 11, 021010 (2021-04-09) | A, C, F | The arXiv title says "in a" where the journal title says "on a"; the cited title is the journal one. Full text has CSUM QPT fidelity 0.889 and readout "largely limited by decay during readout". It states no readout duration. |
| Morvan2021 | confirmed | Qutrit Randomized Benchmarking | A. Morvan, V. V. Ramasesh, M. S. Blok, J. M. Kreikebaum, K. O'Brien, L. Chen, B. K. Mitchell, R. K. Naik, D. I. Santiago, I. Siddiqi | 2021 | 2008.09134 | 10.1103/PhysRevLett.126.210504 | Phys. Rev. Lett. 126, 210504 (2021-05-27) | A, C, P | arXiv v1 only. The arXiv abstract says infidelity "as low as 2.38e-3" and CSUM process fidelity 0.82. The APS abstract says "average single-qutrit process infidelity of 3.8e-3" and CSUM 0.85. The notes report both correctly. |
| Goss2022 | confirmed | High-fidelity qutrit entangling gates for superconducting circuits | N. Goss, A. Morvan, B. Marinelli, B. K. Mitchell, L. B. Nguyen, R. K. Naik, L. Chen, C. Jünger, J. M. Kreikebaum, D. I. Santiago, J. J. Wallman, I. Siddiqi | 2022 | 2206.07216 | 10.1038/s41467-022-34851-z | Nat. Commun. 13, 7481 (2022-12-05) | A, C, P | An Author Correction exists: Nat. Commun. 14, 4256 (2023), DOI 10.1038/s41467-023-40049-8. It only fixes a funding sentence in the acknowledgements, so no numbers change. The abstract confirms CZ† 97.3(1)% and CZ 95.2(3)%. |
| CerveraLierta2022 | confirmed | Experimental high-dimensional Greenberger-Horne-Zeilinger entanglement with superconducting transmon qutrits | A. Cervera-Lierta, M. Krenn, A. Aspuru-Guzik, A. Galda | 2022 | 2104.05627 | 10.1103/PhysRevApplied.17.024062 | Phys. Rev. Applied 17, 024062 (2022-02-23) | A, C | The abstract confirms GHZ fidelity 76 ± 1% on a cloud device. |
| Liu2023 | confirmed | Performing SU(d) operations and rudimentary algorithms in a superconducting transmon qudit for d=3 and d=4 | P. Liu, R. Wang, J.-N. Zhang, Y. Zhang, X. Cai, H. Xu, Z. Li, J. Han, X. Li, G. Xue, W. Liu, L. You, Y. Jin, H. Yu | 2023 | 2304.11841 | 10.1103/PhysRevX.13.021028 | Phys. Rev. X 13, 021028 (2023-05-23) | A, C | The journal uses title case. Nothing else differs. |
| Fischer2023 | confirmed | Universal qudit gate synthesis for transmons | L. E. Fischer, A. Chiesa, F. Tacchino, D. J. Egger, S. Carretta, I. Tavernelli | 2023 | 2212.04496 | 10.1103/PRXQuantum.4.030327 | PRX Quantum 4, 030327 (2023-08-28) | A, C | The abstract predicts two-ququart CR fidelities "beyond 99%", consistent with the notes. |
| Roy2023 | confirmed | Two-Qutrit Quantum Algorithms on a Programmable Superconducting Processor | T. Roy, Z. Li, E. Kapit, D. I. Schuster | 2023 | 2211.06523 | 10.1103/PhysRevApplied.19.064024 | Phys. Rev. Applied 19, 064024 (2023-06-07) | A, C | The arXiv title is "Realization of two-qutrit quantum algorithms on a programmable superconducting processor". The arXiv page carries no DOI link. |
| Luo2023 | confirmed | Experimental Realization of Two Qutrits Gate with Tunable Coupling in Superconducting Circuits | K. Luo, W. Huang, Z. Tao, L. Zhang, Y. Zhou, J. Chu, W. Liu, B. Wang, J. Cui, S. Liu, F. Yan, M.-H. Yung, Y. Chen, T. Yan, D. Yu | 2023 | 2206.11199 | 10.1103/PhysRevLett.130.030603 | Phys. Rev. Lett. 130, 030603 (2023-01-20) | A, C | The abstract confirms CPhase 89.3% and EPR 95.5%. |
| Goss2024 | corrected | Extending the computational reach of a superconducting qutrit processor | N. Goss, S. Ferracin, A. Hashim, A. Carignan-Dugas, J. M. Kreikebaum, R. K. Naik, D. I. Santiago, I. Siddiqi | 2024 | 2305.16507 | 10.1038/s41534-024-00892-z | npj Quantum Inf. 10, 101 (2024-10-14) | A, C, P, F | Venue corrected: the article number is 101, from Crossref and the nature.com page. The arXiv page carries no DOI link. I found no readout duration in the arXiv HTML. |
| Subramanian2023 | confirmed | Efficient two-qutrit gates in superconducting circuits using parametric coupling | M. Subramanian, A. Lupascu | 2023 | 2309.05766 | 10.1103/PhysRevA.108.062616 | Phys. Rev. A 108, 062616 (2023-12-20) | A, C | |
| Goss2026 | confirmed | A Qutrit Time Crystal Stabilized with Native Chiral Interactions | N. Goss, N. Suri, B. Marinelli, L. Chen, A. Hashim, S. Anand, A. Morvan, R. K. Naik, E. Rrapaj, D. I. Santiago, W. de Jong, N. Y. Yao, J. E. Moore, I. Siddiqi | 2026 | 2605.14293 | none | arXiv preprint, v1 2026-05-14 | A, S, F | No journal version found. Full text confirms "designed, fabricated, and packaged by IQM", control by LBNL's open-source qubic system, a 500 ns readout, 93% average qutrit readout fidelity, terminal readout only, and 6 native pulses per Clifford. |
| Nguyen2024 | corrected | Empowering a qudit-based quantum processor by traversing the dual bosonic ladder | L. B. Nguyen, N. Goss, K. Siva, Y. Kim, E. Younis, B. Qing, A. Hashim, D. I. Santiago, I. Siddiqi | 2024 | 2312.17741 | 10.1038/s41467-024-51434-2 | Nat. Commun. 15, 7117 (2024-08-19) | A, C, P | Venue corrected: the article number is 7117. The journal title is the cited one. The arXiv title is "Empowering high-dimensional quantum computing by traversing the dual bosonic ladder". |
| Vora2026 | corrected | ML-Powered FPGA-based Real-Time Quantum State Discrimination Enabling Mid-circuit Measurements | N. R. Vora, Y. Xu, A. Hashim, N. Fruitwala, H. N. Nguyen, N. Goss, H. Liao, J. Balewski, A. Rajagopala, K. Nowrouzi, Q. Ji, K. B. Whaley, I. Siddiqi, P. Nguyen, G. Huang | 2024 | 2406.18807 | none | arXiv preprint, v1 2024-06-27, v5 2026-09-13 | A, S | Year corrected to 2024: arXiv's citation date is the v1 date, and 2026 is the v5 date. Title and 15 authors confirmed. Crossref also lists two-page IEEE QCE abstracts titled "QubiCML: ML-Powered Real-Time Quantum State Discrimination Enabling Mid-Circuit Measurements": QCE 2024 pp. 414-415, DOI 10.1109/QCE60285.2024.10332, and QCE 2025 pp. 526-527, DOI 10.1109/QCE65121.2025.10428, each with 14 authors and no N. Goss. I did not read them. The abstract confirms 40 ns inference and a conditional qutrit protocol. |
| Chen2023 | confirmed | Transmon qubit readout fidelity at the threshold for quantum error correction without a quantum-limited amplifier | L. Chen, H.-X. Li, Y. Lu, C. W. Warren, C. J. Križan, S. Kosen, M. Rommel, S. Ahmed, A. Osman, J. Biznárová, A. Fadavi Roudsari, B. Lienhard, M. Caputo, K. Grigoras, L. Grönberg, J. Govenius, A. F. Kockum, P. Delsing, J. Bylander, G. Tancredi | 2023 | 2208.05879 | 10.1038/s41534-023-00689-6 | npj Quantum Inf. 9, 26 (2023-03-16) | A, C | The abstract confirms 140 ns, 99.5% two-state and 96.9% three-state fidelity. |
| Kanazawa2023 | confirmed | Qutrit state discrimination with mid-circuit measurements | N. Kanazawa, H. Emori, D. C. McKay | 2023 | 2309.11303 | none | arXiv preprint, v1 2023-09-20 | A, S | No journal version found by Crossref title search, Crossref author search or web search. |
| Kehrer2024 | confirmed | Improving transmon qudit measurement on IBM Quantum hardware | T. Kehrer, T. Nadolny, C. Bruder | 2024 | 2307.13504 | 10.1103/PhysRevResearch.6.013050 | Phys. Rev. Research 6, 013050 (2024-01-12) | A, C | |
| Peterer2015 | confirmed | Coherence and Decay of Higher Energy Levels of a Superconducting Transmon Qubit | M. J. Peterer, S. J. Bader, X. Jin, F. Yan, A. Kamal, T. J. Gudmundsen, P. J. Leek, T. P. Orlando, W. D. Oliver, S. Gustavsson | 2015 | 1409.6031 | 10.1103/PhysRevLett.114.010501 | Phys. Rev. Lett. 114, 010501 (2015-01-06) | A, C | The abstract confirms mainly sequential decay with all relaxation times above 20 μs. |
| Wang2025 | corrected | High-E_J/E_C transmon qudits with up to 12 levels | Z. Wang, R. W. Parker, E. Champion, M. S. Blok | 2025 | 2407.17407 | 10.1103/PhysRevApplied.23.034046 | Phys. Rev. Applied 23, 034046 (2025-03-20) | A, C, P | Title corrected. The cited "Systematic study of High E_J/E_C transmon qudits up to d = 12" is the arXiv title. The journal title above comes from the APS page and Crossref. Other fields confirmed. |
| Magnard2018 | confirmed | Fast and Unconditional All-Microwave Reset of a Superconducting Qubit | P. Magnard, P. Kurpiers, B. Royer, T. Walter, J.-C. Besse, S. Gasparinetti, M. Pechal, J. Heinsoo, S. Storz, A. Blais, A. Wallraff | 2018 | 1801.07689 | 10.1103/PhysRevLett.121.060502 | Phys. Rev. Lett. 121, 060502 (2018-08-07) | A, C | The abstract confirms 0.2% residual excitation in under 500 ns with no feedback. |
| McEwen2021 | confirmed | Removing leakage-induced correlated errors in superconducting quantum error correction | M. McEwen, D. Kafri, Z. Chen, et al. (51 authors, last R. Barends) | 2021 | 2102.06131 | 10.1038/s41467-021-21982-y | Nat. Commun. 12, 1761 (2021-03-19) | A, C | The author count of 51 matches on arXiv and Crossref. |
| Haupt2025 | corrected | Mitigating errors in state preparation and measurement with noncomputational states | C. J. Haupt, A. Carrera Vazquez, L. E. Fischer, S. Woerner, D. J. Egger | 2026 | 2506.09145 | 10.1103/jv9h-spzw | Phys. Rev. Applied 25, 024079 (2026-02-25) | A, C, P, F | Venue and year corrected; the journal was previously unknown. The arXiv v1 title (2025-06-10) was "State-preparation and measurement error mitigation with non-computational states". v2 and the journal use the cited title. Full text has the quoted "ineffective at removing the \|2⟩-state population" and an IBM Eagle experiment with a slow passive reset. |
| Hyyppa2024 | confirmed | Reducing leakage of single-qubit gates for superconducting quantum processors using analytical control pulse envelopes | E. Hyyppä, A. Vepsäläinen, M. Papič, C. F. Chan, S. Inel, A. Landra, W. Liu, J. Luus, F. Marxer, C. Ockeloen-Korppi, S. Orbell, B. Tarasinski, J. Heinsoo | 2024 | 2402.17757 | 10.1103/PRXQuantum.5.030353 | PRX Quantum 5, 030353 (2024-09-19) | A, C | The abstract confirms leakage below 3.0e-5 at 6.25 ns and an anharmonicity of -212 MHz. |
| LiuBJ2026 | confirmed | Hardware-Efficient Erasure Qubits With Superconducting Transmon Qutrits | B.-J. Liu, Y.-Y. Wang, Y.-X. Wang, M. Badbaria, S. Puri, C. Wang | 2026 | 2604.08672 | none | arXiv preprint, v1 2026-04-09 | A, S | No journal version found. The abstract confirms a postselected logical T1 above 500 μs, more than 300 μs coherence with dynamical decoupling, and Clifford infidelity of order 1e-4. |
| Brock2025 | confirmed | Quantum error correction of qudits beyond break-even | B. L. Brock, S. Singh, A. Eickbusch, V. V. Sivak, A. Z. Ding, L. Frunzio, S. M. Girvin, M. H. Devoret | 2025 | 2409.15065 | 10.1038/s41586-025-08899-y | Nature 641, 612-618 (2025; online 2025-05-14) | A, C | The abstract confirms gains of 1.82 ± 0.03 for the qutrit and 1.87 ± 0.03 for the ququart. |
| Abdurakhimov2024 | confirmed | Technology and Performance Benchmarks of IQM's 20-Qubit Quantum Computer | L. Abdurakhimov, J. Adam, H. Ahmad, et al. (114 authors) | 2024 | 2408.12433 | none | arXiv preprint, v1 2024-08-22 | A, S, F | No journal version found. The PDF has a CZ "typical duration of 20 ns – 40 ns", median single-qubit error 9e-4, CZ error 5e-3, two-qubit Clifford error 2e-2, and T1 "of the order of 40 µs". |
| Renger2025 | corrected | Superconducting qubit-resonator quantum processor with effective all-to-all connectivity | M. Renger, J. Verjauw, N. Wurz, et al. (65 authors) | 2026 | 2503.10903 | 10.1103/8x68-cx1w | Phys. Rev. Research 8, 013148 (2026-02-11) | A, S, C, P | A journal version exists, which the list missed. Crossref's "has-preprint" relation links it to arXiv:2503.10903. The arXiv title, "A Superconducting Qubit-Resonator Quantum Processor with Effective All-to-All Connectivity", starts with "A"; the journal title does not. The author count of 65 matches. |
| Vigneau2025 | corrected | Quantum Error Detection in Qubit-Resonator Star Architecture | F. Vigneau, S. Majumder, A. Rath, P. Parrado-Rodríguez, F. R. Fernandes Pereira, H.-S. Ku, F. Šimkovic IV, S. Pogorzalek, T. Jones, N. Wurz, M. Renger, J. Verjauw, P. Yang, W. Kindel, F. Deppe, J. Heinsoo | 2025 | 2503.12869 | 10.1103/8m33-wn4g | PRX Quantum 6, 040369 (2025-12-22) | A, C, P, F | Venue corrected to PRX Quantum. Author count corrected to 16. The arXiv metadata lists H.-S. Ku twice, which makes 17 entries; the APS page and Crossref list him once. Full text confirms the 2.00 µs cycle, the no-reset rule "s_n = +1 if d_n ⊕ d_{n-1} = 0", and "the \|2⟩ states are mostly classified as \|1⟩". |
| IQMclientDocs | confirmed | IQM client documentation, `iqm.iqm_client.models`, Release 36.0.0 | IQM Quantum Computers (IQM Finland Oy) | 2026 | none | none | Online docs, last updated 2026-10-06, accessed 2026-10-28 | W | docs.meetiqm.com redirects here. All quoted sentences are present except one: see flag 6. The operation list is measure, prx, cc_prx, reset, cz, move, barrier, delay, as the notes say. |
| IQMpulseDocs | confirmed | IQM pulse documentation: gates, Shelved_Measure_Constant, lru, prx, Reset_Conditional, Release 15.0.0 | IQM Quantum Computers (IQM Finland Oy) | 2026 | none | none | Online docs, iqm.pulse 15.0.0, last updated 2026-10-06 | W | All five pages fetched. Present: "surrounded by two prx_12 gates", acquisition type default `threshold`, the LRU sentence, LRU_F0G1 "driving the \|f0⟩ ↔ \|g1⟩ transition", and the three Reset_Conditional sentences. The `prx` page does not list prx_12, as the notes say. The pages show "Author name not set". |
| IQMpullaDocs | confirmed | IQM Pulla documentation: Custom Gates and Gate Implementations, Release 15.0.0 | IQM Quantum Computers (IQM Finland Oy) | 2026 | none | none | Online docs, IQM Pulla 15.0.0, last updated 2026-10-06 | W | The page does not mention drive frequencies or the 1-2 transition, as the notes say. |
| IQMpulseArticle2025 | confirmed | How IQM's pulse-level access is fuelling value for end-users | M. Bruce (IQM Quantum Computers) | 2025 | none | none | IQM website article, 28 Feb 2025, banner "IQM Pulla updated April 2026" | W | The ?p=5957 link redirects to iqm.tech/blog/how-iqms-pulse-level-access-is-fuelling-value-for-end-users/. The author box names Michael Bruce, Public Relations Manager. "drive frequencies and pulse amplitudes" is present. |
| AWSGarnetDyn2025 | confirmed | Experiment with dynamic circuits on IQM Garnet with Amazon Braket | T. Chen, S. Smart, Z. Mohammad | 2025 | none | none | AWS Quantum Technologies Blog, 26 Jun 2025 | W | The byline reads Tim Chen, Scott Smart, and Zia Mohammad. Present: "experimental capability", "conditional operations can only be applied to qubits in a local group", "only supported as in verbatim compilation". |
| AWSEmerald2025 | confirmed | Amazon Braket launches new 54-qubit superconducting quantum processor from IQM | Z. Mohammad, C. Panchalam Govindarajan, P. Komar, S. Seegerer | 2025 | none | none | AWS Quantum Technologies Blog, 21 Jul 2025 | W | Present: "Early characterization data shows median single-qubit gate fidelity of 99.93% and median two-qubit gate fidelity of 99.5%", and code using measure_ff, cc_prx and active reset. |

## Flags on the notes file

These refer to `lit-L6-scqutrit.md`.

1. §3.3, Wang2025. The citation pairs the arXiv title with the Phys. Rev. Applied
   reference. The published title is "High-E_J/E_C transmon qudits with up to 12 levels".
2. §4 table, Renger2025. The table lists it as an arXiv preprint, but it is published as
   Phys. Rev. Research 8, 013148 (2026), DOI 10.1103/8x68-cx1w, under a title without the
   leading "A".
3. §4 table, Vigneau2025. "17 authors" should read 16, and "APS journal" should read
   PRX Quantum 6, 040369 (2025).
4. §3.5, Haupt2025. "journal name not checked" resolves to Phys. Rev. Applied 25, 024079
   (2026). The publication year is 2026.
5. §3.1. Goss2024's article number is 101, which the notes say was not shown. Nguyen2024's
   article number is 7117.
6. §4.1. The notes quote `move` as "Star architecture only". The page actually says "MOVE is
   only available in quantum computers with the IQM Star architecture." The meaning is the
   same, but the quotation is not verbatim.
7. §1 item 4 cites Blok2021 and Goss2024 for readout durations of 0.5 to 1.5 μs. The notes'
   own §3.1 and §2 table say Blok2021 states no duration. I found no readout duration in the
   arXiv HTML of either paper. Goss2026 does state 500 ns. I did not check the Vora2026 or
   Hyyppa2024 durations, so I cannot say whether they support the range.
8. Vora2026. The key and year use the v5 date. The paper was first submitted in 2024.

Every number that appears both in the notes and in an abstract I read agrees:

- Morvan2021: 2.38e-3 and 0.82 on arXiv; 3.8e-3 and 0.85 in the published abstract.
- Goss2022: 97.3(1)% and 95.2(3)%.
- Luo2023: 89.3% and 95.5%.
- CerveraLierta2022: 76 ± 1%.
- Chen2023: 140 ns, 99.5% and 96.9%.
- Wang2025: e_f < 3e-3, 93.8%, E_J/E_C up to 325.
- Magnard2018: 0.2% in under 500 ns.
- Hyyppa2024: below 3.0e-5 at 6.25 ns.
- Brock2025: gains of 1.82 and 1.87.
- Vigneau2025: 0.25(2)% to 0.91(3)% per cycle.
- Renger2025: GHZ fidelity 0.86.
- Abdurakhimov2024: median two-qubit fidelity 99.5%.
- Goss2026: 15 qutrits.
- LiuBJ2026: postselected T1 above 500 μs.
- Vora2026: 40 ns inference.
- Blok2021: F_avg = 0.568.

## Process note

My first 27 arXiv requests put the user's e-mail address in the HTTP User-Agent header. I
should not have sent it. All later requests used a neutral User-Agent.
