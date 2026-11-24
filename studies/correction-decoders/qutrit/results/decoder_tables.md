#### p = 0.0002 (physical RB at p_RB = 0.001: r = 0.000741 per depth step)

| Strategy | r per depth step [95% CI] | acceptance per depth step | accepted P(correct) at m = 0 | accepted P(correct) at m = 22 | acceptance at m = 22 |
|---|---|---|---|---|---|
| LRB-D, every round (unif 1) | 7.7e-07 [7.4e-07, 7.9e-07] | 0.9779 | 0.999008 | 0.998991 | 0.6044 |
| LRB-D, every 2nd round (unif 2) | 1.3e-06 [1.3e-06, 1.3e-06] | 0.9819 | 0.999008 | 0.998979 | 0.6604 |
| LRB-D, every 4th round (unif 4) | 2.8e-06 [2.7e-06, 2.8e-06] | 0.984 | 0.999008 | 0.998946 | 0.6938 |
| LRB-D, final round only (const 1) | 9.8e-06 [8.7e-06, 1.1e-05] | 0.986 | 0.999008 | 0.998574 | 0.7237 |
| LRB-D, no ancilla checks (const 0) | 3.7e-07 [2.7e-07, 5e-07] | 0.9984 | 1 | 0.999972 | 0.9649 |
| LRB-D, final round X-part only | 2.2e-06 [2.1e-06, 2.3e-06] | 0.9781 | 0.999007 | 0.998963 | 0.6094 |
| Blip-forgiving detection (BFD) | 2.1e-06 [2e-06, 2.1e-06] | 0.9859 | 0.999008 | 0.998963 | 0.7218 |
| BFD, final round X-part only | 5.3e-06 [5.1e-06, 5.4e-06] | 0.9861 | 0.999007 | 0.998902 | 0.7297 |
| MLD, reject if confidence < 0.97 | 4.8e-05 [4.7e-05, 4.8e-05] | 0.9931 | 0.999003 | 0.99796 | 0.8505 |
| MLD, reject if confidence < 0.9 | 0.00014 [0.00014, 0.00015] | 0.9951 | 0.998971 | 0.995837 | 0.8894 |
| MLD, reject if confidence < 0.5 | 0.00092 [0.00089, 0.00096] | 0.9973 | 0.99719 | 0.977528 | 0.9435 |
| Full MLD correction (all rounds) | 0.00222 [0.00217, 0.00225] | 1 | 0.99719 | 0.95077 | 1 |
| Full MLD, every 4th round only | 0.00257 [0.00252, 0.0026] | 1 | 0.99719 | 0.943636 | 1 |
| Full MLD, final round only | 0.00301 [0.00295, 0.00304] | 1 | 0.99719 | 0.93317 | 1 |
| Full MLD, const 0 circuits | 0.00033 [0.00032, 0.00034] | 1 | 1 | 0.992621 | 1 |
| Causal hard-decision, lookahead 2 | 0.00224 [0.00219, 0.00227] | 1 | 0.99719 | 0.950197 | 1 |
| Causal hard-decision, lookahead 1 | 0.00227 [0.00222, 0.0023] | 1 | 0.99719 | 0.949557 | 1 |
| Causal hard-decision, lookahead 0 | 0.00272 [0.00267, 0.00276] | 1 | 0.99719 | 0.940618 | 1 |
| No decoding, no rejection | 0.0057 [0.00561, 0.00576] | 1 | 0.996925 | 0.88271 | 1 |

#### p = 0.0005 (physical RB at p_RB = 0.0025: r = 0.00188 per depth step)

| Strategy | r per depth step [95% CI] | acceptance per depth step | accepted P(correct) at m = 0 | accepted P(correct) at m = 22 | acceptance at m = 22 |
|---|---|---|---|---|---|
| LRB-D, every round (unif 1) | 4.9e-06 [4.8e-06, 5e-06] | 0.9457 | 0.997511 | 0.997407 | 0.284 |
| LRB-D, every 2nd round (unif 2) | 8.3e-06 [8.1e-06, 8.5e-06] | 0.9554 | 0.997511 | 0.99733 | 0.3545 |
| LRB-D, every 4th round (unif 4) | 1.8e-05 [1.7e-05, 1.8e-05] | 0.9605 | 0.997511 | 0.997113 | 0.4012 |
| LRB-D, final round only (const 1) | 6.6e-05 [5.8e-05, 7.5e-05] | 0.9656 | 0.997511 | 0.994305 | 0.4483 |
| LRB-D, no ancilla checks (const 0) | 2.3e-06 [1.7e-06, 3.2e-06] | 0.9959 | 1 | 0.999823 | 0.9147 |
| LRB-D, final round X-part only | 1.4e-05 [1.3e-05, 1.4e-05] | 0.9461 | 0.997507 | 0.997227 | 0.29 |
| Blip-forgiving detection (BFD) | 1.3e-05 [1.3e-05, 1.3e-05] | 0.9649 | 0.997511 | 0.997231 | 0.4406 |
| BFD, final round X-part only | 3.2e-05 [3.2e-05, 3.3e-05] | 0.9654 | 0.997507 | 0.996874 | 0.453 |
| MLD, reject if confidence < 0.97 | 0.00011 [0.00011, 0.00011] | 0.9794 | 0.997486 | 0.995039 | 0.6196 |
| MLD, reject if confidence < 0.9 | 0.00046 [0.00046, 0.00047] | 0.9868 | 0.997381 | 0.987397 | 0.7312 |
| MLD, reject if confidence < 0.5 | 0.00245 [0.00239, 0.00253] | 0.9931 | 0.993002 | 0.943008 | 0.8607 |
| Full MLD correction (all rounds) | 0.00569 [0.00559, 0.00577] | 1 | 0.993002 | 0.881067 | 1 |
| Full MLD, every 4th round only | 0.00686 [0.00676, 0.00694] | 1 | 0.993002 | 0.859465 | 1 |
| Full MLD, final round only | 0.00872 [0.00857, 0.00884] | 1 | 0.993002 | 0.821113 | 1 |
| Full MLD, const 0 circuits | 0.00083 [0.00081, 0.00086] | 1 | 1 | 0.981707 | 1 |
| Causal hard-decision, lookahead 2 | 0.00582 [0.00571, 0.0059] | 1 | 0.993002 | 0.878245 | 1 |
| Causal hard-decision, lookahead 1 | 0.00598 [0.00586, 0.00607] | 1 | 0.993002 | 0.87521 | 1 |
| Causal hard-decision, lookahead 0 | 0.00719 [0.00708, 0.00728] | 1 | 0.993002 | 0.853952 | 1 |
| No decoding, no rejection | 0.0142 [0.014, 0.0143] | 1 | 0.99237 | 0.744044 | 1 |

#### p = 0.001 (physical RB at p_RB = 0.005: r = 0.00376 per depth step)

| Strategy | r per depth step [95% CI] | acceptance per depth step | accepted P(correct) at m = 0 | accepted P(correct) at m = 22 | acceptance at m = 22 |
|---|---|---|---|---|---|
| LRB-D, every round (unif 1) | 2e-05 [2e-05, 2e-05] | 0.8944 | 0.994994 | 0.994569 | 0.08069 |
| LRB-D, every 2nd round (unif 2) | 3.4e-05 [3.3e-05, 3.5e-05] | 0.9128 | 0.994994 | 0.994251 | 0.1258 |
| LRB-D, every 4th round (unif 4) | 7.4e-05 [7.3e-05, 7.6e-05] | 0.9226 | 0.994994 | 0.99333 | 0.1614 |
| LRB-D, final round only (const 1) | 0.0003 [0.00025, 0.00035] | 0.9336 | 0.994994 | 0.977754 | 0.2066 |
| LRB-D, no ancilla checks (const 0) | 9.1e-06 [6.8e-06, 1.3e-05] | 0.9919 | 1 | 0.999268 | 0.8377 |
| LRB-D, final round X-part only | 5.7e-05 [5.5e-05, 5.9e-05] | 0.8952 | 0.994977 | 0.993832 | 0.0842 |
| Blip-forgiving detection (BFD) | 5.3e-05 [5.2e-05, 5.4e-05] | 0.9303 | 0.994994 | 0.993867 | 0.1914 |
| BFD, final round X-part only | 0.00013 [0.00013, 0.00013] | 0.9315 | 0.994977 | 0.992449 | 0.2025 |
| MLD, reject if confidence < 0.97 | 0.00024 [0.00024, 0.00025] | 0.9522 | 0.994968 | 0.989954 | 0.3262 |
| MLD, reject if confidence < 0.9 | 0.00094 [0.00093, 0.00095] | 0.9687 | 0.994695 | 0.975029 | 0.4773 |
| MLD, reject if confidence < 0.5 | 0.00518 [0.00508, 0.0053] | 0.9853 | 0.985911 | 0.887473 | 0.724 |
| Full MLD correction (all rounds) | 0.0118 [0.0116, 0.012] | 1 | 0.985911 | 0.776407 | 1 |
| Full MLD, every 4th round only | 0.015 [0.0148, 0.0152] | 1 | 0.985911 | 0.730309 | 1 |
| Full MLD, final round only | 0.0201 [0.0197, 0.0205] | 1 | 0.985911 | 0.654096 | 1 |
| Full MLD, const 0 circuits | 0.00167 [0.00161, 0.00172] | 1 | 1 | 0.963917 | 1 |
| Causal hard-decision, lookahead 2 | 0.0123 [0.0121, 0.0125] | 1 | 0.985911 | 0.768216 | 1 |
| Causal hard-decision, lookahead 1 | 0.0129 [0.0126, 0.013] | 1 | 0.985911 | 0.760043 | 1 |
| Causal hard-decision, lookahead 0 | 0.0155 [0.0153, 0.0157] | 1 | 0.985911 | 0.7239 | 1 |
| No decoding, no rejection | 0.028 [0.0276, 0.0283] | 1 | 0.984844 | 0.5875 | 1 |

#### p = 0.002 (physical RB at p_RB = 0.01: r = 0.00752 per depth step)

| Strategy | r per depth step [95% CI] | acceptance per depth step | accepted P(correct) at m = 0 | accepted P(correct) at m = 22 | acceptance at m = 22 |
|---|---|---|---|---|---|
| LRB-D, every round (unif 1) | 8.4e-05 [8.2e-05, 8.5e-05] | 0.7999 | 0.989875 | 0.988109 | 0.006518 |
| LRB-D, every 2nd round (unif 2) | 0.00014 [0.00014, 0.00015] | 0.8333 | 0.989875 | 0.986767 | 0.01588 |
| LRB-D, every 4th round (unif 4) | 0.00033 [0.00032, 0.00033] | 0.8516 | 0.989875 | 0.98259 | 0.02631 |
| LRB-D, final round only (const 1) | 0.00137 [0.00113, 0.00174] | 0.879 | 0.989875 | 0.869281 | 0.05206 |
| LRB-D, no ancilla checks (const 0) | 3.6e-05 [2.7e-05, 5e-05] | 0.9841 | 1 | 0.996867 | 0.7052 |
| LRB-D, final round X-part only | 0.00024 [0.00023, 0.00025] | 0.8015 | 0.989807 | 0.98503 | 0.00712 |
| Blip-forgiving detection (BFD) | 0.00022 [0.00021, 0.00022] | 0.8635 | 0.989875 | 0.985284 | 0.03484 |
| BFD, final round X-part only | 0.00054 [0.00053, 0.00055] | 0.8658 | 0.989807 | 0.979473 | 0.03919 |
| MLD, reject if confidence < 0.97 | 0.00032 [0.00031, 0.00033] | 0.879 | 0.989778 | 0.983291 | 0.05435 |
| MLD, reject if confidence < 0.9 | 0.00173 [0.00171, 0.00174] | 0.9221 | 0.988901 | 0.954488 | 0.1547 |
| MLD, reject if confidence < 0.5 | 0.0103 [0.0101, 0.0105] | 0.964 | 0.971773 | 0.796527 | 0.4505 |
| Full MLD correction (all rounds) | 0.0256 [0.0252, 0.0259] | 1 | 0.971773 | 0.607637 | 1 |
| Full MLD, every 4th round only | 0.0339 [0.0335, 0.0343] | 1 | 0.971773 | 0.535905 | 1 |
| Full MLD, final round only | 0.0455 [0.0446, 0.0463] | 1 | 0.971773 | 0.456386 | 1 |
| Full MLD, const 0 circuits | 0.00333 [0.00322, 0.00343] | 1 | 1 | 0.929795 | 1 |
| Causal hard-decision, lookahead 2 | 0.0272 [0.0267, 0.0275] | 1 | 0.971773 | 0.590673 | 1 |
| Causal hard-decision, lookahead 1 | 0.0288 [0.0283, 0.0292] | 1 | 0.971773 | 0.576407 | 1 |
| Causal hard-decision, lookahead 0 | 0.0346 [0.0342, 0.0349] | 1 | 0.971773 | 0.531805 | 1 |
| No decoding, no rejection | 0.0549 [0.0541, 0.0555] | 1 | 0.969874 | 0.428999 | 1 |

#### p = 0.005 (physical RB at p_RB = 0.025: r = 0.0186 per depth step)

| Strategy | r per depth step [95% CI] | acceptance per depth step | accepted P(correct) at m = 0 | accepted P(correct) at m = 22 | acceptance at m = 22 |
|---|---|---|---|---|---|
| LRB-D, every round (unif 1) | 0.0006 [0.00059, 0.00061] | 0.5725 | 0.973833 | 0.961411 | 3.47e-06 |
| LRB-D, every 2nd round (unif 2) | 0.00107 [0.00104, 0.00109] | 0.6348 | 0.973833 | 0.951541 | 3.28e-05 |
| LRB-D, every 4th round (unif 4) | 0.00278 [0.00272, 0.00283] | 0.6722 | 0.973833 | 0.914076 | 0.000125 |
| LRB-D, final round only (const 1) | 0.0337 [0.0285, 0.0404] | 0.8331 | 0.973833 | 0.3642 | 0.01279 |
| LRB-D, no ancilla checks (const 0) | 0.00022 [0.00016, 0.00031] | 0.9626 | 1 | 0.976545 | 0.4364 |
| LRB-D, final round X-part only | 0.00172 [0.00167, 0.00179] | 0.5761 | 0.973397 | 0.940149 | 4.45e-06 |
| Blip-forgiving detection (BFD) | 0.00154 [0.00151, 0.00157] | 0.6835 | 0.973833 | 0.942659 | 0.000171 |
| BFD, final round X-part only | 0.00388 [0.00382, 0.00397] | 0.6894 | 0.973397 | 0.904473 | 0.00024 |
| MLD, reject if confidence < 0.97 | n/a [n/a, n/a] | 0.246 | 0.973833 | n/a | 0 |
| MLD, reject if confidence < 0.9 | 0.00291 [0.00284, 0.00299] | 0.7241 | 0.973105 | 0.928001 | 0.000681 |
| MLD, reject if confidence < 0.5 | 0.0195 [0.019, 0.0201] | 0.8509 | 0.931504 | 0.665523 | 0.0305 |
| Full MLD correction (all rounds) | 0.076 [0.0751, 0.0767] | 1 | 0.929221 | 0.377161 | 1 |
| Full MLD, every 4th round only | 0.101 [0.0996, 0.101] | 1 | 0.929221 | 0.349655 | 1 |
| Full MLD, final round only | 0.123 [0.121, 0.125] | 1 | 0.929221 | 0.339183 | 1 |
| Full MLD, const 0 circuits | 0.00829 [0.00801, 0.00855] | 1 | 1 | 0.83815 | 1 |
| Causal hard-decision, lookahead 2 | 0.0828 [0.0818, 0.0836] | 1 | 0.929221 | 0.366199 | 1 |
| Causal hard-decision, lookahead 1 | 0.0879 [0.0868, 0.0888] | 1 | 0.929221 | 0.360341 | 1 |
| Causal hard-decision, lookahead 0 | 0.103 [0.102, 0.104] | 1 | 0.929221 | 0.348301 | 1 |
| No decoding, no rejection | 0.129 [0.127, 0.13] | 1 | 0.927128 | 0.338928 | 1 |


#### Check policy at p = 0.001

| Policy (rounds used) | LRB-D: r per step | LRB-D: acceptance per step | LRB-D: curvature c2 | BFD on the same rounds: r | BFD: acceptance per step | LRB-D / BFD acceptance at m = 22 |
|---|---|---|---|---|---|---|
| `unif1` | 2e-05 | 0.894 | -1.1e-07 | 5.3e-05 | 0.93 | 0.0807 / 0.191 |
| `unif2` | 3.4e-05 | 0.913 | -8.1e-08 | 0.0001 | 0.931 | 0.126 / 0.195 |
| `unif3` | 5.7e-05 | 0.919 | -2e-07 | 0.00016 | 0.932 | 0.148 / 0.197 |
| `unif4` | 7.4e-05 | 0.923 | 5.6e-07 | 0.0002 | 0.932 | 0.161 / 0.199 |
| `unif6` | 0.00011 | 0.926 | -1.6e-06 | 0.00025 | 0.932 | 0.176 / 0.201 |
| `unif11` | 0.0002 | 0.929 | 5.4e-06 | 0.0003 | 0.933 | 0.184 / 0.204 |
| `unif22` | 0.0003 | 0.933 | -2.6e-05 | 0.0003 | 0.934 | 0.196 / 0.207 |
| `const1` | 0.0003 | 0.934 | -6.8e-05 | 0.0003 | 0.934 | 0.207 / 0.207 |
| `const2` | 0.00017 | 0.932 | -2.1e-05 | 0.0003 | 0.934 | 0.193 / 0.207 |
| `const3` | 0.00013 | 0.929 | -1.2e-05 | 0.00024 | 0.933 | 0.184 / 0.203 |
| `const4` | 0.00011 | 0.928 | -6.7e-06 | 0.0002 | 0.932 | 0.176 / 0.201 |
| `const11` | 7.5e-05 | 0.91 | 1.9e-05 | 0.00014 | 0.931 | 0.131 / 0.196 |
| `const22` | 4.5e-05 | 0.916 | 5.8e-05 | 9.8e-05 | 0.932 | 0.0841 / 0.192 |

#### Check policy at p = 0.002

| Policy (rounds used) | LRB-D: r per step | LRB-D: acceptance per step | LRB-D: curvature c2 | BFD on the same rounds: r | BFD: acceptance per step | LRB-D / BFD acceptance at m = 22 |
|---|---|---|---|---|---|---|
| `unif1` | 8.4e-05 | 0.8 | -5.2e-07 | 0.00022 | 0.863 | 0.00652 / 0.0348 |
| `unif2` | 0.00014 | 0.833 | -4.1e-07 | 0.00044 | 0.867 | 0.0159 / 0.0377 |
| `unif3` | 0.00024 | 0.846 | -8.7e-07 | 0.00071 | 0.868 | 0.0222 / 0.0393 |
| `unif4` | 0.00033 | 0.852 | 2.4e-06 | 0.00092 | 0.869 | 0.0263 / 0.0405 |
| `unif6` | 0.00051 | 0.858 | -8.8e-06 | 0.0012 | 0.872 | 0.0315 / 0.0428 |
| `unif11` | 0.001 | 0.865 | 2.6e-05 | 0.0014 | 0.877 | 0.0352 / 0.0476 |
| `unif22` | 0.0014 | 0.876 | -0.00027 | 0.0014 | 0.879 | 0.0434 / 0.0521 |
| `const1` | 0.0014 | 0.879 | -0.00065 | 0.0014 | 0.879 | 0.0521 / 0.0521 |
| `const2` | 0.0008 | 0.87 | -0.00014 | 0.0014 | 0.879 | 0.0394 / 0.0521 |
| `const3` | 0.00059 | 0.865 | -7e-05 | 0.0011 | 0.874 | 0.0349 / 0.0452 |
| `const4` | 0.00048 | 0.862 | -3.7e-05 | 0.00093 | 0.872 | 0.0317 / 0.0429 |
| `const11` | 0.00034 | 0.829 | 0.0001 | 0.00065 | 0.867 | 0.0173 / 0.0381 |
| `const22` | 0.00019 | 0.843 | 0.00033 | 0.00041 | 0.872 | 0.00709 / 0.0351 |

