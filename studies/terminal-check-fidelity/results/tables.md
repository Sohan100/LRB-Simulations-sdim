
### Table A. What the terminal check measures (exact)

| code | noise | p | logical pair, truth | coincidence part | F_cov | discard per pair | physical RB per pair | RB / logical | break-even depth | unif-1 per pair (noisy checks) | single-exp fit of const-0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qutrit | SI1000 | 0.001 | 2.8e-6 | 2.5e-6 | 1.12 | 1.0% | 7.6e-4 | 267 | 301 | 8.3e-6 | 3.1e-5 |
| qutrit | SI1000 | 0.002 | 1.1e-5 | 1.0e-5 | 1.12 | 1.9% | 1.5e-3 | 135 | 152 | 3.3e-5 | 1.2e-4 |
| qutrit | SI1000 | 0.005 | 7.2e-5 | 6.4e-5 | 1.12 | 4.7% | 3.8e-3 | 53 | 60 | 2.3e-4 | 7.4e-4 |
| qutrit | SI1000 | 0.01 | 2.9e-4 | 2.6e-4 | 1.12 | 9.2% | 7.6e-3 | 26 | 29 | 1.2e-3 | 2.8e-3 |
| qutrit | SD6 | 0.001 | 7.5e-5 | 6.8e-5 | 1.11 | 4.4% | 8.0e-3 | 106 | 118 | 8.3e-5 | 7.8e-4 |
| qutrit | SD6 | 0.002 | 3.0e-4 | 2.7e-4 | 1.11 | 8.6% | 1.6e-2 | 52 | 58 | 3.2e-4 | 3.0e-3 |
| qutrit | SD6 | 0.005 | 2.0e-3 | 1.8e-3 | 1.10 | 19.9% | 3.9e-2 | 20 | 22 | 1.4e-3 | 1.8e-2 |
| qutrit | SD6 | 0.01 | 8.1e-3 | 7.5e-3 | 1.09 | 34.9% | 7.6e-2 | 9 | 10 | 6.1e-3 | 5.3e-2 |
| qubit | SI1000 | 0.001 | 1.6e-6 | 1.3e-6 | 1.26 | 0.4% | 3.4e-4 | 214 | 270 | 2.0e-5 | 1.6e-5 |
| qubit | SI1000 | 0.002 | 6.3e-6 | 5.0e-6 | 1.26 | 0.8% | 6.9e-4 | 109 | 138 | 8.0e-5 | 6.2e-5 |
| qubit | SI1000 | 0.005 | 4.0e-5 | 3.1e-5 | 1.26 | 2.1% | 1.7e-3 | 43 | 54 | 5.8e-4 | 3.8e-4 |
| qubit | SI1000 | 0.01 | 1.6e-4 | 1.3e-4 | 1.26 | 4.1% | 3.4e-3 | 21 | 27 | 3.0e-3 | 1.4e-3 |
| qubit | SD6 | 0.001 | 4.0e-5 | 3.3e-5 | 1.21 | 2.0% | 3.4e-3 | 84 | 102 | 1.1e-4 | 3.9e-4 |
| qubit | SD6 | 0.002 | 1.6e-4 | 1.3e-4 | 1.21 | 4.0% | 6.7e-3 | 42 | 51 | 4.0e-4 | 1.5e-3 |
| qubit | SD6 | 0.005 | 1.0e-3 | 8.3e-4 | 1.21 | 9.5% | 1.7e-2 | 17 | 20 | 2.6e-3 | 8.4e-3 |
| qubit | SD6 | 0.01 | 4.0e-3 | 3.3e-3 | 1.20 | 17.6% | 3.3e-2 | 8 | 10 | 1.1e-2 | 2.6e-2 |

### Table B. Estimators on exact data (infinite shots and sequences), noisy SPAM

| code | noise | closing | depth grid | single exp / truth | direct c2 / truth | free fit / truth | block fit / coincidence | block fit x F_cov / truth |
|---|---|---|---|---|---|---|---|---|
| qubit | SD6 | compiled | designed | 10 to 42 | 0.86 to 0.92 | 1.005 to 1.019 | 1.000 to 1.006 | 1.000 to 1.006 |
| qubit | SD6 | mirror | designed | 5 to 21 | 0.86 to 0.91 | 0.993 to 1.001 | 1.000 to 1.001 | 1.000 to 1.001 |
| qubit | SD6 | mirror | production | 9 to 13 | 0.84 to 0.91 | 0.994 to 0.998 | 0.999 to 1.003 | 0.999 to 1.003 |
| qubit | SI1000 | compiled | designed | 73 to 136 | 0.70 to 0.89 | 1.004 to 1.008 | 1.000 to 1.001 | 1.000 to 1.001 |
| qubit | SI1000 | mirror | designed | 37 to 106 | 0.68 to 0.89 | 1.001 to 1.002 | 1.000 to 1.000 | 1.000 to 1.000 |
| qubit | SI1000 | mirror | production | 32 to 37 | 0.73 to 0.91 | 1.001 to 1.005 | 1.000 to 1.000 | 1.000 to 1.000 |
| qutrit | SD6 | compiled | designed | 7 to 21 | 1.11 to 1.30 | 1.010 to 1.018 | 0.998 to 1.000 | 0.998 to 1.000 |
| qutrit | SD6 | mirror | designed | 4 to 11 | 1.10 to 1.17 | 1.000 to 1.002 | 0.999 to 0.999 | 0.999 to 0.999 |
| qutrit | SD6 | mirror | production | 8 to 13 | 1.12 to 1.12 | 1.000 to 1.002 | 0.998 to 0.999 | 0.998 to 0.999 |
| qutrit | SI1000 | compiled | designed | 46 to 126 | 1.16 to 1.55 | 1.009 to 1.021 | 0.999 to 1.000 | 0.999 to 1.000 |
| qutrit | SI1000 | mirror | designed | 23 to 64 | 1.17 to 1.52 | 1.001 to 1.003 | 1.000 to 1.000 | 1.000 to 1.000 |
| qutrit | SI1000 | mirror | production | 25 to 33 | 1.02 to 1.50 | 1.001 to 1.002 | 1.000 to 1.000 | 1.000 to 1.000 |

### Table C. Standard error at finite shots (200 synthetic experiments per row), mirror closing, noisy SPAM

| code | p | sequences | shots per circuit | depths (max) | total shots | block fit: mean / ref | block fit: SE | bootstrap SE | free fit: mean / truth | free fit: SE |
|---|---|---|---|---|---|---|---|---|---|---|
| qubit | 0.002 | 10 | 100 | 9 (119) | 9.0e3 | 1.015 | 6.8% |  | 13.62 | 2824% |
| qubit | 0.002 | 10 | 1000 | 9 (119) | 9.0e4 | 0.999 | 4.2% |  | 1.14 | 314% |
| qubit | 0.002 | 10 | 10000 | 9 (119) | 9.0e5 | 0.999 | 3.5% |  | 0.92 | 88% |
| qubit | 0.002 | 30 | 100 | 9 (119) | 2.7e4 | 1.000 | 4.0% |  | 1.80 | 640% |
| qubit | 0.002 | 30 | 1000 | 9 (119) | 2.7e5 | 0.999 | 2.1% | 2.3% | 1.08 | 170% |
| qubit | 0.002 | 30 | 10000 | 9 (119) | 2.7e6 | 0.998 | 2.1% |  | 1.03 | 50% |
| qubit | 0.002 | 100 | 100 | 9 (119) | 9.0e4 | 0.998 | 2.1% |  | 0.90 | 285% |
| qubit | 0.002 | 100 | 1000 | 9 (119) | 9.0e5 | 1.000 | 1.2% |  | 1.04 | 95% |
| qubit | 0.002 | 100 | 10000 | 9 (119) | 9.0e6 | 1.000 | 1.1% |  | 1.02 | 30% |
| qubit | 0.005 | 10 | 100 | 9 (48) | 9.0e3 | 0.998 | 9.4% |  | 1.27 | 580% |
| qubit | 0.005 | 10 | 1000 | 9 (48) | 9.0e4 | 0.996 | 5.6% |  | 1.13 | 173% |
| qubit | 0.005 | 10 | 10000 | 9 (48) | 9.0e5 | 1.002 | 5.4% |  | 0.94 | 58% |
| qubit | 0.005 | 30 | 100 | 9 (48) | 2.7e4 | 1.007 | 5.4% |  | 1.09 | 288% |
| qubit | 0.005 | 30 | 1000 | 9 (48) | 2.7e5 | 0.998 | 3.3% | 3.2% | 1.01 | 93% |
| qubit | 0.005 | 30 | 10000 | 9 (48) | 2.7e6 | 1.001 | 3.1% |  | 0.96 | 30% |
| qubit | 0.005 | 100 | 100 | 9 (48) | 9.0e4 | 1.001 | 3.2% |  | 1.17 | 174% |
| qubit | 0.005 | 100 | 1000 | 9 (48) | 9.0e5 | 0.998 | 1.9% |  | 0.96 | 54% |
| qubit | 0.005 | 100 | 10000 | 9 (48) | 9.0e6 | 1.000 | 1.7% |  | 1.00 | 17% |
| qubit | 0.01 | 10 | 100 | 8 (25) | 8.0e3 | 0.997 | 12.4% |  | 0.95 | 414% |
| qubit | 0.01 | 10 | 1000 | 8 (25) | 8.0e4 | 1.007 | 7.9% |  | 0.97 | 131% |
| qubit | 0.01 | 10 | 10000 | 8 (25) | 8.0e5 | 1.006 | 7.6% |  | 1.00 | 43% |
| qubit | 0.01 | 30 | 100 | 8 (25) | 2.4e4 | 1.001 | 8.0% |  | 1.06 | 252% |
| qubit | 0.01 | 30 | 1000 | 8 (25) | 2.4e5 | 1.001 | 4.7% | 4.6% | 0.99 | 80% |
| qubit | 0.01 | 30 | 10000 | 8 (25) | 2.4e6 | 1.003 | 4.4% |  | 0.97 | 25% |
| qubit | 0.01 | 100 | 100 | 8 (25) | 8.0e4 | 1.001 | 3.9% |  | 1.10 | 135% |
| qubit | 0.01 | 100 | 1000 | 8 (25) | 8.0e5 | 1.001 | 2.6% |  | 0.98 | 44% |
| qubit | 0.01 | 100 | 10000 | 8 (25) | 8.0e6 | 1.000 | 2.3% |  | 1.00 | 14% |
| qutrit | 0.002 | 10 | 100 | 9 (52) | 9.0e3 | 1.047 | 8.7% |  | 19.25 | 6110% |
| qutrit | 0.002 | 10 | 1000 | 9 (52) | 9.0e4 | 1.000 | 4.1% |  | 0.62 | 249% |
| qutrit | 0.002 | 10 | 10000 | 9 (52) | 9.0e5 | 0.999 | 3.6% |  | 0.92 | 74% |
| qutrit | 0.002 | 30 | 100 | 9 (52) | 2.7e4 | 1.008 | 4.2% |  | 2.99 | 1192% |
| qutrit | 0.002 | 30 | 1000 | 9 (52) | 2.7e5 | 1.000 | 2.2% | 2.3% | 0.77 | 132% |
| qutrit | 0.002 | 30 | 10000 | 9 (52) | 2.7e6 | 1.000 | 2.1% |  | 0.99 | 42% |
| qutrit | 0.002 | 100 | 100 | 9 (52) | 9.0e4 | 1.002 | 2.1% |  | 1.01 | 263% |
| qutrit | 0.002 | 100 | 1000 | 9 (52) | 9.0e5 | 1.000 | 1.3% |  | 0.97 | 76% |
| qutrit | 0.002 | 100 | 10000 | 9 (52) | 9.0e6 | 1.000 | 1.1% |  | 0.99 | 23% |
| qutrit | 0.005 | 10 | 100 | 8 (21) | 8.0e3 | 1.016 | 10.3% |  | 1.62 | 1405% |
| qutrit | 0.005 | 10 | 1000 | 8 (21) | 8.0e4 | 1.001 | 5.6% |  | 1.03 | 128% |
| qutrit | 0.005 | 10 | 10000 | 8 (21) | 8.0e5 | 0.998 | 5.3% |  | 0.98 | 44% |
| qutrit | 0.005 | 30 | 100 | 8 (21) | 2.4e4 | 0.998 | 5.5% |  | 1.28 | 257% |
| qutrit | 0.005 | 30 | 1000 | 8 (21) | 2.4e5 | 1.003 | 3.4% | 3.3% | 1.07 | 74% |
| qutrit | 0.005 | 30 | 10000 | 8 (21) | 2.4e6 | 0.998 | 3.6% |  | 1.02 | 26% |
| qutrit | 0.005 | 100 | 100 | 8 (21) | 8.0e4 | 0.999 | 2.8% |  | 0.99 | 136% |
| qutrit | 0.005 | 100 | 1000 | 8 (21) | 8.0e5 | 1.000 | 1.7% |  | 1.01 | 41% |
| qutrit | 0.005 | 100 | 10000 | 8 (21) | 8.0e6 | 1.000 | 1.8% |  | 0.99 | 13% |
| qutrit | 0.01 | 10 | 100 | 8 (11) | 8.0e3 | 1.016 | 14.1% |  | 0.86 | 363% |
| qutrit | 0.01 | 10 | 1000 | 8 (11) | 8.0e4 | 1.001 | 8.2% |  | 1.01 | 105% |
| qutrit | 0.01 | 10 | 10000 | 8 (11) | 8.0e5 | 1.003 | 7.3% |  | 1.00 | 33% |
| qutrit | 0.01 | 30 | 100 | 8 (11) | 2.4e4 | 1.002 | 7.6% |  | 0.93 | 190% |
| qutrit | 0.01 | 30 | 1000 | 8 (11) | 2.4e5 | 1.002 | 4.7% | 4.6% | 1.05 | 61% |
| qutrit | 0.01 | 30 | 10000 | 8 (11) | 2.4e6 | 1.000 | 4.4% |  | 0.99 | 22% |
| qutrit | 0.01 | 100 | 100 | 8 (11) | 8.0e4 | 1.001 | 4.0% |  | 1.07 | 107% |
| qutrit | 0.01 | 100 | 1000 | 8 (11) | 8.0e5 | 0.999 | 2.7% |  | 0.97 | 35% |
| qutrit | 0.01 | 100 | 10000 | 8 (11) | 8.0e6 | 1.000 | 2.6% |  | 1.00 | 11% |

### Table D. Design variants at p = 0.005 (SI1000 unless noted), 30 sequences, 1000 shots per circuit

| code | variant | depths (max) | total shots | block fit: mean / ref | block fit: SE | free fit: SE |
|---|---|---|---|---|---|---|
| qubit | baseline: mirror, nested prefixes, designed grid | 9 (48) | 2.7e5 | 0.998 | 3.3% | 93% |
| qutrit | baseline: mirror, nested prefixes, designed grid | 8 (21) | 2.4e5 | 1.003 | 3.4% | 74% |
| qubit | independent sequences at every depth | 9 (48) | 2.7e5 | 1.001 | 1.9% | 101% |
| qubit | compiled recovery Clifford (x = number of Cliffords) | 9 (96) | 2.7e5 | 1.003 | 2.7% | 212% |
| qubit | production depths 0, 2, ..., 22 | 9 (22) | 2.7e5 | 0.998 | 4.4% | 105% |
| qubit | designed grid, reach 0.5 | 8 (24) | 2.4e5 | 1.001 | 4.8% | 104% |
| qubit | designed grid, reach 2 | 9 (96) | 2.7e5 | 1.005 | 2.7% | 114% |
| qubit | readout flip 20 p instead of 5 p | 9 (48) | 2.7e5 | 1.000 | 3.6% | 254% |
| qubit | SD6, p = 0.002 | 8 (25) | 2.4e5 | 1.002 | 3.7% | 33% |
| qutrit | independent sequences at every depth | 8 (21) | 2.4e5 | 0.998 | 2.2% | 76% |
| qutrit | compiled recovery Clifford (x = number of Cliffords) | 9 (42) | 2.7e5 | 1.001 | 2.5% | 172% |
| qutrit | production depths 0, 2, ..., 22 | 9 (22) | 2.7e5 | 1.002 | 3.3% | 74% |
| qutrit | designed grid, reach 0.5 | 8 (11) | 2.4e5 | 0.997 | 4.7% | 89% |
| qutrit | designed grid, reach 2 | 9 (42) | 2.7e5 | 1.002 | 2.5% | 88% |
| qutrit | readout flip 20 p instead of 5 p | 8 (21) | 2.4e5 | 0.998 | 4.0% | 200% |
| qutrit | SD6, p = 0.002 | 8 (12) | 2.4e5 | 1.009 | 4.9% | 49% |

### Table E. Production const-0 circuits re-sampled with the full record (30 sequences x 1M shots)

| code | noise | p | truth (all sequences) | block fit x F_cov | free fit | single-exponential fit | discard per pair: fit / truth |
|---|---|---|---|---|---|---|---|
| qutrit | SI1000 | 0.001 | 2.83e-6 | 2.83e-6 ± 9e-8 | 2.88e-6 ± 2e-7 | 3.11e-5 ± 1e-6 | 0.97% ± 0.02% / 0.96% |
| qutrit | SI1000 | 0.005 | 7.16e-5 | 7.17e-5 ± 2e-6 | 7.25e-5 ± 4e-6 | 7.54e-4 ± 3e-5 | 4.72% ± 0.07% / 4.71% |
| qutrit | SD6 | 0.001 | 7.52e-5 | 7.14e-5 ± 2e-6 | 6.98e-5 ± 3e-6 | 7.10e-4 ± 3e-5 | 4.33% ± 0.05% / 4.44% |
| qutrit | SD6 | 0.005 | 1.95e-3 | 1.78e-3 ± 7e-5 | 1.80e-3 ± 1e-4 | 1.63e-2 ± 8e-4 | 19.13% ± 0.34% / 19.89% |
| qubit | SI1000 | 0.001 | 1.58e-6 | 1.58e-6 ± 6e-8 | 1.50e-6 ± 1e-7 | 1.54e-5 ± 8e-7 | 0.42% ± 0.01% / 0.42% |
| qubit | SI1000 | 0.005 | 3.96e-5 | 3.95e-5 ± 2e-6 | 3.90e-5 ± 2e-6 | 3.78e-4 ± 2e-5 | 2.08% ± 0.05% / 2.07% |
| qubit | SD6 | 0.001 | 4.00e-5 | 3.80e-5 ± 2e-6 | 3.91e-5 ± 2e-6 | 3.78e-4 ± 2e-5 | 1.97% ± 0.04% / 2.02% |
| qubit | SD6 | 0.005 | 9.98e-4 | 9.55e-4 ± 5e-5 | 9.47e-4 ± 7e-5 | 8.08e-3 ± 5e-4 | 9.31% ± 0.22% / 9.49% |

### Table F. Correlated faults on the logical support {0, 2} (SI1000, p = 0.005)

| code | correlated-error probability per Clifford | truth / truth without it | block fit x F_cov / truth | free fit / truth | detected at 2 sigma, 30 seq x 1000 shots | detected, 30 seq x 10000 shots |
|---|---|---|---|---|---|---|
| qubit | 0 | 1.00 | 1.00 | 1.00 | 2% | 2% |
| qubit | 1e-05 | 1.07 | 0.94 | 1.00 | 0% | 2% |
| qubit | 3e-05 | 1.21 | 0.83 | 1.01 | 5% | 18% |
| qubit | 0.0001 | 1.69 | 0.61 | 1.01 | 8% | 72% |
| qubit | 0.0003 | 3.06 | 0.35 | 1.01 | 57% | 100% |
| qubit | 0.001 | 7.87 | 0.16 | 1.02 | 100% | 100% |
| qutrit | 0 | 1.00 | 1.00 | 1.00 | 2% | 0% |
| qutrit | 1e-05 | 1.03 | 0.97 | 1.00 | 0% | 2% |
| qutrit | 3e-05 | 1.08 | 0.93 | 1.01 | 0% | 8% |
| qutrit | 0.0001 | 1.27 | 0.80 | 1.02 | 8% | 17% |
| qutrit | 0.0003 | 1.82 | 0.57 | 1.03 | 17% | 93% |
| qutrit | 0.001 | 3.72 | 0.30 | 1.05 | 92% | 100% |

### Table G. Quasi-static drift of p between shots (log-normal, SI1000, mean p = 0.005)

| code | relative spread | truth / truth without drift | block fit x F_cov / truth | free fit / truth | observed / predicted accepted infidelity at mid depth |
|---|---|---|---|---|---|
| qubit | 0 | 1.00 | 1.00 | 1.00 | 1.004 |
| qubit | 0.1 | 1.01 | 0.98 | 1.36 | 1.011 |
| qubit | 0.2 | 1.04 | 0.92 | 2.32 | 1.033 |
| qubit | 0.3 | 1.09 | 0.83 | 3.66 | 1.066 |
| qubit | 0.5 | 1.27 | 0.60 | 6.17 | 1.152 |
| qutrit | 0 | 1.00 | 1.00 | 1.00 | 1.003 |
| qutrit | 0.1 | 1.01 | 0.98 | 1.19 | 1.010 |
| qutrit | 0.2 | 1.04 | 0.93 | 1.69 | 1.028 |
| qutrit | 0.3 | 1.09 | 0.85 | 2.39 | 1.058 |
| qutrit | 0.5 | 1.26 | 0.63 | 3.73 | 1.136 |
