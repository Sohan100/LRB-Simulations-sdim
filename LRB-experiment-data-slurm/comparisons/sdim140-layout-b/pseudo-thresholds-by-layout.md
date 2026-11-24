# Pseudo-thresholds by check layout (the two-ancilla layout replaces the conventional sites)

p* +- one standard error (sequence bootstrap, 68%); `pending` where the bootstrap has not been run for that exact p*. Sources: the `comparisons/pseudo-thresholds-*current.csv` hand-off files. 2/1 = two ancillas / one ancilla; 2/conv = two ancillas / previous conventional sites; gap = (one ancilla - two ancillas) / SE, the two SEs added in quadrature (independent runs). Two-ancilla rows marked provisional still wait for their extension or refinement.

| code | noise | k | one ancilla | previous: conventional sites | two ancillas | 2/1 | 2/conv | gap | status |
|---|---|---|---|---|---|---|---|---|---|
| qubit | SD6 | 1 | 0.0210 +- 0.0008 | 0.0142 +- 0.0004 | 0.0085 +- 0.0002 | 0.41 | 0.60 | 14.7 sigma | final |
| qubit | SD6 | 2 | 0.0151 +- 0.0005 | 0.0100 +- 0.0003 | 0.0058 +- 0.0001 | 0.38 | 0.58 | 18.3 sigma | final |
| qubit | SD6 | 3 | 0.0094 +- 0.0003 | 0.0061 +- 0.0002 | 0.0035 +- 0.0001 | 0.37 | 0.57 | 18.0 sigma | final |
| qubit | SD6 | 4 | 0.0072 +- 0.0002 | 0.0047 +- 0.0001 | 0.0027 +- 0.0000 | 0.37 | 0.58 | 17.9 sigma | final |
| qubit | SI1000 | 1 | 0.0112 +- 0.0002 | 0.0093 +- 0.0001 | 0.0051 +- 0.0002 | 0.46 | 0.55 | 23.2 sigma | final |
| qubit | SI1000 | 2 | 0.0078 +- 0.0002 | 0.0064 +- 0.0001 | 0.0031 +- 0.0001 | 0.40 | 0.49 | 28.0 sigma | final |
| qubit | SI1000 | 3 | 0.0049 +- 0.0001 | 0.0040 +- 0.0001 | 0.0020 +- 0.0000 | 0.40 | 0.50 | 30.9 sigma | final |
| qubit | SI1000 | 4 | 0.0040 +- 0.0001 | 0.0031 +- 0.0001 | 0.0015 +- 0.0000 | 0.38 | 0.49 | 28.7 sigma | final |
| qutrit | SD6 | 1 | 0.0399 +- 0.0015 | 0.0283 +- 0.0007 | 0.0206 +- 0.0004 | 0.52 | 0.73 | 12.7 sigma | final |
| qutrit | SD6 | 2 | 0.0269 +- 0.0007 | 0.0212 +- 0.0003 | 0.0147 +- 0.0003 | 0.55 | 0.69 | 15.6 sigma | final |
| qutrit | SD6 | 3 | 0.0181 +- 0.0006 | 0.0122 +- 0.0003 | 0.0088 +- 0.0002 | 0.49 | 0.72 | 15.1 sigma | final |
| qutrit | SD6 | 4 | 0.0155 +- 0.0004 | 0.0103 +- 0.0001 | 0.0074 +- 0.0002 | 0.48 | 0.72 | 16.6 sigma | final |
| qutrit | SI1000 | 1 | 0.0292 +- 0.0006 | 0.0266 +- 0.0004 | 0.0191 +- 0.0004 | 0.66 | 0.72 | 14.2 sigma | final |
| qutrit | SI1000 | 2 | 0.0239 +- 0.0005 | 0.0209 +- 0.0003 | 0.0139 +- 0.0002 | 0.58 | 0.66 | 18.3 sigma | final |
| qutrit | SI1000 | 3 | 0.0166 +- 0.0004 | 0.0136 +- 0.0003 | 0.0085 +- 0.0002 | 0.51 | 0.62 | 16.9 sigma | final |
| qutrit | SI1000 | 4 | 0.0132 +- 0.0002 | 0.0108 +- 0.0001 | 0.0064 +- 0.0001 | 0.49 | 0.59 | 28.3 sigma | final |
