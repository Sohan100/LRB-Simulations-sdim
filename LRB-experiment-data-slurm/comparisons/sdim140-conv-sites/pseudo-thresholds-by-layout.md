# Pseudo-thresholds by check layout

p* +- one standard error (sequence bootstrap, 68%); `pending` where the bootstrap has not been run for that exact p*. Sources: the four `comparisons/pseudo-thresholds-*current.csv` hand-off files. Ratio = conventional sites / coherent relay; gap = (coherent relay - conventional sites) / SE, the two SEs added in quadrature (independent runs).

| code | noise | k | one ancilla | coherent relay | conventional sites | ratio | gap |
|---|---|---|---|---|---|---|---|
| qubit | SD6 | 1 | 0.0210 +- 0.0008 | 0.0184 +- 0.0004 | 0.0142 +- 0.0004 | 0.77 | 7.0 sigma |
| qubit | SD6 | 2 | 0.0151 +- 0.0005 | 0.0132 +- 0.0004 | 0.0100 +- 0.0003 | 0.76 | 6.8 sigma |
| qubit | SD6 | 3 | 0.0094 +- 0.0003 | 0.0082 +- 0.0002 | 0.0061 +- 0.0002 | 0.75 | 8.3 sigma |
| qubit | SD6 | 4 | 0.0072 +- 0.0002 | 0.0063 +- 0.0001 | 0.0047 +- 0.0001 | 0.75 | 8.5 sigma |
| qubit | SI1000 | 1 | 0.0112 +- 0.0002 | 0.0102 +- 0.0002 | 0.0093 +- 0.0001 | 0.91 | 3.3 sigma |
| qubit | SI1000 | 2 | 0.0078 +- 0.0002 | 0.0068 +- 0.0001 | 0.0064 +- 0.0001 | 0.94 | 2.2 sigma |
| qubit | SI1000 | 3 | 0.0049 +- 0.0001 | 0.0043 +- 0.0001 | 0.0040 +- 0.0001 | 0.91 | 3.5 sigma |
| qubit | SI1000 | 4 | 0.0040 +- 0.0001 | 0.0033 +- 0.0001 | 0.0031 +- 0.0001 | 0.94 | 2.7 sigma |
| qutrit | SD6 | 1 | 0.0399 +- 0.0015 | 0.0360 +- 0.0013 | 0.0283 +- 0.0007 | 0.79 | 5.2 sigma |
| qutrit | SD6 | 2 | 0.0269 +- 0.0007 | 0.0251 +- 0.0015 | 0.0212 +- 0.0003 | 0.84 | 2.5 sigma |
| qutrit | SD6 | 3 | 0.0181 +- 0.0006 | 0.0145 +- 0.0004 | 0.0122 +- 0.0003 | 0.84 | 4.3 sigma |
| qutrit | SD6 | 4 | 0.0155 +- 0.0004 | 0.0125 +- 0.0003 | 0.0103 +- 0.0001 | 0.83 | 6.8 sigma |
| qutrit | SI1000 | 1 | 0.0292 +- 0.0006 | 0.0288 +- 0.0006 | 0.0266 +- 0.0004 | 0.92 | 2.8 sigma |
| qutrit | SI1000 | 2 | 0.0239 +- 0.0005 | 0.0228 +- 0.0005 | 0.0209 +- 0.0003 | 0.92 | 3.2 sigma |
| qutrit | SI1000 | 3 | 0.0166 +- 0.0004 | 0.0150 +- 0.0003 | 0.0136 +- 0.0003 | 0.91 | 3.1 sigma |
| qutrit | SI1000 | 4 | 0.0132 +- 0.0002 | 0.0119 +- 0.0003 | 0.0108 +- 0.0001 | 0.91 | 3.7 sigma |
