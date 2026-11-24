"""Map literature transmon-qutrit numbers onto the SI1000 parameter p used in the brief.
Idle: Lindblad qutrit (sequential T1 decay 1->0, 2->1; diagonal dephasing matched to pairwise T2).
Process fidelity F_pro = (1/d^2) sum_ij [E(|i><j|)]_ij. After Weyl twirling, identity prob = F_pro,
so an SI1000 N1 location with total prob q has F_pro = 1 - q.
"""
import numpy as np
from scipy.linalg import expm
d=3
def lindblad_super(T1_10, T1_21, T2_01, T2_12, T2_02):
    g10, g21 = 1/T1_10, 1/T1_21
    G = [0.0, g10, g21]                              # total decay rate out of each level
    gp = {}
    for (i,j,T2) in ((0,1,T2_01),(1,2,T2_12),(0,2,T2_02)):
        gp[(i,j)] = max(1/T2 - (G[i]+G[j])/2, 0.0)  # pure-dephasing rate of coherence ij
    # independent diagonal projector noise: gp_ij = (a_i+a_j)/2
    A = np.array([[1,1,0],[0,1,1],[1,0,1]])/2.0
    a = np.linalg.solve(A, np.array([gp[(0,1)],gp[(1,2)],gp[(0,2)]]))
    a = np.clip(a,0,None)
    Ls=[]
    def ket(k): v=np.zeros((d,1)); v[k]=1; return v
    Ls.append(np.sqrt(g10)*ket(0)@ket(1).T)
    Ls.append(np.sqrt(g21)*ket(1)@ket(2).T)
    for k in range(3): Ls.append(np.sqrt(a[k])*ket(k)@ket(k).T)
    I=np.eye(d); S=np.zeros((d*d,d*d),complex)
    for L in Ls:   # column-stacking vec: vec(AXB)=(B^T kron A)vec(X)
        LdL=L.conj().T@L
        S+= np.kron(L.conj(),L) - 0.5*np.kron(I,LdL) - 0.5*np.kron(LdL.T,I)
    return S
def F_pro(S,t):
    E=expm(S*t); tot=0
    for i in range(d):
        for j in range(d):
            X=np.zeros((d,d)); X[i,j]=1
            Y=(E@X.reshape(-1,order='F')).reshape(d,d,order='F')
            tot+=Y[i,j]
    return (tot/d**2).real
sets = {
 "Blok2021 avg (echo T2)":      dict(T1_10=56.0, T1_21=34.8, T2_01=61.2, T2_12=28.0, T2_02=36.6),
 "Blok2021 avg (Ramsey T2*)":   dict(T1_10=56.0, T1_21=34.8, T2_01=39.0, T2_12=14.0, T2_02=14.8),
 "Goss2024 avg (echo, T2_02~T2_12)": dict(T1_10=57.0, T1_21=34.3, T2_01=70.7, T2_12=32.3, T2_02=32.3),
 "Liu2023 single qudit (Ramsey)": dict(T1_10=180., T1_21=101./1.0, T2_01=76., T2_12=37., T2_02=37.),
 "Luo2023 during gate (Q1)":    dict(T1_10=27.1, T1_21=13.9, T2_01=16.5, T2_12=12.3, T2_02=11.64),
}
print("Idle process infidelity e(t) and SI1000-equivalent p = e/2 (measurement-layer idle is N1 with prob 2p)")
for name,s in sets.items():
    S=lindblad_super(**s)
    row=[]
    for t in (0.14,0.5,1.0,1.5,2.2):
        e=1-F_pro(S,t); row.append(f"t={t}us: e={e:.4f} p={e/2:.4f}")
    print(f"  {name}:\n    "+" | ".join(row))
# sanity: pure depolarizing check of F_pro definition (analytic: N1 with prob q -> F_pro = 1-q)
q=0.05; W=[]
w=np.exp(2j*np.pi/3); X=np.roll(np.eye(3),1,axis=0); Z=np.diag([1,w,w*w])
paulis=[np.linalg.matrix_power(X,a)@np.linalg.matrix_power(Z,b) for a in range(3) for b in range(3)]
def chan(rho): return (1-q)*rho + q/8*sum(P@rho@P.conj().T for P in paulis[1:])
tot=sum(chan(np.outer(np.eye(3)[i],np.eye(3)[j]))[i,j] for i in range(3) for j in range(3))/9
print("sanity: N1 depolarizing q=0.05 -> 1-F_pro =", round(1-tot.real,6))
print("\nGate/readout -> SI1000 p (process infidelity e_pro = (d+1)/d * r for RB r; 2q: p = e_pro; 1q: p = e_pro/0.1; readout: p = (1-F_assign)/5)")
r_list = {"Morvan2021 qutrit RB r (arXiv Table 1, best/worst)": (2.38e-3, 3.94e-3),
          "Liu2023 SU(3) Clifford r": (5.4e-3,5.4e-3)}
for k,(lo,hi) in r_list.items():
    print(f"  1q {k}: e_pro {4/3*lo:.2e}-{4/3*hi:.2e} -> p {4/3*lo/0.1:.3f}-{4/3*hi/0.1:.3f}")
print(f"  1q Morvan2021 published avg process infidelity 3.8e-3 -> p {3.8e-3/0.1:.3f}")
for k,e in {"Goss2022 CZdag (CB process infid 2.7%)":0.027,"Goss2022 CZ (4.8%)":0.048,
            "Blok2021 CSUM QPT (1-0.889)":0.111,"Luo2023 CPhase QPT (1-0.893)":0.107,
            "Morvan2021 CSUM CB (1-0.82, arXiv)":0.18}.items():
    print(f"  2q {k}: p = {e:.3f}")
for k,F in {"Chen2023 3-state 96.9%":0.969,"Blok2021 avg 0.96":0.96,"IQM-chip qutrit 93%":0.93,
            "QubiCML qutrit avg 94.5-97.3%":0.955}.items():
    print(f"  readout {k}: p = {(1-F)/5:.4f}")
