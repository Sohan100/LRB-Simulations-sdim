# A2. Optimal postselection and correction decisions for qutrit LRB

Agent A2 (theory), 2026-10-28. Nothing committed.

- Script: `notes/scripts/A2_optimal_rule_check.py`
- Full output: `notes/scripts/A2_optimal_rule_check_output.txt`
- Run: `module load python/3.11 && python A2_optimal_rule_check.py` (one core,
  about 1.5 min of CPU; 1.5 to 2.5 min wall time on the loaded login node;
  `--quick` for a smaller run).

Labels: **[VA]** verified analytically, **[VN]** verified numerically,
**[LIT]** from literature fetched in this task (Section 8), **[CONJ]**
conjecture. Section 7 lists the claims with ids `A2-optimal-postselection-Cn`.

---

## 0. Summary

1. **Optimal rule.** Among all rules (randomised ones included) that map the
   full detector history $D$ of a shot to "reject" or to "accept and apply
   logical shift correction $c$", the probability that an accepted shot is
   correct, at fixed acceptance $\alpha$, is maximised by accepting the
   histories with the largest MAP probability $q(D)=\max_l P(l\mid D)$
   (threshold rule, randomising only on the threshold level set) and applying
   $\hat c(D)=\arg\max_l P(l\mid D)$. The optimal curve $S^*(\alpha)$ is
   nonincreasing and $\beta^*(\alpha)=\alpha S^*(\alpha)$ is concave with slope
   equal to the threshold. [VA; VN by LP and exhaustive enumeration]
2. **Detection is one feasible rule**, so $S_{\rm det}\le S^*(\alpha_{\rm det})$.
   On the repository's own circuits (exact $P(D,L)$, depth $m=0$ and $m=2$,
   $p=10^{-3}$ and $10^{-2}$), `unif 1` detection lies *exactly on* the
   frontier because the all-trivial history is the most reliable history.
   `const 1`, `const 2` and `unif 2` lie strictly below it (gap in $S$ up to
   $1.4\times10^{-2}$ at $p=0.01$, $m=2$). [VN]
3. **Metrics.** For qutrits $\lambda:=\mathrm{Re}\,E[\omega^R\mid\mathrm{acc}]=(3S-1)/2$
   exactly, so survival $S$ and $\lambda$ rank rules identically. The legacy
   metric $|E[\omega^R\mid\mathrm{acc}]|=\sqrt{(3Q-1)/2}$, $Q=\sum_r p_r^2$, is a
   symmetric function of the residual distribution: it cannot tell a perfect
   decoder from one that is always wrong by a fixed shift. [VA]
4. **The repository's noise is conjugation symmetric**, $P(D,l)=P(-D,-l)$, for
   every circuit. Hence for detection (and any covariant rule) the legacy
   metric equals $|\lambda|$ exactly. Exact values agree with the baseline CSVs
   to $\le 4.6\times10^{-5}$ (within 1.6 standard errors). [VA, VN]
5. **Legacy and survival can disagree for correction rules.** A correction
   map that is never right scores $|E\omega^R|=0.5$, above the $0.4$ of the
   MAP rule (Example A). Report $S$ (equivalently $\lambda$) and $\alpha$, keep
   the legacy metric only for continuity, and report
   $\mathrm{Im}\,E[\omega^R]$ as a bias diagnostic. [VA, VN]
6. **RB behaviour.** Per-round ("local") rules under Markovian Pauli noise give
   exactly $\alpha(m)=Aa^m$ and $\lambda(m)=Bf^m$. Global rules do not. A fixed
   posterior threshold $t$ pins $S(m)\ge t$ at every depth, so a
   conditional decay rate can be made as slow as desired by postselecting
   harder. [VA, VN]
7. **Per-depth optimality and decay rate.** At a common acceptance profile the
   per-depth optimal rule has the best *asymptotic* decay exponent. It need
   not have the best *fitted* $f$ over a finite window. [VA]
8. **Figures of merit.** Report the per-step acceptance $a$, the conditional
   decay $f$, the combined decay $af$ (rejections scored as random outcomes),
   and the shot cost $N^*(\epsilon)\approx 4(1+\eta/\delta)/\epsilon^2$
   ($\eta=-\ln a$, $\delta=-\ln f$), or
   $N^*\approx 2.77\,(1-B)(\eta/\delta)^2/\epsilon^2$ when a depth-independent
   offset $1-B$ dominates. Baseline `unif 1` at $p=10^{-3}$: $\eta=0.11$,
   $\delta=2.7\times10^{-5}$, $N^*\approx4\times10^7$ submitted shots for 10%
   relative precision on the conditional error. [VA, VN]
9. **If rejected shots count as random outcomes**, the best rule is MAP
   correction with no rejection. [VA, VN]
10. **A postselected decay rate is a logical error rate** only for local rules
    under Markovian noise with acceptance independent of the logical state. It
    is then the error rate of a renormalised heralded channel and must be
    quoted with the rejection rate, as erasure-qubit experiments do [LIT].
    Otherwise it is a fit parameter of a conditioned process. [VA]

---

## 1. Setting and notation

Fix one LRB circuit (one Clifford sequence, one $p$). A shot yields:

- the detector history $D\in\mathcal D=\mathbb Z_3^{4(m+1)}$: the shifts of
  all four ancilla outcomes in all $m+1$ executed rounds, whether or not a
  postselection policy uses them;
- the logical shift $L\in\mathbb Z_3$ of $x_0+x_3$. Every LRB circuit
  implements the logical identity and the noiseless reference value of
  $x_0+x_3$ is 0 (checked on the DEM reference shots of six circuits, where
  $x_0$ and $x_3$ individually vary but sum to 0 mod 3), so the raw logical
  outcome $K$ equals $L$.

Notation: joint law $\pi(D,l)=P(D,L=l)$; $\pi(D)=\sum_l\pi(D,l)$; posterior
$P(l\mid D)=\pi(D,l)/\pi(D)$; MAP probability $q(D)=\max_lP(l\mid D)\in[1/3,1]$;
MAP class $\hat c(D)\in\arg\max_lP(l\mid D)$; $\omega=e^{2\pi i/3}$.

A **decision rule** $\delta$ assigns to each $D$ numbers $\delta(c\mid D)\ge0$,
$c\in\mathbb Z_3$, with $a_\delta(D):=\sum_c\delta(c\mid D)\le1$. Here
$\delta(c\mid D)$ is the probability of accepting the shot and applying the
logical frame correction $c$ (the reported outcome is $K-c$). Deterministic
rules take values in $\{0,1\}$. The residual of an accepted shot is
$R=L-C$; the shot is correct iff $R=0$. Define

$$\alpha(\delta)=\sum_D\pi(D)\,a_\delta(D),\qquad
\beta(\delta)=\sum_D\sum_c\delta(c\mid D)\,\pi(D,c)=P(\mathrm{acc},R=0),\qquad
S(\delta)=\beta/\alpha .$$

**Admissibility.** A rule may depend on the circuit (the Clifford sequence is
known) and on any part of $D$, never on $K$. In LRB the ideal outcome is known,
so a rule that reads $K$ is postselection on success ("accept iff $K=0$" gives
$S=1$). A rule fitted to data must be evaluated on other shots or circuits.

The repository's policies (`unif k`, `const c`) are deterministic rules of this
form: accept iff $D_U=0$ for a set $U$ of rounds that always contains the
terminal round, and apply $c=0$.

---

## 2. The optimal accept/correct rule

### 2.1 Theorem 1 (MAP-threshold rule) [VA]

Fix $\alpha\in(0,1]$. Let $G(t)=P_\pi(q(D)>t)$, $G^-(t)=P_\pi(q(D)\ge t)$, and
choose $t^*$ with $G(t^*)\le\alpha\le G^-(t^*)$. Define

$$\delta^*(c\mid D)=\mathbf 1[c=\hat c(D)]\times
\begin{cases}1,&q(D)>t^*\\ \gamma,&q(D)=t^*\\ 0,&q(D)<t^*\end{cases},
\qquad\gamma=\frac{\alpha-G(t^*)}{G^-(t^*)-G(t^*)}$$

(any $\gamma$ if the denominator vanishes). Then $\alpha(\delta^*)=\alpha$ and
$\beta(\delta^*)=\beta^*(\alpha):=\max\{\beta(\delta):\alpha(\delta)=\alpha\}$, with

$$\beta^*(\alpha)=E\big[q(D)\mathbf 1\{q(D)>t^*\}\big]+t^*\big(\alpha-G(t^*)\big)
=\int_0^\alpha Q(u)\,du ,$$

where $Q$ is the quantile function (decreasing rearrangement) of $q(D)$ under
$\pi$. A rule with acceptance $\alpha$ is optimal iff, for $\pi$-almost every
$D$: (a) $\delta(\cdot\mid D)$ is supported on $\arg\max_lP(l\mid D)$; and
(b) $a_\delta(D)=1$ when $q(D)>t^*$ and $a_\delta(D)=0$ when $q(D)<t^*$.

*Proof.* (i) Correction. For every $D$,
$\sum_c\delta(c\mid D)\pi(D,c)\le a_\delta(D)\,\pi(D)\,q(D)$, with equality iff
$\delta(\cdot\mid D)$ is supported on the argmax set. Hence
$\beta(\delta)\le\sum_D\pi(D)a_\delta(D)q(D)$.
(ii) Acceptance. If $\alpha(\delta)=\alpha$,

$$\sum_D\pi(D)a_\delta(D)q(D)=t^*\alpha+\sum_D\pi(D)a_\delta(D)\big(q(D)-t^*\big)
\le t^*\alpha+\sum_D\pi(D)a^*(D)\big(q(D)-t^*\big)=\beta(\delta^*),$$

because $a^*=1$ where $q>t^*$, $a^*=0$ where $q<t^*$, and terms with
$q=t^*$ vanish. The last equality uses $\alpha(\delta^*)=\alpha$ and equality
in (i) for $\delta^*$. The optimality conditions are the equality cases of
(i) and (ii). $\square$

**Remarks.**

1. *Ties and randomisation.* Ties inside the argmax can be broken in any way,
   deterministic or random, without changing $\beta$. At the threshold, any
   subset or fractions of the histories with $q=t^*$ with total mass
   $\alpha-G(t^*)$ may be accepted. Randomisation is needed only when $\alpha$
   falls strictly inside an atom of the law of $q(D)$. Sorting the histories
   by decreasing $q$ and accepting the first $k$ gives deterministic rules
   that attain $\beta^*$ at the breakpoints $\alpha_k$. Between breakpoints
   the frontier is the linear interpolation, which is the LP relaxation of a
   0/1 knapsack problem. No rule, deterministic or not, exceeds it.
2. *Neyman–Pearson form.* After step (i) the acceptance problem is the
   Neyman–Pearson problem with "size" measure $\mu_0(D)=\pi(D)$ and "power"
   measure $\mu_1(D)=\pi(D,\hat c(D))$. Its likelihood ratio is
   $d\mu_1/d\mu_0=q(D)$; the lemma holds for finite measures. Lagrangian form:
   maximising $\beta-\tau\alpha$ (gain 1 per correct accepted shot, price
   $\tau$ per accepted shot) gives "accept iff $q(D)>\tau$". Chow's cost form,
   minimising $c_eP(\mathrm{error})+c_rP(\mathrm{reject})$, gives "accept iff
   $q(D)>1-c_r/c_e$".
3. *Shape of the frontier.* $\beta^*$ is concave and nondecreasing with slope
   $t^*(\alpha)=Q(\alpha)$, the success probability of the marginal accepted
   history. $S^*(\alpha)=\beta^*(\alpha)/\alpha$ is the mean of $Q$ on
   $[0,\alpha]$, hence nonincreasing. It runs from $\operatorname{ess\,sup}q$
   at $\alpha\to0$ to $S^*(1)=E[q(D)]$, the success probability of the MAP
   decoder without rejection. The minimal undetected-error probability
   $\varepsilon^*(\alpha)=\alpha-\beta^*(\alpha)$ is convex with slope
   $1-t^*$. In threshold form, with rejection $R(t)=P(q<t)$, the undetected
   error is $E(t)=\int_{[t,1]}(1-s)\,dR(s)$: the error–reject curve is fixed
   by the reject function.
4. *Pooling over circuits.* If the acceptance is fixed per circuit, the
   per-circuit rule is optimal for each circuit and hence for the circuit
   average of $S$. If only the pooled acceptance is fixed, the pooled $\beta$
   is maximised by one common threshold across circuits (apply the theorem to
   the disjoint union of the circuits' history spaces).
5. The rule is circuit specific: $q(D)$ and $\hat c(D)$ depend on the Clifford
   sequence through the DEM.

### 2.2 Classical references

- **Chow (1970).** The abstract (fetched through Semantic Scholar; Crossref
  metadata fetched too) describes an optimum rejection rule, a general
  relation between error and reject probabilities, and states that the error
  rate can be evaluated directly from the reject function [LIT]. Grandvalet et
  al. (2008) state that the Bayes rule for classification with a reject
  option, "known as Chow's rule", is defined by thresholds on posterior
  probabilities [LIT]. Theorem 1 is this rule for three classes, with
  acceptance-constrained randomisation made explicit.
- **Forney (1968).** Crossref metadata fetched [LIT]. The IEEE Xplore page
  returned an empty page, so I did not read the paper itself. Merhav
  (arXiv:0711.2501, text fetched) states Forney's decoder as
  $\mathcal R^*_m=\{y:P(y\mid x_m)/\sum_{m'\ne m}P(y\mid x_{m'})\ge e^{nT}\}$,
  with erasure otherwise (his Eq. (9)). He writes that Forney showed, using
  the Neyman–Pearson theorem, that this decoder gives the best tradeoff
  between the error-or-erasure and undetected-error probabilities [LIT].
  With equal priors the ratio equals $P(m\mid y)/(1-P(m\mid y))$, so Forney's
  rule is the posterior threshold $q\ge t=e^{nT}/(1+e^{nT})$ [VA]. Here $T\ge0$
  means $t\ge1/2$. For $\mathbb Z_3$, thresholds in $[1/3,1/2)$ (where
  Forney's rule would return lists) are still covered by Theorem 1, because
  only the argmax is output.
- **Neyman and Pearson (1933).** Crossref metadata only; the publisher page
  returned HTTP 403 [LIT].
- **Soft output in QEC.** Meister, Pattison and Preskill define soft-output
  decoders as decoders that estimate the probability of a logical failure
  given the syndrome, and propose discarding runs where it is too large [LIT].
  Bombín et al. postselect on the "logical gap", roughly the weight difference
  between inequivalent corrections [LIT]. In this notation the soft output is
  $1-q(D)$, so thresholding it is Theorem 1. Under a minimum-weight
  approximation $P(\text{class})\propto\epsilon^{w_{\min}(\text{class})}$,
  $\log(q/q_{(2)})\approx\text{gap}\cdot\log(1/\epsilon)$ where $q_{(2)}$ is
  the second-largest posterior, so gap thresholding approximates it [VA under
  that approximation]. Chen et al. report that in the toric code logical
  failures come mostly from exponentially unlikely syndromes, so rejecting
  them lowers $p_f$ to $p_f^b$ with $b\ge2$ [LIT].

### 2.3 Pure detection lies on or below the frontier

**Corollary.** A detection policy $\delta_U$ (accept iff $D_U=0$, apply $c=0$)
is a feasible rule. Hence $S_U\le S^*(\alpha_U)$ with $\alpha_U=\pi(D_U=0)$.
Equality holds iff (up to threshold ties) every accepted history (positive
mass, $D_U=0$) has $q(D)\ge q(D')$ for every rejected history $D'$, and
$0\in\arg\max P(\cdot\mid D)$ for every accepted history. For `unif 1`
($U$ = all rounds) the accepted set is the single history $D=0$. The
condition then reads: the all-trivial history is top-ranked by $q$ and its
MAP class is 0. [VA]

**Numerical confirmation.**

*Random likelihood tables* (section [T] of the script) [VN]:

- For 120 random tables (3 to 29 histories, one dominant "trivial"
  history), the LP optimum equals the frontier to $5.9\times10^{-12}$.
- Of 60,000 random deterministic or randomised rules, none exceeds the
  frontier (maximum excess $2.2\times10^{-16}$).
- For 40 tables with 6 histories, all $4^6$ deterministic rules were
  enumerated. None exceeds the frontier, and the top-$k$ rules attain it at
  every breakpoint.
- The detection-type rule (accept the dominant history, $c=0$) is on the
  frontier in 47% of the tables and strictly below it otherwise.

*Code capacity* $[[5,1,2]]_3$ (section [C]): i.i.d. single-qutrit
depolarizing ($X^aZ^b$ with probability $p/8$ each) and perfect checks. The
syndrome is either all four checks (Z+X) or the two X-checks only; $L$ is the
shift of $x_0+x_3$. Exact enumeration of $9^5$ configurations [VN]:

| $p$ | syndrome | $\alpha_{\rm det}$ | $1-S_{\rm det}$ | $1-S^*(\alpha_{\rm det})$ | mass ranked above $D=0$ | $1-S^*(1)$ (MAP) | $1-S$ min-weight | tied syndromes |
|---|---|---|---|---|---|---|---|---|
| 0.001 | Z+X | 0.9950 | 6.27e-8 | 6.27e-8 | 0 | 5.03e-4 | 5.04e-4 | 30 |
| 0.01 | Z+X | 0.9510 | 6.45e-6 | 6.45e-6 | 0 | 5.30e-3 | 5.38e-3 | 30 |
| 0.05 | Z+X | 0.7741 | 1.84e-4 | 1.84e-4 | 0 | 3.20e-2 | 3.37e-2 | 30 |
| 0.01 | X only | 0.9631 | 5.73e-5 | 5.73e-5 | 0 | 1.49e-2 | 1.49e-2 | 6 |

Detection is exactly on the frontier in all cases because $D=0$ is the most
reliable syndrome. At $p=0.01$ (Z+X) the frontier is $S^*(0.98)=0.99987$,
$S^*(0.99)=0.99972$, $S^*(1)=0.99470$. All exact posterior ties are between
label 0 and one nonzero label, none between labels 1 and 2. The Z-check
syndrome cuts the MAP failure probability at $\alpha=1$ by a factor of about
2.8 ($5.3\times10^{-3}$ versus $1.49\times10^{-2}$): it locates the qutrit
hit by a depolarizing error, which resolves many distance-2 ambiguities.

*Repository circuits, exact* (section [D]). For depth $m\le2$ the full joint
law $P(D,L)$ on $\mathbb Z_3^{4(m+1)+1}$ (at most $3^{13}$ points) is
computed exactly from the compact DEM. The characteristic function is a
product over noise locations,
$\hat P(k)=\prod_i\big[1-p_i\tfrac{3^{w_i}}{3^{w_i}-1}\mathbf 1(R_i^{\mathsf T}k\ne0)\big]$,
where $R_i$ is the location's response matrix and $w_i$ its fault-vector
width. One $n$-dimensional DFT then gives $P$; this takes a few seconds per
circuit. The baseline circuit files were read only.

Averages over Clifford sequences (30 sequences; 8 at $p=0.01$, $m=2$). "CSV"
is the baseline run's 1M-shot result. $S^*$ is the full-history frontier at
the policy's acceptance [VN]:

| $p$ | $m$ | policy (rounds used) | rejection | $S$ | $\lambda=(3S-1)/2$ | legacy, exact | $S^*$ (same $\alpha$) | CSV legacy | CSV rejection |
|---|---|---|---|---|---|---|---|---|---|
| 1e-3 | 0 | terminal [0] | 0.0611 | 0.994994 | 0.992491 | 0.992491 | 0.994994 | 0.992503 | 0.0611 |
| 1e-3 | 2 | unif 1 [0,1,2] | 0.2473 | 0.994926 | 0.992389 | 0.992389 | 0.994926 | 0.992367 | 0.2473 |
| 1e-3 | 2 | unif 2 [1,2] | 0.2220 | 0.994915 | 0.992372 | 0.992372 | 0.994918 | 0.992350 | 0.2220 |
| 1e-3 | 2 | const 1 [2] | 0.1891 | 0.994792 | 0.992188 | 0.992188 | 0.994862 | 0.992152 | 0.1892 |
| 1e-3 | 2 | const 2 [0,2] | 0.2156 | 0.994844 | 0.992266 | 0.992266 | 0.994914 | 0.992232 | 0.2156 |
| 1e-2 | 0 | terminal [0] | 0.4615 | 0.944773 | 0.917159 | 0.917159 | 0.944773 | 0.917113 | 0.4615 |
| 1e-2 | 2 | unif 1 | 0.9405 | 0.935731 | 0.903596 | 0.903596 | 0.935731 | (8 seq.) | |
| 1e-2 | 2 | unif 2 | 0.9165 | 0.933934 | 0.900901 | 0.900901 | 0.934314 | | |
| 1e-2 | 2 | const 1 | 0.8658 | 0.909647 | 0.864470 | 0.864470 | 0.923620 | | |
| 1e-2 | 2 | const 2 | 0.9067 | 0.921379 | 0.882069 | 0.882069 | 0.933129 | | |

Full-history frontier (MAP-threshold), averaged over sequences:

| $p$ | $m$ | $S^*(0.90)$ | $S^*(0.95)$ | $S^*(0.99)$ | $S^*(1)$ |
|---|---|---|---|---|---|
| 1e-3 | 0 | 0.994994 | 0.994984 | 0.988186 | 0.985922 |
| 1e-3 | 2 | 0.992221 | 0.981564 | 0.966641 | 0.961859 |
| 1e-2 | 0 | 0.886734 | 0.878118 | 0.866027 | 0.861874 |
| 1e-2 | 2 | 0.635001 | 0.622036 | 0.611681 | 0.609018 |

Observations [VN]:

- In every circuit the all-trivial history is top-ranked (no mass ranked
  above it), so `unif 1` is exactly on the frontier.
- The exact $\lambda$ agrees with the CSV legacy fidelity to $\le4.6\times10^{-5}$.
  The expected standard error of a 30-sequence mean is $2\times10^{-5}$ at
  $p=10^{-3}$, $m=2$, and $8.5\times10^{-5}$ at $p=10^{-2}$, $m=0$.
  Rejections agree to $10^{-4}$. This validates both the exact computation
  and the identity of Section 3.4.

### 2.4 Model mismatch [VA]

A real decoder uses a model posterior $\tilde P(l\mid D)$. Let $\tilde\delta$
be the threshold rule built from $\tilde P$, with the threshold tuned so that
its acceptance under the true law is $\alpha$; acceptance is directly
measurable. Then

$$\beta^*(\alpha)-\beta(\tilde\delta)\le2\Delta,\qquad
\Delta=E_\pi\max_c\big|P(c\mid D)-\tilde P(c\mid D)\big| .$$

*Proof.* For any rule $\delta$, $|\beta(\delta)-\tilde\beta(\delta)|\le\Delta$,
where $\tilde\beta$ uses $\tilde P$ with the true $\pi(D)$. Then
$\beta(\tilde\delta)\ge\tilde\beta(\tilde\delta)-\Delta\ge\tilde\beta(\delta^*)-\Delta\ge\beta(\delta^*)-2\Delta$.
$\square$

### 2.5 Consequences for the correction study

- At the `unif 1` acceptance no rule can beat `unif 1` detection; this is
  verified for $m\le2$ and conjectured for larger $m$ (Section 6). For this
  distance-2 code, correction therefore cannot raise accepted fidelity at
  equal acceptance. Its value lies in three other directions:
  - moving along the frontier to higher acceptance at a controlled loss. At
    $p=10^{-3}$, $m=2$: `unif 1` has $\alpha=0.753$, $S=0.99493$; the frontier
    gives $S^*=0.99222$ at $\alpha=0.90$, $0.98156$ at $0.95$, and $0.96186$
    at $\alpha=1$ (MAP, no rejection);
  - replacing policies that ignore rounds by full-history rules at the same
    acceptance (`const 1` at $m=2$ gains $7\times10^{-5}$ in $S$ at
    $p=10^{-3}$ and $1.4\times10^{-2}$ at $p=10^{-2}$);
  - lowering the cost per submitted shot (Section 4.5).
- If rejected shots are scored as random outcomes, correction wins outright
  (Section 4.5 (iv)). At $p=10^{-3}$, $m=2$: $\alpha\lambda=0.747$ for
  `unif 1` versus $0.943$ for MAP without rejection. At $p=10^{-2}$, $m=2$:
  $0.054$ versus $0.414$.

---

## 3. Which accepted-fidelity metric

### 3.1 Three functionals of the accepted residual distribution [VA]

Let $p_r=P(R=r\mid\mathrm{acc})$ and $z=E[\omega^R\mid\mathrm{acc}]=\sum_rp_r\omega^r$.

- survival $S=p_0$;
- polarisation $\lambda=\mathrm{Re}\,z=p_0-\tfrac12(p_1+p_2)=\tfrac{3S-1}{2}$, since
  $\mathrm{Re}\,\omega=\mathrm{Re}\,\omega^2=-\tfrac12$;
- legacy (repository) metric $F_{\rm leg}=|z|$, with
  $\mathrm{Im}\,z=\tfrac{\sqrt3}{2}(p_1-p_2)$ and

$$F_{\rm leg}^2=\lambda^2+\tfrac34(p_1-p_2)^2=\frac{3Q-1}{2},\qquad Q=\sum_rp_r^2 .$$

For a depolarizing logical channel with parameter $\lambda$, $S=(1+2\lambda)/3$.
This is the brief's $F=(1+2f)/3$ and the qutrit case of
$r=(d-1)(1-p)/d$ (Magesan, Gambetta, Emerson, Eq. (35)) [LIT].

### 3.2 Survival and polarisation are equivalent for $d=3$ [VA]

$\lambda$ is an increasing affine function of $S$. The two therefore order all
rules identically, at any acceptance. At fixed $\alpha$, maximising $S$,
$\lambda$, $\beta$ and the joint $J=\alpha\lambda=(3\beta-\alpha)/2$ are the
same problem, solved by Theorem 1.

This is special to $d=3$. For $d\ge5$, $\mathrm{Re}\,E\omega^R=\sum_rp_r\cos(2\pi r/d)$
is not a function of $p_0$. Its optimum at fixed $\alpha$ thresholds
$g(D)=\max_c\sum_lP(l\mid D)\cos(2\pi(l-c)/d)$, a different rule.

### 3.3 Properties of the legacy metric [VA]

(a) **Symmetric in $(p_0,p_1,p_2)$.** $F_{\rm leg}$ is a function of the
collision probability $Q$ only. It is invariant under all six affine
relabellings $r\mapsto\pm r+s$ of $\mathbb Z_3$. Shifting every correction by
a constant $s$ maps $p_r\mapsto p_{r+s}$: $F_{\rm leg}$ is unchanged while the
survival moves from $p_0$ to $p_s$. A decoder that is always wrong by the
same shift has $F_{\rm leg}=1$ and $S=0$.

(b) **Bounds.** $F_{\rm leg}\ge|\lambda|$, with equality iff $p_1=p_2$. For
$\lambda>0$,
$0\le F_{\rm leg}-\lambda=\tfrac34(p_1-p_2)^2/(F_{\rm leg}+\lambda)\le\tfrac38(1-S)^2/\lambda$:
the discrepancy is second order in the residual error.

(c) **Not linear in the accepted distribution** (it is convex). Pooling
populations with different biases gives less than their average. The
repository's average of per-circuit $F_{\rm leg}$ is not the $F_{\rm leg}$ of
the pooled residual.

### 3.4 Conjugation symmetry of the repository's noise [VA, VN]

**Claim.** For every circuit simulated with the compact DEM and the SI1000
model, $\pi(D,l)=\pi(-D,-l)$.

*Proof.* Each noise location draws an independent fault vector $v$ from a law
invariant under $v\mapsto-v$:

- `N1 d`: uniform over the 8 nonzero $(a,b)$;
- `N1 f`: $X^a$ with $a$ uniform on $\{1,2\}$;
- `N2`: a uniform dense `prob_dist` over the 80 nonzero 4-vectors. The script
  asserts this for every N2 location it loads.

The detector and logical response is $\mathbb Z_3$-linear in $v$
(`dem_simulation.py`, `apply_location_response`). The total shift is a sum of
independent symmetric terms, hence symmetric. $\square$

**Consequence.** Call a rule covariant if $\delta(c\mid D)=\delta(-c\mid-D)$.
For a covariant rule,
$P(\mathrm{acc},R=r)=\sum_{D,c}\delta(c\mid D)\pi(D,c+r)=P(\mathrm{acc},R=-r)$,
so $p_1=p_2$, $\mathrm{Im}\,z=0$ and $F_{\rm leg}=|\lambda|$. Every detection
policy is covariant.

A covariant optimal rule always exists. Since $q(-D)=q(D)$ and
$\arg\max P(\cdot\mid-D)=-\arg\max P(\cdot\mid D)$, choose
$\hat c(-D)=-\hat c(D)$ for $D\ne0$ ($D\ne-D$ in odd characteristic). At
$D=0$ the posterior is symmetric, so $\hat c(0)=0$ unless
$P(1\mid0)=P(2\mid0)>P(0\mid0)$; in that case a covariant rule randomises
between 1 and 2.

**Verification [VN].**

- Exact $P(D,L)$ for 98 circuits: $\max|P(x)-P(-x)|\le1.4\times10^{-16}$.
- $|F_{\rm leg}-\lambda|\le10^{-16}$ for all four policies.
- The exact values match the baseline CSVs (Section 2.3).
- Hence the baseline legacy numbers are survival probabilities in disguise:
  $S=(1+2F_{\rm leg})/3$ per circuit whenever $\lambda\ge0$.
- In the code-capacity model, breaking ties by smallest label happens to be
  covariant, because no tie involves labels 1 and 2 only. A decoder that
  breaks such a tie by label would not be covariant.

### 3.5 When maximising $S$ also maximises the legacy metric

- *Always for $\lambda$* (Section 3.2). [VA]
- *Within covariant rules under symmetric noise:* $F_{\rm leg}=|\lambda|$, so
  the covariant MAP-threshold rule maximises both. [VA]
- *Over all rules, geometric statement.* The achievable set
  $\Phi_\alpha=\{z(\delta):\alpha(\delta)=\alpha\}$ is convex. It is invariant
  under multiplication by $\omega$ (constant shift of the correction), and
  under conjugation when the noise is symmetric. $\max\lambda$ is its support
  function at angle 0; $\max F_{\rm leg}$ is its largest modulus. They
  coincide iff a point of largest modulus lies on a ray at angle
  $0\bmod 2\pi/3$. [VA]
- *Sufficient condition, "aligned posteriors".* Suppose every history's
  posterior has its two non-maximal entries equal. Then
  $z(D):=E[\omega^L\mid D]=\omega^{\hat c(D)}(3q(D)-1)/2$, and for any rule

  $$\alpha|z(\delta)|\le\sum_D\pi(D)a_\delta(D)|z(D)|=\tfrac12\Big(3\sum_D\pi(D)a_\delta(D)q(D)-\alpha\Big)\le\alpha\lambda^*(\alpha).$$

  The MAP-threshold rule attains this bound with all terms on the positive real
  axis. So it maximises $S$, $\lambda$ and $F_{\rm leg}$ at every $\alpha$.
  [VA]
- *High-fidelity bound.* Let $\delta_L$ maximise $F_{\rm leg}$ at fixed
  $\alpha$, with $\lambda_L>0$; choosing the best of its three constant
  shifts guarantees $\lambda_L\ge0$. Then
  $\lambda^*-\lambda_L\le F_L-\lambda_L\le\tfrac38(1-S_L)^2/\lambda_L$, that
  is, $S^*-S_L\le(1-S_L)^2/(4\lambda_L)$. [VA] In 2000 random high-fidelity
  tables there were no violations and the loss was 0. [VN]

**Counterexamples [VN].**

*Example A* (a mirror pair $D,-D$, consistent with symmetric noise).
$\pi(D)=\pi(-D)=1/2$, $P(\cdot\mid D)=(0.6,0.4,0)$, $P(\cdot\mid-D)=(0.6,0,0.4)$,
$\alpha=1$:

| corrections $(c_D,c_{-D})$ | $S$ | $\lambda$ | $F_{\rm leg}$ |
|---|---|---|---|
| (0,0) MAP, covariant | 0.600 | +0.400 | 0.400 |
| (0,2) | 0.500 | +0.250 | 0.500 |
| (2,1), never correct | 0.000 | −0.500 | 0.500 |

The legacy metric prefers two non-MAP rules, one of which is never right.

*Example B* (postselection, rank reversal). Accept one of two histories with
posteriors $(0.70,0.15,0.15)$ and $(0.65,0.35,0)$. Survival and $\lambda$
prefer the first ($S=0.70$, $\lambda=0.55$). The legacy metric prefers the
second ($F_{\rm leg}=0.5635>0.55$, $S=0.65$).

*Random tables* (5 histories, $\alpha=1$, all $3^5$ correction maps; the most
favourable $F_{\rm leg}$ maximiser is taken):

- with diffuse posteriors, the $F_{\rm leg}$ maximiser has strictly lower $S$
  in 27.9% of 2000 tables (maximum loss 0.124);
- with high-fidelity posteriors, in 0%.

### 3.6 Estimation [VA, VN]

- $\hat\lambda=\frac1N\sum_i\mathrm{Re}\,\omega^{R_i}$ over $N$ accepted shots is
  unbiased, with $\mathrm{Var}\,\hat\lambda=V(\lambda)/N$ and
  $V(\lambda)=\frac{1+\lambda}{2}-\lambda^2=\frac{(1-\lambda)(1+2\lambda)}{2}$.
  This holds for any $p_1,p_2$, since $(\mathrm{Re}\,\omega^r)^2\in\{1,\tfrac14\}$.
  When $p_1=p_2$, $1/V$ is the Fisher information per accepted shot.
  (Monte Carlo: 0.2026 against 0.2025.)
- $\hat F_{\rm leg}=|\frac1N\sum_i\omega^{R_i}|$ is biased upward:
  - for $p_1=p_2$ and $\lambda\sqrt N\gg1$, $E\hat F_{\rm leg}\approx F_{\rm leg}+(1-\lambda)/(4N\lambda)$;
  - for a fully depolarised outcome ($\lambda=0$), $E\hat F_{\rm leg}\approx\sqrt{\pi/(4N)}$
    (Rayleigh), not 0.
- **Baseline evidence [VN].** At $p=0.01$, `const 1`, $m\ge14$, the
  acceptance saturates at $0.012344$, which is $3^{-4}=0.012346$: the
  terminal 4-trit syndrome is uniformly random. That leaves
  $N_{\rm acc}\approx12{,}300$ shots per circuit, and the CSV legacy
  fidelities are 0.0105, 0.0071, 0.0085 and 0.0082 against the floor
  $\sqrt{\pi/(4N_{\rm acc})}=0.0080$. The accepted state is fully depolarised;
  the reported number is pure noise floor.
- Low-acceptance points must be excluded from decay fits:
  - in the baseline CSVs, `unif 1` and `unif 2` at $p=0.01$ have NaN
    fidelities for $m\ge14$, and few accepted shots before that;
  - in the current `extract_statistics`, a circuit with zero accepted shots
    contributes fidelity 0.

### 3.7 Recommendation

For a correction study, report per depth and per strategy:

1. **Primary:** survival $S(m)=P(R=0\mid\mathrm{acc})$ and
   $\lambda(m)=\mathrm{Re}\,E[\omega^R\mid\mathrm{acc}]=(3S-1)/2$. They are the
   same information for $d=3$. $\lambda$ is the decay-friendly form: it tends
   to 0 for a depolarised logical state, consistent with the repository's
   offset-0 LRB fit. Use the unbiased $\hat\lambda$.
2. **Acceptance** $\alpha(m)$, and the joint $J(m)=\alpha(m)\lambda(m)$.
3. **For continuity:** the legacy $|E\omega^R|$. For detection on this noise
   model it equals $\lambda$ (Section 3.4). Also report
   $\mathrm{Im}\,E[\omega^R\mid\mathrm{acc}]$, i.e. $p_1-p_2$, as a bias
   diagnostic: it is nonzero for non-covariant tie-breaking or sign-convention
   errors.
4. **Do not** use the legacy metric to select decoders, tie-breaking rules or
   frame conventions.
5. Drop points with very few accepted shots instead of setting them to 0. Say
   whether circuit averages are means of per-circuit ratios (current code) or
   pooled ratios.

---

## 4. From per-depth optimality to RB decay

### 4.1 Quantities versus depth

For a rule $\delta_m$ at each depth $m$ (it sees the $m+1$ rounds), define:

- acceptance $\alpha(m)$;
- conditional $S(m)$ and $\lambda(m)$;
- the joint (unnormalised) polarisation
  $J(m)=E[\mathrm{Re}\,\omega^R\mathbf 1_{\rm acc}]=\alpha(m)\lambda(m)=(3\beta(m)-\alpha(m))/2$,
  which is linear in the rule.

$J(m)$ is the polarisation the experiment would report if every rejected
shot were replaced by a uniformly random outcome.

"A fixed decision rule at every depth" can mean either of two things:

- (a) a fixed per-round (local) rule, e.g. "reject if any used round flags" or
  "decode each round and reject if that round is ambiguous";
- (b) a fixed posterior threshold $t$ applied to the full history.

The frontier rule at a prescribed acceptance profile is neither: its threshold
$t^*(m)$ changes with $m$.

### 4.2 Local rules under Markovian noise give exact exponentials [VA, VN]

Let each step $j$ produce $(\Delta D_j,l_j)$, i.i.d. with law $\pi_1$. A local
rule decides $(\mathrm{acc}_j,c_j)$ from $\Delta D_j$ alone with $\delta_1$.
The shot is accepted iff every step is, and $R=\sum_j(l_j-c_j)$. Then

$$E[\omega^R\mathbf 1_{\rm acc}]=\prod_{j=1}^mE[\omega^{l_j-c_j}\mathbf 1_{\mathrm{acc}_j}]=\hat t^{\,m},\qquad
\hat t=\sum_{\Delta D,c,l}\delta_1(c\mid\Delta D)\,\pi_1(\Delta D,l)\,\omega^{l-c},\qquad
\alpha(m)=a^m,$$

with $a=\sum_{\Delta D}\pi_1(\Delta D)a_{\delta_1}(\Delta D)$. Thus
$J(m)=\mathrm{Re}\,\hat t^{\,m}$ and $\lambda(m)=\mathrm{Re}(\hat t/a)^m$.

- If $\pi_1$ is conjugation symmetric and $\delta_1$ covariant, $\hat t$ is
  real. Then $\lambda(m)=f^m$ with $f=\hat t/a$, the polarisation of the
  renormalised accepted step. Preparation and readout steps that differ from
  the bulk add the constants $A$ and $B$.
- A biased (non-covariant) local rule has $\hat t=|\hat t|e^{i\theta}$. Then
  $\lambda(m)=|\hat t/a|^m\cos(m\theta)$, while $F_{\rm leg}(m)=|\hat t/a|^m$:
  the legacy metric hides a shift that accumulates coherently from step to
  step.

In the toy multi-round model (section [R]: code capacity per step,
$p=0.02$, perfect checks, Monte Carlo with $2\times10^5$ shots), the exact
values $\alpha_{\rm DET}=0.90397^m$, $\lambda_{\rm DET}=0.99996^m$ and
$\lambda_{\rm LMAP}=0.983211^m$ are reproduced by the Monte Carlo: fitted
$a=0.90398$, $f=0.999957$ and $f=0.983197$.

The real LRB circuit is not a product of identical steps:

- check rounds follow only the forward Cliffords;
- the $m$ inverse Cliffords are unchecked, and only the final round sees their
  accumulated syndrome;
- raw detectors make a data error visible in every later round;
- logical Cliffords permute error types.

Even so, the baseline `unif 1` acceptance is exponential to within a maximum
log-residual of $2.3\times10^{-3}$ ($p=10^{-3}$) and $4.4\times10^{-3}$
($p=2\times10^{-3}$) over $m=0,\dots,22$, and $\lambda$ to within
$7.5\times10^{-5}$ and $2.8\times10^{-4}$ [VN].

`const 1` (terminal only) is not exponential: log-residuals reach
$9\times10^{-3}$ to $6\times10^{-2}$ at $p\le2\times10^{-3}$, and are of
order 1 at $p=10^{-2}$, where its acceptance saturates at $3^{-4}$ instead of
decaying [VN]. In the toy model, terminal-only detection (DETT) likewise has
a non-exponential $\lambda(m)$: $1-\lambda$ grows faster than linearly, and
the exponential fit returns $B=1.008>1$. Error pairs whose syndromes cancel
at the end accumulate roughly as $m^2$.

### 4.3 Global rules; fixed posterior thresholds [VA, VN]

For rule (b), $S(m)=E[q(D)\mid q(D)\ge t]\ge t$ at every depth with
$\alpha(m)>0$. So $\lambda(m)\ge(3t-1)/2$: the conditional fidelity is pinned
and all of the decay moves into the acceptance.

In the toy model with $t=0.99$, $S\ge0.9969$ at every $m\le22$ while
$\alpha$ falls from 0.936 to 0.193. The fitted conditional decay is
$f=0.99980$ per step, against $0.98386$ for full-history MAP without
rejection. A conditional decay rate is therefore a property of the pair
(rule, acceptance profile), not of the hardware alone.

### 4.4 Does per-depth optimality give the best decay rate?

**Proposition [VA].** Fix an acceptance profile $\alpha(m)$. The per-depth
optimal rule has $\lambda^*(m)\ge\lambda_\delta(m)$ for every $m$ and every
rule $\delta$ with the same profile. Hence

$$\liminf_m\tfrac1m\ln\lambda^*(m)\ge\liminf_m\tfrac1m\ln\lambda_\delta(m)$$

(and likewise for $\limsup$): it has the best asymptotic decay exponent.

**This does not extend to fitted decay constants on a finite window [VA].**
The fitted $f$ is the slope of $\ln\lambda$, and pointwise domination does not
order slopes. Example: on $m\in[0,22]$, $\lambda_1(m)=0.99^m\ge\lambda_2(m)=0.8\cdot0.995^m$
everywhere, yet $f_1=0.99<f_2=0.995$. This only shows that the implication
fails; I did not construct two actual rules with these curves.

In the toy model the pointwise-optimal rules happen to fit the larger $f$ as
well [VN]:

- FRONT_T, the frontier at the DETT acceptance profile, has $f=0.999914$
  against $0.997827$ for DETT, and its $S$ is higher or equal at every depth
  (by up to 0.031);
- FRONT coincides exactly with DET, because the all-trivial history is
  top-ranked.

Comparing decay rates of strategies with **different** acceptance profiles
is meaningless without one of the joint figures of merit below.

### 4.5 Figures of merit [VA, VN]

Fit $\alpha(m)=Aa^m$ and $\lambda(m)=Bf^m$, and write $\eta=-\ln a\approx1-a$,
$\delta=-\ln f\approx1-f$. In the repository a step $m$ contains one forward
Clifford, one check round and one inverse Clifford, so per logical Clifford
$f_C=f^{1/2}$.

**(i) Conditional decay.** $f$, and the conditional error per step
$r_c=\tfrac23(1-f)$, from $r=(d-1)(1-p)/d$ [LIT].

**(ii) Per-step acceptance.** $a$, and the rejection per step $\eta$. This is
the analogue of an erasure rate per Clifford.

**(iii) Shot cost.** At depth $m$, $N$ submitted shots give about $N\alpha(m)$
accepted shots, and $\mathrm{Var}\,\hat\lambda=V(\lambda)/(N\alpha)$. With
$\partial\lambda/\partial f=m\lambda/f$ and $B$ known (for example from the
$m=0$ point), the number of submitted shots needed for relative precision
$\epsilon$ on $1-f$ (hence on $r_c$) is the single-depth Cramér–Rao count

$$N(m)=\frac{V(\lambda_m)\,f^2}{Aa^m\,m^2\lambda_m^2\,\epsilon^2(1-f)^2},\qquad\lambda_m=Bf^m .$$

For $A=B=1$:
$N(m)=\dfrac{e^{\eta m}(e^{\delta m}-1)(e^{\delta m}+2)}{2\epsilon^2m^2(e^\delta-1)^2}$.
Minimising over $m$:

- *no rejection* ($\eta=0$): $m^*\approx0.9/\delta$, $N^*\approx4.0/\epsilon^2$;
- *rejection-dominated* ($\eta\gg\delta$): $m^*\approx1/\eta$, $N^*\approx\tfrac{3e}{2}(\eta/\delta)/\epsilon^2$;
- a rule of thumb covering both, $N^*\approx4(1+\eta/\delta)/\epsilon^2$, is
  within 4% of the exact minimum in all five cases checked;
- *offset-dominated* ($1-B\gg\delta/\eta$, i.e. a depth-independent error
  invisible to the rule): $m^*\approx2/\eta$,
  $N^*\approx\tfrac{3e^2}{8}(1-B)(\eta/\delta)^2/\epsilon^2$, within 10% of
  the exact minimum in both cases checked.

**(iv) Rejections scored as failures.** $J(m)=AB\,(af)^m$ in the Markov case,
with total error per step $r_{\rm tot}=\tfrac23(1-af)$. Since
$J=\sum_D\pi(D)\sum_c\delta(c\mid D)\,(3P(c\mid D)-1)/2$ and every term is
$\ge0$ at the argmax ($q\ge1/3$), $J$ is maximised by never rejecting and
applying MAP corrections: $J^*=(3E[q]-1)/2$. If a rejected shot costs as much
as a wrong answer, correction beats detection at every depth [VA]. This is
confirmed in the toy model (GMAP has the largest $J$) and on the real circuits
(Section 2.5) [VN].

**(v) Cost per accepted shot.** $1/\alpha(m)=A^{-1}a^{-m}$, which is the
relevant overhead when rejection is allowed and the accepted output is used
(sampling tasks).

Which to quote:

- the pair $(\eta,r_c)$ describes the logical qutrit when it is kept;
- $r_{\rm tot}$ describes it as a component that must answer every time;
- $N^*$ is the cost of benchmarking it.

**Toy model** (section [R], per step, $p=0.02$; $\epsilon=0.1$):

| rule | $a$ | $\eta$ | $f$ | $r_c$ | $r_{\rm tot}$ | $N^*$ ($m^*$) |
|---|---|---|---|---|---|---|
| DET (all rounds) | 0.90398 | 0.096 | 0.999957 | 2.9e-5 | 6.40e-2 | 9.6e5 (10) |
| FRONT (frontier at DET profile) | identical to DET | | | | | |
| DETT (final syndrome only) | 0.90896 | 0.091 | 0.9978 (non-exponential) | 1.4e-3 | 6.20e-2 | invalid fit |
| FRONT_T (frontier at DETT profile) | 0.90896 | 0.091 | 0.999914 | 5.8e-5 | 6.07e-2 | 4.6e5 (10) |
| LTH (per-step $q\ge0.9$ + MAP) | 0.97890 | 0.021 | 0.998867 | 7.6e-4 | 1.48e-2 | 8.0e3 (45) |
| GTH (history $q\ge0.99$) | 0.92777 | 0.072 | 0.999803 | 1.3e-4 | 4.83e-2 | 1.7e5 (14) |
| LMAP (per-step MAP, no rejection) | 1 | 0 | 0.983197 | 1.12e-2 | 1.12e-2 | 4.0e2 (53) |
| GMAP (history MAP, no rejection) | 1 | 0 | 0.983860 | 1.08e-2 | 1.08e-2 | 4.0e2 (55) |

**Baseline LRB-D** (section [B]): unweighted log-linear fits over all depths
with at least 1000 accepted shots, so they differ slightly from the
repository's weighted fits. $\epsilon=0.1$, $f$ is per $m$, and $\eta$ in the
table is $1-a$:

| $p$ | policy | $a$ | $\eta$ | $f$ | $r_c$ | $r_{\rm tot}$ | $\eta/\delta$ | $N^*$ ($m^*$) |
|---|---|---|---|---|---|---|---|---|
| 1e-3 | unif 1 | 0.89435 | 0.106 | 0.9999727 | 1.8e-5 | 7.0e-2 | 3.9e3 | 4.0e7 (17) |
| 1e-3 | unif 2 | 0.91278 | 0.087 | 0.9999517 | 3.2e-5 | 5.8e-2 | 1.8e3 | 9.1e6 (21) |
| 1e-3 | const 1 | 0.93356 | 0.066 | 0.998836 | 7.8e-4 | 4.5e-2 | 57 | 3.2e4 (17), non-exp. |
| 1e-3 | const 0 (other circuits) | 0.99192 | 0.008 | 0.9999499 | 3.3e-5 | 5.4e-3 | 161 | 6.6e4 (123) |
| 2e-3 | unif 1 | 0.79998 | 0.200 | 0.9998854 | 7.6e-5 | 1.3e-1 | 1.7e3 | 2.0e7 (9) |

For `unif 1` at $p=10^{-3}$, the cost is dominated by two things. The
depth-independent offset $1-B\approx7.5\times10^{-3}$ (errors after the last
check, which detection cannot see) buries the decay signal
$m\delta\approx5\times10^{-4}$ at the useful depths $m\approx2/\eta$, and
the acceptance decays as $e^{-\eta m}$. The offset-dominated formula gives
$3.5\times10^7$, against $4.0\times10^7$ from the exact minimisation. The
`const 0` circuits have no ancilla checks at all, so their $(a,f)$ belong to
a different experiment, not to a different rule on the same circuits.

### 4.6 Is a postselected decay rate a logical error rate?

It is the error rate of a fixed logical channel when three conditions hold:

- (a) acceptance factorises into per-step events decided from that step's
  syndrome data;
- (b) faults are independent across steps (Markovian noise);
- (c) the acceptance probability does not depend on the logical state. This
  holds for Pauli noise with stabilizer checks, because the syndrome statistics
  depend only on the Pauli fault.

Then the accepted branch of one step is a CP map $T_{\rm acc}$ with
$\mathrm{Tr}\,T_{\rm acc}(\rho)=a$ for every logical $\rho$, and
$\Lambda_{\rm acc}=T_{\rm acc}/a$ is CPTP. $f$ is its (twirled) polarisation:
a residual error rate per step given no flag. It must be quoted together with
$\eta$. [VA]

This is the practice in erasure-qubit RB. Levine et al. extract the erasure
error per Clifford from the decay of the postselection probability
($4.38(3)\times10^{-3}$). They extract the residual error ($1.01(1)\times10^{-4}$
per Clifford) from the decay of the postselected survival, fitted with offset
1/2 [LIT].

If (c) fails (coherent errors, leakage), the conditional map
$\rho\mapsto T_{\rm acc}(\rho)/\mathrm{Tr}\,T_{\rm acc}(\rho)$ is nonlinear in
$\rho$, and standard RB theory does not apply to the postselected data [VA].
If (a) or (b) fails, the conditioned process is not a fixed per-step channel.
This covers global (MAP over the whole history) rules, depth-dependent
policies such as `const c`, the unchecked inverse half, and terminal-only
detection. $f$ is then a fit parameter: it depends on the depth window and
the acceptance profile, and Section 4.3 shows it can be pushed towards 1. It
must not be compared with unpostselected logical error rates. Conditioning on
an event at the end of the circuit also correlates the steps, which is a
mechanism for apparent non-Markovianity in postselected LRB (study question 2)
[CONJ as to its size here].

---

## 5. The script `A2_optimal_rule_check.py`

| section | content | result |
|---|---|---|
| [T] | random tables: LP (HiGHS) vs frontier; 60,000 random rules; exhaustive $4^6$ enumeration; detection-type rule | 5 checks pass |
| [C] | exact code-capacity $[[5,1,2]]_3$ (Z+X or X-only syndrome): detection vs frontier, MAP, min-weight lookup decoder, ties, symmetry | detection on the frontier; symmetry to $5\times10^{-17}$ |
| [M] | Examples A and B; brute force of $\arg\max S$ vs $\arg\max F_{\rm leg}$; loss bound; variance identity | 2 checks pass |
| [R] | toy multi-round LRB with 8 rules: $\alpha(m)$, $\lambda(m)$, fits, $N^*$; cost-formula checks | 4 checks pass |
| [D] | exact $P(D,L)$ of 98 baseline circuits ($m=0,2$; $p=10^{-3},10^{-2}$); policies vs frontier vs CSV | matches CSV |
| [B] | baseline CSV figures of merit and legacy-metric noise floor | tables above |

All 12 checks print `PASS`. The script limits BLAS to one thread through
`threadpoolctl`; numpy is preloaded by a site hook here, so environment
variables set inside the script have no effect. It reads, but never writes,
the baseline run folder.

---

## 6. Open points

- **[CONJ]** The all-trivial history remains top-ranked for $m>2$, so
  `unif 1` stays exactly on the frontier at its own acceptance. A history
  with a single isolated measurement flip has the same undetectable-error
  background plus an $O(p)$ penalty, which suggests this holds, but exact
  verification is infeasible beyond $m=2$ ($3^{21}$ states). It needs a
  sampled frontier: estimate $q(D)$ with a decoder on sampled shots and sort.
- **[CONJ]** For $m\ge4$ the frontier gain over `const c` grows with $m$,
  because the information in the ignored rounds grows. Based on the toy model
  (FRONT_T versus DETT) only.
- The effect of global conditioning on apparent non-Markovianity
  (Section 4.6) is left to the non-Markovianity study.

---

## 7. Claims

| id | claim | label | evidence |
|---|---|---|---|
| A2-optimal-postselection-C1 | At fixed acceptance, the MAP-threshold rule (accept largest $q(D)$, output argmax, randomise only on $q=t^*$) maximises $P(R=0\mid\mathrm{acc})$; optimality conditions (a), (b) | VA, VN | Theorem 1 proof (2.1); [T]: LP = frontier to 5.9e-12, 60,000 random and $40\times4^6$ exhaustive rules never above |
| A2-optimal-postselection-C2 | $\beta^*$ is concave with slope $t^*$; $S^*$ is nonincreasing from ess sup $q$ to $E[q]$; $\varepsilon^*$ is convex; error determined by reject function | VA | 2.1 Remark 3 |
| A2-optimal-postselection-C3 | Theorem 1 is the 3-class case of Chow's optimum reject rule, and of Forney's erasure rule written as a posterior threshold $t=e^{nT}/(1+e^{nT})$ | LIT, VA | Chow 1970 abstract; Grandvalet et al. 2008; Merhav arXiv:0711.2501 Eq. (9) |
| A2-optimal-postselection-C4 | Any detection policy is feasible, hence $S_U\le S^*(\alpha_U)$; equality iff accepted histories are top-ranked and have MAP class 0 | VA | 2.3 Corollary |
| A2-optimal-postselection-C5 | On the repository circuits ($m=0,2$; $p=10^{-3},10^{-2}$) `unif 1` is exactly on the frontier; `unif 2`, `const 1`, `const 2` are strictly below (e.g. `const 1`, $p=0.01$, $m=2$: 0.9096 vs 0.9236) | VN | [D], exact $P(D,L)$ for 98 circuits |
| A2-optimal-postselection-C6 | Code capacity $[[5,1,2]]_3$: detection is on the frontier for $p\le0.05$; MAP without rejection fails with probability $5.3\times10^{-3}$ at $p=0.01$ (Z+X), $1.49\times10^{-2}$ with X-checks only | VN | [C] |
| A2-optimal-postselection-C7 | For $d=3$, $\mathrm{Re}\,E[\omega^R]=(3S-1)/2$, so $S$ and $\lambda$ give identical rule rankings; not true for $d\ge5$ | VA | 3.1, 3.2 |
| A2-optimal-postselection-C8 | $|E\omega^R|^2=(3Q-1)/2$: symmetric in $(p_0,p_1,p_2)$, invariant under constant shifts; $\ge|\lambda|$ with equality iff $p_1=p_2$; excess $\le\tfrac38(1-S)^2/\lambda$ | VA | 3.3 |
| A2-optimal-postselection-C9 | The SI1000 compact DEM is conjugation symmetric, $P(D,l)=P(-D,-l)$, for every circuit | VA, VN | proof in 3.4 from `dem_simulation.py`; $\max|P(x)-P(-x)|\le1.4\times10^{-16}$ on 98 circuits |
| A2-optimal-postselection-C10 | For covariant rules, including all detection policies, the legacy metric equals $|\lambda|$; baseline legacy numbers equal $(3S-1)/2$; exact values match the CSVs to $\le4.6\times10^{-5}$ | VA, VN | 3.4; [D] |
| A2-optimal-postselection-C11 | The legacy metric can prefer non-MAP rules, even a never-correct one ($F_{\rm leg}=0.5$ vs 0.4); rank reversal under postselection; in 27.9% of diffuse random tables its maximiser has lower $S$ | VN | [M], Examples A and B |
| A2-optimal-postselection-C12 | Aligned posteriors are sufficient for the MAP rule to maximise $S$, $\lambda$ and $F_{\rm leg}$ together; in general $S^*-S_L\le(1-S_L)^2/(4\lambda_L)$ | VA, VN | 3.5; [M]: 0 violations in 2000 tables |
| A2-optimal-postselection-C13 | $\hat\lambda$ is unbiased with variance $[(1+\lambda)/2-\lambda^2]/N$; $\hat F_{\rm leg}$ is biased up, with floor $\sqrt{\pi/(4N)}$ at $\lambda=0$, seen in the baseline (`const 1`, $p=0.01$: 0.0071 to 0.0105 vs floor 0.0080) | VA, VN | 3.6; [M] variance check; [B] floor |
| A2-optimal-postselection-C14 | `const 1` acceptance saturates at $3^{-4}$ when the data are scrambled ($p=0.01$, $m\ge14$: 0.012344 vs 0.012346) | VN | [B], baseline CSV |
| A2-optimal-postselection-C15 | Local rules under Markovian noise give exactly $\alpha=Aa^m$, $\lambda=Bf^m$, $J=AB(af)^m$; a biased local rule gives $\lambda\propto\cos(m\theta)$ while $F_{\rm leg}$ shows no oscillation | VA, VN | 4.2 derivation; [R] exact vs MC |
| A2-optimal-postselection-C16 | Baseline `unif 1` acceptance and $\lambda$ are close to exponential (log-residuals $\le4.4\times10^{-3}$ and $\le2.8\times10^{-4}$ for $p\le2\times10^{-3}$); `const 1` is not | VN | [B] |
| A2-optimal-postselection-C17 | A fixed posterior threshold $t$ gives $S(m)\ge t$ at every depth: conditional decay can be made arbitrarily slow while acceptance decays | VA, VN | 4.3; [R]: $S\ge0.9969$, $\alpha$ 0.94 to 0.19 |
| A2-optimal-postselection-C18 | At a common acceptance profile the per-depth optimum has the best asymptotic decay exponent, but not necessarily the best fitted $f$ on a finite window | VA | 4.4 proposition and counterexample |
| A2-optimal-postselection-C19 | Single-depth Cramér–Rao shot cost $N(m)$ (4.5 (iii)); $N^*\approx4(1+\eta/\delta)/\epsilon^2$, or $\approx\tfrac{3e^2}{8}(1-B)(\eta/\delta)^2/\epsilon^2$ when the offset dominates | VA, VN | 4.5; [R] exact minimisation within 4% and 10% |
| A2-optimal-postselection-C20 | With rejections scored as random outcomes, MAP correction without rejection maximises $J=\alpha\lambda$ at every depth | VA, VN | 4.5 (iv); [R]; [D] ($p=10^{-3}$, $m=2$: 0.943 vs 0.747) |
| A2-optimal-postselection-C21 | A postselected decay rate is the error rate of a CPTP renormalised heralded channel only under conditions (a) to (c) of 4.6; otherwise it is a fit parameter; report it with $\eta$, as erasure-qubit RB does | VA, LIT | 4.6; Levine et al. 2024 |
| A2-optimal-postselection-C22 | Plug-in regret: $\beta^*(\alpha)-\beta(\tilde\delta)\le2E\max_c|P(c\mid D)-\tilde P(c\mid D)|$ | VA | 2.4 |
| A2-optimal-postselection-C23 | `unif 1` stays exactly on the frontier for $m>2$ | CONJ | 6 |
| A2-optimal-postselection-C24 | The exact characteristic-function computation of $P(D,L)$ for $m\le2$ reproduces the baseline CSV fidelities and rejections | VN | [D] |

---

## 8. References (all fetched in this task)

1. C. K. Chow, "On optimum recognition error and reject tradeoff", IEEE Trans.
   Inf. Theory 16(1), 41–46 (1970). DOI 10.1109/TIT.1970.1054406. Fetched:
   https://api.crossref.org/works/10.1109/TIT.1970.1054406 and
   https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TIT.1970.1054406
   (abstract). https://ieeexplore.ieee.org/document/1054406/ returned an
   empty page.
2. G. D. Forney, "Exponential error bounds for erasure, list, and decision
   feedback schemes", IEEE Trans. Inf. Theory 14(2), 206–220 (1968). DOI
   10.1109/TIT.1968.1054129. Fetched:
   https://api.crossref.org/works/10.1109/TIT.1968.1054129 (metadata; the
   abstract is elided there and the IEEE page was empty).
3. N. Merhav, "Error exponents of erasure/list decoding revisited via moments
   of distance enumerators", arXiv:0711.2501 (2007); the arXiv page lists the
   related DOI 10.1109/TIT.2008.929004. Fetched: https://arxiv.org/abs/0711.2501
   and https://ar5iv.labs.arxiv.org/html/0711.2501 (Eq. (9), Forney's
   decoder, and its Neyman–Pearson optimality).
4. J. Neyman, E. S. Pearson, "On the problem of the most efficient tests of
   statistical hypotheses", Phil. Trans. R. Soc. A 231, 289–337 (1933). DOI
   10.1098/rsta.1933.0009. Fetched:
   https://api.crossref.org/works/10.1098/rsta.1933.0009 (metadata only).
5. Y. Grandvalet, A. Rakotomamonjy, J. Keshet, S. Canu, "Support vector
   machines with a reject option", NIPS 21 (2008). Fetched:
   https://papers.nips.cc/paper/3594-support-vector-machines-with-a-reject-option
6. E. Magesan, J. M. Gambetta, J. Emerson, "Characterizing quantum gates via
   randomized benchmarking", Phys. Rev. A 85, 042311 (2012), arXiv:1109.6887,
   DOI 10.1103/PhysRevA.85.042311. Fetched: https://arxiv.org/abs/1109.6887
   and https://ar5iv.labs.arxiv.org/html/1109.6887 (Eqs. (35) and (40)).
7. J. Combes, C. Granade, C. Ferrie, S. T. Flammia, "Logical randomized
   benchmarking", arXiv:1702.03688 (2017). Fetched:
   https://arxiv.org/abs/1702.03688 (context only).
8. H. Bombín, M. Pant, S. Roberts, K. I. Seetharam, "Fault-tolerant
   post-selection for low overhead magic state preparation", PRX Quantum 5,
   010302 (2024), arXiv:2212.00813, DOI 10.1103/PRXQuantum.5.010302. Fetched:
   https://arxiv.org/abs/2212.00813
9. N. Meister, C. A. Pattison, J. Preskill, "Efficient soft-output decoders
   for the surface code", arXiv:2405.07433 (2024). Fetched:
   https://arxiv.org/abs/2405.07433
10. H. Chen, D. Xu, G. M. Sommers, et al., "Scalable accuracy gains from
    postselection in quantum error correcting codes", arXiv:2510.05222 (2025).
    Fetched: https://arxiv.org/abs/2510.05222
11. H. Levine, A. Haim, J. S. C. Hung, et al. (113 authors), "Demonstrating a
    long-coherence dual-rail erasure qubit using tunable transmons", Phys.
    Rev. X 14, 011051 (2024), arXiv:2307.08737, DOI
    10.1103/PhysRevX.14.011051. Fetched: https://arxiv.org/abs/2307.08737 and
    https://arxiv.org/html/2307.08737 (Fig. 3 caption and Appendix F: erasure
    error from the decay of the postselection probability, residual error
    from the postselected survival).
