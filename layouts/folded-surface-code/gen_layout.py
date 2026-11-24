#!/usr/bin/env python
"""Nearest-neighbour bilayer layout of the folded (unrotated) surface code, any distance d >= 2.

Builds the layout, checks it, and writes
  data/layout-d<d>.json      qubits, couplers and crossovers with coordinates
  data/summary.json          counts and check results for every d
  figures/folded-d<d>.tex    TikZ picture of the layout (family-d<d>.tex: same, common scale)
  figures/unfolded-d3.tex    the unfolded distance-3 patch with the fold line
  figures/tile.tex           the repeating cells

Conventions (picture coordinates: x to the right, y downwards, lattice pitch a = 1)
  unfolded patch   sites (x, y), 0 <= x, y <= L = 2d-2; data on x+y even, checks on x+y odd
                   X-type check on (x odd, y even), Z-type check on (x even, y odd)
  fold             (x, y) <-> (y, x); fold qubits are the sites (k, k)
  bottom layer     the half x <= y, drawn at (x, y)
  top layer        the mirrored half x > y; the image of (y, x), with x < y, is drawn at
                   (x + 1/2, y - 1/2), i.e. at the centre of a bottom-layer cell
Usage: python gen_layout.py [dmax]   (default 8; figures for d = 2..6)
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------------------------
def build(d):
    L = 2 * d - 2
    [(k, k) for k in range(L + 1)]
    bottom = sorted([(x, y) for y in range(L + 1) for x in range(y) if (x + y) % 2 == 0],
                    key=lambda s: (s[1], s[0]))          # reading order: row, then column
    # ---- data-qubit numbers: d fold qubits, then d-1 blocks of
    #      (d-1 top-layer qubits, their d-1 bottom-layer partners, one fold qubit)
    label = {}          # key ('F', k) | ('B', x, y) | ('T', x, y)  ->  number
    n = 0
    for k in range(d):
        label[('F', k)] = n
        n += 1
    for g in range(d - 1):
        grp = bottom[g * (d - 1):(g + 1) * (d - 1)]
        for (x, y) in grp:
            label[('T', x, y)] = n
            n += 1
        for (x, y) in grp:
            label[('B', x, y)] = n
            n += 1
        label[('F', d + g)] = n
        n += 1
    assert n == d * d + (d - 1) * (d - 1)

    def key_of_unfolded(u, v):
        """data key of the unfolded site (u, v)"""
        if u == v:
            return ('F', u)
        if u < v:
            return ('B', u, v)
        return ('T', v, u)

    def pos(key):
        if key[0] == 'F':
            return (float(key[1]), float(key[1]))
        if key[0] in ('B', 'AB'):
            return (float(key[1]), float(key[2]))
        return (key[1] + 0.5, key[2] - 0.5)              # 'T', 'AT'

    data = []
    for key, q in sorted(label.items(), key=lambda kv: kv[1]):
        if key[0] == 'F':
            unf, role = (key[1], key[1]), 'fold'
        elif key[0] == 'B':
            unf, role = (key[1], key[2]), 'bottom'
        else:
            unf, role = (key[2], key[1]), 'top'
        data.append(dict(q=q, role=role, pos=pos(key), unfolded=unf, key=key))
    partner = {}
    for (x, y) in bottom:
        partner[label[('B', x, y)]] = label[('T', x, y)]
        partner[label[('T', x, y)]] = label[('B', x, y)]

    # ---- check ancillas
    anc = []
    asites = sorted([(x, y) for y in range(L + 1) for x in range(y) if (x + y) % 2 == 1],
                    key=lambda s: (s[1], s[0]))
    for layer in ('bottom', 'top'):
        for (x, y) in asites:
            u, v = (x, y) if layer == 'bottom' else (y, x)       # unfolded site of this check
            typ = 'X' if u % 2 == 1 else 'Z'
            sup = []
            for (du, dv) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                uu, vv = u + du, v + dv
                if 0 <= uu <= L and 0 <= vv <= L:
                    # qudit exponent: X check = vertex (out +, in -), Z check = plaquette circulation
                    if typ == 'X':
                        sgn = +1 if (du, dv) in ((1, 0), (0, 1)) else -1
                    else:
                        sgn = +1 if (du, dv) in ((0, -1), (1, 0)) else -1
                    sup.append((label[key_of_unfolded(uu, vv)], sgn, key_of_unfolded(uu, vv)))
            key = ('AB', x, y) if layer == 'bottom' else ('AT', x, y)
            anc.append(dict(layer=layer, type=typ, pos=pos(key), unfolded=(u, v), key=key,
                            support=sorted(s[0] for s in sup),
                            signed={s[0]: s[1] for s in sup}, nbr_keys=[s[2] for s in sup]))
    for i, a in enumerate(anc):
        a['a'] = i

    # ---- couplers
    couplers = []
    qpos = {q['q']: q['pos'] for q in data}
    qrole = {q['q']: q['role'] for q in data}
    for a in anc:
        for q in a['support']:
            p0, p1 = a['pos'], qpos[q]
            ln = float(np.hypot(p0[0] - p1[0], p0[1] - p1[1]))
            if a['layer'] == 'bottom':
                kind = 'bottom'
            else:
                kind = 'top-fold' if qrole[q] == 'fold' else 'top'
            couplers.append(dict(kind=kind, ends=('a%d' % a['a'], 'q%d' % q), p0=p0, p1=p1, length=ln))
    for (x, y) in bottom:
        qb, qt = label[('B', x, y)], label[('T', x, y)]
        p0, p1 = qpos[qb], qpos[qt]
        couplers.append(dict(kind='pair', ends=('q%d' % qb, 'q%d' % qt), p0=p0, p1=p1,
                             length=float(np.hypot(p0[0] - p1[0], p0[1] - p1[1]))))

    # ---- crossovers: proper intersections of coupler segments
    def cross(c1, c2):
        (x1, y1), (x2, y2) = c1['p0'], c1['p1']
        (x3, y3), (x4, y4) = c2['p0'], c2['p1']
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        s = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
        if 1e-9 < t < 1 - 1e-9 and 1e-9 < s < 1 - 1e-9:
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        return None

    crossings = []
    for i, j in itertools.combinations(range(len(couplers)), 2):
        p = cross(couplers[i], couplers[j])
        if p is not None:
            crossings.append(dict(p=p, i=i, j=j, kinds=sorted((couplers[i]['kind'], couplers[j]['kind']))))
    return dict(d=d, L=L, data=data, anc=anc, couplers=couplers, crossings=crossings,
                partner=partner, label=label)


# ----------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------
def rank_mod(M, q):
    M = np.array(M, dtype=np.int64) % q
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c] % q:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, c]), -1, q)) % q
        for i in range(rows):
            if i != r and M[i, c] % q:
                M[i] = (M[i] - M[i, c] * M[r]) % q
        r += 1
        if r == rows:
            break
    return r


def check_matrices(lay, q):
    n = len(lay['data'])
    HX = [[0] * n for _ in lay['anc'] if _['type'] == 'X']
    HZ = [[0] * n for _ in lay['anc'] if _['type'] == 'Z']
    ix = iz = 0
    for a in lay['anc']:
        if a['type'] == 'X':
            for k, s in a['signed'].items():
                HX[ix][k] = s % q
            ix += 1
        else:
            for k, s in a['signed'].items():
                HZ[iz][k] = s % q
            iz += 1
    return np.array(HX, dtype=np.int64), np.array(HZ, dtype=np.int64)


def logicals(lay, q):
    """X_L on the unfolded column u = 0, Z_L on the unfolded row v = 0 (exponents all +1 work
    for the sign convention used here: checked by commutation below)."""
    n = len(lay['data'])
    xl = np.zeros(n, dtype=np.int64)
    zl = np.zeros(n, dtype=np.int64)
    for qd in lay['data']:
        u, v = qd['unfolded']
        if u == 0:
            xl[qd['q']] = 1
        if v == 0:
            zl[qd['q']] = 1
    return xl, zl


def min_logical_weight(H_same, H_other, wmax, q=2):
    """smallest weight <= wmax of an operator of one Pauli type that commutes with H_other and is
    not in the row space of H_same (qubits). Returns None if there is none up to wmax."""
    n = H_same.shape[1]
    r0 = rank_mod(H_same, 2)
    for w in range(1, wmax + 1):
        for supp in itertools.combinations(range(n), w):
            v = np.zeros(n, dtype=np.int64)
            v[list(supp)] = 1
            if ((H_other @ v) % 2).any():
                continue
            if rank_mod(np.vstack([H_same, v]), 2) > r0:
                return w
    return None


def clifford_checks(lay, q):
    """Fold-transversal gates, as linear maps on Pauli exponent vectors (a | b), phases dropped."""
    n = len(lay['data'])
    HX, HZ = check_matrices(lay, q)
    S = np.vstack([np.hstack([HX, np.zeros_like(HX)]), np.hstack([np.zeros_like(HZ), HZ])]) % q
    rS = rank_mod(S, q)
    xl, zl = logicals(lay, q)
    XL = np.concatenate([xl, np.zeros(n, dtype=np.int64)])
    ZL = np.concatenate([np.zeros(n, dtype=np.int64), zl])
    par = lay['partner']
    unf = {qd['q']: qd['unfolded'] for qd in lay['data']}
    role = {qd['q']: qd['role'] for qd in lay['data']}

    def in_span(v):
        return rank_mod(np.vstack([S, v % q]), q) == rS

    def apply_H(M, fpat):
        out = np.zeros_like(M)
        for i in range(n):
            j = par.get(i, i)                       # SWAP of the fold pairs
            e = fpat(unf[j])
            # F^e on qubit j after the swap:  X -> Z^e,  Z -> X^(-e)
            out[:, j] = (out[:, j] - e * M[:, n + i]) % q
            out[:, n + j] = (out[:, n + j] + e * M[:, i]) % q
        return out

    def apply_S(M, ppat, cpat):
        out = M.copy()
        for i in range(n):
            if role[i] == 'fold':
                out[:, n + i] = (out[:, n + i] + ppat(unf[i]) * M[:, i]) % q
            else:
                j = par[i]
                u, v = unf[i]
                out[:, n + j] = (out[:, n + j] + cpat((min(u, v), max(u, v))) * M[:, i]) % q
        return out

    res = {}
    fpats = {'F on every qubit': lambda s: 1,
             'F on even rows, F^-1 on odd rows': lambda s: 1 if s[1] % 2 == 0 else -1}
    for name, fp in fpats.items():
        M = apply_H(S, fp)
        ok = all(in_span(r) for r in M)
        if ok:
            hx = apply_H(XL[None, :], fp)[0]
            hz = apply_H(ZL[None, :], fp)[0]
            act = None
            for s1 in (1, -1):
                for s2 in (1, -1):
                    if in_span(hx - s1 * ZL) and in_span(hz - s2 * XL):
                        act = 'X_L -> Z_L^%+d, Z_L -> X_L^%+d' % (s1, s2)
            res['H'] = dict(pattern=name, logical=act)
            break
    ppats = {'P on every fold qubit': lambda s: 1,
             'P^((-1)^k) on fold qubit (k,k)': lambda s: 1 if s[0] % 2 == 0 else -1}
    cpats = {'CZ on every pair': lambda s: 1,
             'CZ^((-1)^x) on the pair of (x,y)': lambda s: 1 if s[0] % 2 == 0 else -1,
             'CZ^-1 on every pair': lambda s: -1,
             'CZ^(-(-1)^x) on the pair of (x,y)': lambda s: -1 if s[0] % 2 == 0 else 1}
    for (pn, pp), (cn, cp) in itertools.product(ppats.items(), cpats.items()):
        M = apply_S(S, pp, cp)
        if all(in_span(r) for r in M):
            sx = apply_S(XL[None, :], pp, cp)[0]
            sz = apply_S(ZL[None, :], pp, cp)[0]
            act = None
            for s1 in (1, -1):
                if in_span(sx - XL - s1 * ZL) and in_span(sz - ZL):
                    act = 'X_L -> X_L Z_L^%+d, Z_L -> Z_L' % s1
            if act:
                res['S'] = dict(pattern=pn + '; ' + cn, logical=act)
                break
    return res


def verify(lay, do_distance=True):
    import networkx as nx
    d, _L = lay['d'], lay['L']
    out = dict(d=d)
    n = len(lay['data'])
    na = len(lay['anc'])
    cp = lay['couplers']
    out['data'] = n
    out['fold_qubits'] = sum(q['role'] == 'fold' for q in lay['data'])
    out['pairs'] = len(lay['partner']) // 2
    out['ancillas'] = na
    out['qubits'] = n + na
    out['couplers_check'] = sum(c['kind'] != 'pair' for c in cp)
    out['couplers_top_fold'] = sum(c['kind'] == 'top-fold' for c in cp)
    out['couplers_pair'] = sum(c['kind'] == 'pair' for c in cp)
    out['couplers'] = len(cp)
    out['crossovers'] = len(lay['crossings'])
    kinds = sorted(set(tuple(c['kinds']) for c in lay['crossings']))
    out['crossover_kinds'] = [list(k) for k in kinds]
    lens = sorted(set(round(c['length'], 9) for c in cp))
    out['coupler_lengths'] = lens
    # formulas
    assert n == 2 * d * d - 2 * d + 1 and out['fold_qubits'] == 2 * d - 1 and out['pairs'] == (d - 1) ** 2
    assert na == 2 * d * (d - 1) and out['qubits'] == (2 * d - 1) ** 2
    assert out['couplers_check'] == 4 * (2 * d - 1) * (d - 1)
    assert out['couplers_top_fold'] == 4 * (d - 1)
    assert out['couplers'] == (d - 1) * (9 * d - 5)
    assert out['crossovers'] == 2 * (d - 1) * (2 * d - 3)
    assert kinds == [('bottom', 'top')]
    assert lens == [round(2 ** -0.5, 9), 1.0]
    # every crossover: two couplers at right angles, each crossed at its midpoint
    for c in lay['crossings']:
        for k in (c['i'], c['j']):
            m = ((cp[k]['p0'][0] + cp[k]['p1'][0]) / 2, (cp[k]['p0'][1] + cp[k]['p1'][1]) / 2)
            assert abs(m[0] - c['p'][0]) < 1e-9 and abs(m[1] - c['p'][1]) < 1e-9
    per = {}
    for c in lay['crossings']:
        for k in (c['i'], c['j']):
            per[k] = per.get(k, 0) + 1
    out['max_crossovers_per_coupler'] = max(per.values()) if per else 0
    # distinct positions, minimum spacing
    P = [q['pos'] for q in lay['data']] + [a['pos'] for a in lay['anc']]
    assert len(set((round(p[0], 6), round(p[1], 6)) for p in P)) == len(P)
    dmin = min(np.hypot(p[0] - r[0], p[1] - r[1]) for p, r in itertools.combinations(P, 2))
    out['min_site_spacing'] = round(float(dmin), 9)
    # coupling graph
    G = nx.Graph()
    for c in cp:
        G.add_edge(*c['ends'], kind=c['kind'])
    assert G.number_of_nodes() == n + na and G.number_of_edges() == len(cp)
    deg = dict(G.degree())
    out['max_degree_data'] = max(v for k, v in deg.items() if k[0] == 'q')
    out['max_degree_ancilla'] = max(v for k, v in deg.items() if k[0] == 'a')
    hist = {}
    for k, v in deg.items():
        hist.setdefault(k[0], {}).setdefault(v, 0)
        hist[k[0]][v] += 1
    out['degree_histogram'] = {k: dict(sorted(v.items())) for k, v in hist.items()}
    out['planar'] = bool(nx.check_planarity(G)[0])
    Gc = G.copy()
    Gc.remove_edges_from([e for e in G.edges if G.edges[e]['kind'] == 'pair'])
    out['planar_without_pair_couplers'] = bool(nx.check_planarity(Gc)[0])
    assert out['planar'] == (d == 2) and out['planar_without_pair_couplers']
    out['bipartite'] = bool(nx.is_bipartite(G))
    # three frequency groups: ancillas | fold + bottom data | top data
    role = {'q%d' % q['q']: q['role'] for q in lay['data']}
    col = {v: (0 if v[0] == 'a' else (2 if role[v] == 'top' else 1)) for v in G.nodes}
    assert all(col[u] != col[v] for u, v in G.edges)
    out['three_groups_ok'] = True
    # layer exchange is a symmetry of the coupling graph
    amap = {}
    akey = {a['key']: a['a'] for a in lay['anc']}
    for a in lay['anc']:
        other = ('AT' if a['key'][0] == 'AB' else 'AB',) + a['key'][1:]
        amap['a%d' % a['a']] = 'a%d' % akey[other]
    qmap = {'q%d' % q['q']: 'q%d' % lay['partner'].get(q['q'], q['q']) for q in lay['data']}
    mp = {**amap, **qmap}
    assert all(G.has_edge(mp[u], mp[v]) for u, v in G.edges)
    out['layer_exchange_symmetry'] = True
    # the code
    for q in (2, 3):
        HX, HZ = check_matrices(lay, q)
        assert not ((HX @ HZ.T) % q).any()
        k = n - rank_mod(HX, q) - rank_mod(HZ, q)
        assert k == 1
        xl, zl = logicals(lay, q)
        assert not ((HZ @ xl) % q).any() and not ((HX @ zl) % q).any() and (xl @ zl) % q != 0
        out['clifford_q%d' % q] = clifford_checks(lay, q)
        assert 'H' in out['clifford_q%d' % q] and 'S' in out['clifford_q%d' % q], (d, q)
    # fold maps X checks onto Z checks
    sig = lambda s: frozenset(lay['partner'].get(i, i) for i in s)
    Xs = set(frozenset(a['support']) for a in lay['anc'] if a['type'] == 'X')
    Zs = set(frozenset(a['support']) for a in lay['anc'] if a['type'] == 'Z')
    assert set(sig(s) for s in Xs) == Zs
    out['fold_maps_X_checks_to_Z_checks'] = True
    # each layer holds half of the checks of each type
    out['checks_per_layer_and_type'] = {
        l: {t: sum(a['layer'] == l and a['type'] == t for a in lay['anc']) for t in 'XZ'}
        for l in ('bottom', 'top')}
    if do_distance:
        HX, HZ = check_matrices(lay, 2)
        dx = min_logical_weight(HX, HZ, d)
        dz = min_logical_weight(HZ, HX, d)
        assert dx == d and dz == d, (d, dx, dz)
        out['distance_exhaustive'] = d
    return out


# ----------------------------------------------------------------------------------------------
# TikZ
# ----------------------------------------------------------------------------------------------
STYLE = r"""%% generated by gen_layout.py -- do not edit by hand
"""


def f(v):
    return ('%.4f' % v).rstrip('0').rstrip('.')


def tikz_folded(lay, unit, fontcmd, node_r=0.215, dia=0.25, bridge=0.1, letters=True, numbers=True,
                lw=0.9, top_scale=0.86):
    """picture coordinates: (x, -y) * unit cm"""
    lay['d']
    cp = lay['couplers']
    o = [STYLE, r'\begin{tikzpicture}[x=%scm,y=%scm,line cap=round,line join=round]' % (f(unit), f(unit))]
    P = lambda p: '(%s,%s)' % (f(p[0]), f(-p[1]))
    lwb, lwt, lwp = lw, lw, 2.6 * lw
    # bottom couplers
    for c in cp:
        if c['kind'] == 'bottom':
            o.append(r'\draw[botline,line width=%spt] %s -- %s;' % (f(lwb), P(c['p0']), P(c['p1'])))
    # pair couplers
    for c in cp:
        if c['kind'] == 'pair':
            o.append(r'\draw[pairline,line width=%spt] %s -- %s;' % (f(lwp), P(c['p0']), P(c['p1'])))
    # top couplers with bridges
    cross_of = {}
    for x in lay['crossings']:
        for k in (x['i'], x['j']):
            if cp[k]['kind'] == 'top':
                cross_of.setdefault(k, []).append(x['p'])
    for k, c in enumerate(cp):
        if c['kind'] == 'top-fold':
            o.append(r'\draw[topline,line width=%spt] %s -- %s;' % (f(lwt), P(c['p0']), P(c['p1'])))
        elif c['kind'] == 'top':
            p0, p1 = c['p0'], c['p1']
            xs = cross_of.get(k, [])
            assert len(xs) <= 1
            horiz = abs(p0[1] - p1[1]) < 1e-9
            if horiz:
                a, b = (p0, p1) if p0[0] < p1[0] else (p1, p0)          # left -> right
            else:
                a, b = (p0, p1) if p0[1] < p1[1] else (p1, p0)          # top -> bottom
            if not xs:
                o.append(r'\draw[topline,line width=%spt] %s -- %s;' % (f(lwt), P(a), P(b)))
                continue
            xc = xs[0]
            r = bridge
            if horiz:
                s = '(%s,%s)' % (f(xc[0] - r), f(-xc[1]))
                arc = r'arc[start angle=180,end angle=0,radius=%s]' % f(r)
            else:
                s = '(%s,%s)' % (f(xc[0]), f(-xc[1] + r))
                arc = r'arc[start angle=90,end angle=-90,radius=%s]' % f(r)
            # white gap in the line underneath, then the hop
            o.append(r'\draw[white,line width=%spt,line cap=butt] %s %s;' % (f(lwt + 2.2), s, arc))
            o.append(r'\draw[topline,line width=%spt] %s -- %s %s -- %s;' % (f(lwt), P(a), s, arc, P(b)))
    # ancillas
    for a in lay['anc']:
        sty = 'botanc' if a['layer'] == 'bottom' else 'topanc'
        x, y = a['pos'][0], -a['pos'][1]
        o.append(r'\draw[%s,line width=%spt] (%s,%s) -- (%s,%s) -- (%s,%s) -- (%s,%s) -- cycle;' % (
            sty, f(0.6 * lw), f(x - dia), f(y), f(x), f(y + dia), f(x + dia), f(y), f(x), f(y - dia)))
        if letters:
            o.append(r'\node[anclabel,%s] at (%s,%s) {%s$%s$};' % (
                'botanctext' if a['layer'] == 'bottom' else 'topanctext', f(x), f(y), fontcmd[1], a['type']))
    # data qubits
    for q in lay['data']:
        sty = {'fold': 'foldq', 'bottom': 'botq', 'top': 'topq'}[q['role']]
        x, y = q['pos'][0], -q['pos'][1]
        rr = node_r * (top_scale if q['role'] == 'top' else 1.0)
        o.append(r'\draw[%s,line width=%spt] (%s,%s) circle[radius=%s];' % (sty, f(0.6 * lw), f(x), f(y), f(rr)))
        if numbers:
            o.append(r'\node[qlabel] at (%s,%s) {%s%d};' % (f(x), f(y), fontcmd[0], q['q']))
    o.append(r'\end{tikzpicture}')
    return '\n'.join(o) + '\n'


def tikz_unfolded(lay, unit, fontcmd, node_r=0.215, dia=0.25, lw=0.9):
    _d, L = lay['d'], lay['L']
    o = [STYLE, r'\begin{tikzpicture}[x=%scm,y=%scm,line cap=round,line join=round]' % (f(unit), f(unit))]
    qof = {q['unfolded']: q for q in lay['data']}
    for a in lay['anc']:
        u, v = a['unfolded']
        sty = 'botline' if a['layer'] == 'bottom' else 'topline'
        for (du, dv) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (u + du, v + dv) in qof:
                o.append(r'\draw[%s,line width=%spt] (%s,%s) -- (%s,%s);' % (sty, f(lw), f(u), f(-v), f(u + du), f(-v - dv)))
    m = 0.55
    o.append(r'\draw[foldline] (%s,%s) -- (%s,%s);' % (f(-m), f(m), f(L + m), f(-L - m)))
    o.append(r'\node[foldtext,anchor=south west,inner sep=1pt] at (%s,%s) {fold};' % (f(L + m - 0.25), f(-L - m + 0.42)))
    for a in lay['anc']:
        u, v = a['unfolded']
        sty = 'botanc' if a['layer'] == 'bottom' else 'topanc'
        x, y = u, -v
        o.append(r'\draw[%s,line width=%spt] (%s,%s) -- (%s,%s) -- (%s,%s) -- (%s,%s) -- cycle;' % (
            sty, f(0.6 * lw), f(x - dia), f(y), f(x), f(y + dia), f(x + dia), f(y), f(x), f(y - dia)))
        o.append(r'\node[anclabel,%s] at (%s,%s) {%s$%s$};' % (
            'botanctext' if a['layer'] == 'bottom' else 'topanctext', f(x), f(y), fontcmd[1], a['type']))
    for q in lay['data']:
        u, v = q['unfolded']
        sty = {'fold': 'foldq', 'bottom': 'botq', 'top': 'topq'}[q['role']]
        o.append(r'\draw[%s,line width=%spt] (%s,%s) circle[radius=%s];' % (sty, f(0.6 * lw), f(u), f(-v), f(node_r)))
        o.append(r'\node[qlabel] at (%s,%s) {%s%d};' % (f(u), f(-v), fontcmd[0], q['q']))
    # curved arrows joining two of the fold pairs, to show what the fold does
    o.append(r'\end{tikzpicture}')
    return '\n'.join(o) + '\n'


def tikz_tile(unit=2.3, lw=1.0):
    """Two bulk cells and one cell on the fold, with the two coupler lengths marked."""
    o = [STYLE, r'\begin{tikzpicture}[x=%scm,y=%scm,line cap=round,line join=round]' % (f(unit), f(unit))]
    r, dia, br = 0.13, 0.155, 0.055

    def dq(sty, x, y):
        o.append(r'\draw[%s,line width=%spt] (%s,%s) circle[radius=%s];' % (sty, f(0.6 * lw), f(x), f(y), f(r)))

    def an(sty, x, y):
        o.append(r'\draw[%s,line width=%spt] (%s,%s) -- (%s,%s) -- (%s,%s) -- (%s,%s) -- cycle;' % (
            sty, f(0.6 * lw), f(x - dia), f(y), f(x), f(y + dia), f(x + dia), f(y), f(x), f(y - dia)))

    def hop_h(x0, x1, y, xc):
        s = '(%s,%s)' % (f(xc - br), f(y))
        arc = r'arc[start angle=180,end angle=0,radius=%s]' % f(br)
        o.append(r'\draw[white,line width=%spt,line cap=butt] %s %s;' % (f(lw + 2.2), s, arc))
        o.append(r'\draw[topline,line width=%spt] (%s,%s) -- %s %s -- (%s,%s);' % (f(lw), f(x0), f(y), s, arc, f(x1), f(y)))

    def hop_v(x, y0, y1, yc):
        s = '(%s,%s)' % (f(x), f(yc + br))
        arc = r'arc[start angle=90,end angle=-90,radius=%s]' % f(br)
        o.append(r'\draw[white,line width=%spt,line cap=butt] %s %s;' % (f(lw + 2.2), s, arc))
        o.append(r'\draw[topline,line width=%spt] (%s,%s) -- %s %s -- (%s,%s);' % (f(lw), f(x), f(y0), s, arc, f(x), f(y1)))

    # ---------------- (a) bulk: bottom sites on the corners of two cells side by side
    # bottom lattice: columns x = 0,1,2 ; rows y = 0 (top), -1 (bottom)
    o.append(r'\begin{scope}')
    o.append(r'\node[panel] at (-0.62,0.62) {(a)};')
    for (x0, y0, x1, y1) in ((0, 0, 1, 0), (1, 0, 2, 0), (0, -1, 1, -1), (1, -1, 2, -1),
                             (0, 0, 0, -1), (1, 0, 1, -1), (2, 0, 2, -1)):
        o.append(r'\draw[botline,line width=%spt] (%s,%s) -- (%s,%s);' % (f(lw), x0, y0, x1, y1))
    for (x0, y0, x1, y1) in ((0, 0, -0.45, 0), (0, -1, -0.45, -1), (2, 0, 2.45, 0), (2, -1, 2.45, -1),
                             (0, 0, 0, 0.45), (1, 0, 1, 0.45), (2, 0, 2, 0.45),
                             (0, -1, 0, -1.45), (1, -1, 1, -1.45), (2, -1, 2, -1.45)):
        o.append(r'\draw[botline,line width=%spt,dash pattern=on 2pt off 2pt] (%s,%s) -- (%s,%s);' % (f(lw), x0, y0, f(x1), f(y1)))
    # pair coupler: bottom data at (0,-1) with the top data at the centre of the left cell
    o.append(r'\draw[pairline,line width=%spt] (0,-1) -- (0.5,-0.5);' % f(2.6 * lw))
    # top lattice: data at (0.5,-0.5), ancilla at (1.5,-0.5)
    hop_h(0.5, 1.5, -0.5, 1.0)
    for (x, y0, y1, yc) in ((0.5, 0.45, -0.5, 0.0), (0.5, -0.5, -1.45, -1.0), (1.5, 0.45, -0.5, 0.0), (1.5, -0.5, -1.45, -1.0)):
        hop_v(x, y0, y1, yc)
    hop_h(-0.4, 0.5, -0.5, 0.0)
    hop_h(1.5, 2.4, -0.5, 2.0)
    # nodes
    dq('botq', 0, -1); an('botanc', 1, -1); dq('botq', 2, -1)
    an('botanc', 0, 0); dq('botq', 1, 0); an('botanc', 2, 0)
    dq('topq', 0.5, -0.5); an('topanc', 1.5, -0.5)
    # dimensions
    o.append(r'\draw[dim] (1,-1.72) -- (2,-1.72) node[midway,below,dimtext] {$a$};')
    o.append(r'\draw[dim] (2.72,0) -- (2.72,-1) node[midway,right,dimtext] {$a$};')
    o.append(r'\end{scope}')
    # ---------------- (b) a cell on the fold
    o.append(r'\begin{scope}[shift={(4.25,0)}]')
    o.append(r'\node[panel] at (-0.62,0.62) {(b)};')
    # bottom: ancilla at (0,-1), fold qubits at (0,0) and (1,-1); nothing at (1,0)
    o.append(r'\draw[botline,line width=%spt] (0,0) -- (0,-1) -- (1,-1);' % f(lw))
    for (x0, y0, x1, y1) in ((0, -1, -0.45, -1), (0, -1, 0, -1.45), (1, -1, 1, -1.45), (0, 0, -0.45, 0)):
        o.append(r'\draw[botline,line width=%spt,dash pattern=on 2pt off 2pt] (%s,%s) -- (%s,%s);' % (f(lw), x0, y0, f(x1), f(y1)))
    o.append(r'\draw[topline,line width=%spt] (0,0) -- (0.5,-0.5) -- (1,-1);' % f(lw))
    o.append(r'\draw[topline,line width=%spt,dash pattern=on 2pt off 2pt] (0,0) -- (-0.3,0.3);' % f(lw))
    o.append(r'\draw[topline,line width=%spt,dash pattern=on 2pt off 2pt] (1,-1) -- (1.3,-1.3);' % f(lw))
    hop_h(-0.4, 0.5, -0.5, 0.0)
    hop_v(0.5, -0.5, -1.45, -1.0)
    dq('foldq', 0, 0); dq('foldq', 1, -1); an('botanc', 0, -1); an('topanc', 0.5, -0.5)
    o.append(r'\draw[dim] (0.66,-0.34) -- (1.16,-0.84) node[midway,above right=-2pt,dimtext] {$a/\sqrt2$};')
    o.append(r'\end{scope}')
    o.append(r'\end{tikzpicture}')
    return '\n'.join(o) + '\n'


# ----------------------------------------------------------------------------------------------
def main():
    dmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    summary = []
    prev = None
    for d in range(2, dmax + 1):
        lay = build(d)
        res = verify(lay, do_distance=(d <= 5))
        # the layout of distance d-1 is the upper-left corner of the layout of distance d
        rnd = lambda p: (round(p[0], 6), round(p[1], 6))
        sites = {rnd(q['pos']): q['role'] for q in lay['data']}
        sites.update({rnd(a['pos']): a['layer'] + a['type'] for a in lay['anc']})
        edges = {frozenset((rnd(c['p0']), rnd(c['p1']))) for c in lay['couplers']}
        if prev is not None:
            assert all(sites.get(k) == v for k, v in prev[0].items()) and prev[1] <= edges
        res['contains_previous_distance'] = prev is not None
        prev = (sites, edges)
        summary.append(res)
        js = dict(
            distance=d, pitch='a = 1; x to the right, y downwards',
            data_qubits=[dict(q=q['q'], role=q['role'], x=q['pos'][0], y=q['pos'][1],
                              unfolded=list(q['unfolded']),
                              fold_partner=lay['partner'].get(q['q'])) for q in lay['data']],
            ancillas=[dict(a=a['a'], layer=a['layer'], check=a['type'], x=a['pos'][0], y=a['pos'][1],
                           unfolded=list(a['unfolded']), data=a['support'],
                           qudit_exponents={str(k): v for k, v in sorted(a['signed'].items())})
                      for a in lay['anc']],
            couplers=[dict(kind=c['kind'], between=list(c['ends']), length=round(c['length'], 6))
                      for c in lay['couplers']],
            crossovers=[dict(x=c['p'][0], y=c['p'][1],
                             couplers=[list(lay['couplers'][c['i']]['ends']), list(lay['couplers'][c['j']]['ends'])])
                        for c in lay['crossings']],
            counts={k: res[k] for k in ('data', 'fold_qubits', 'pairs', 'ancillas', 'qubits',
                                        'couplers', 'couplers_pair', 'crossovers')})
        with open(os.path.join(HERE, 'data', 'layout-d%d.json' % d), 'w') as fh:
            json.dump(js, fh, indent=1)
        print('d=%d  data %d (fold %d, pairs %d)  ancillas %d  couplers %d (pair %d)  crossovers %d  '
              'deg %d/%d  planar %s / without pairs %s  distance %s' % (
                  d, res['data'], res['fold_qubits'], res['pairs'], res['ancillas'], res['couplers'],
                  res['couplers_pair'], res['crossovers'], res['max_degree_data'], res['max_degree_ancilla'],
                  res['planar'], res['planar_without_pair_couplers'], res.get('distance_exhaustive', 'not run')))
        for q in (2, 3):
            print('      q=%d  H_L: %s  [%s]' % (q, res['clifford_q%d' % q]['H']['pattern'], res['clifford_q%d' % q]['H']['logical']))
            print('      q=%d  S_L: %s  [%s]' % (q, res['clifford_q%d' % q]['S']['pattern'], res['clifford_q%d' % q]['S']['logical']))
        if d <= 6:
            # one picture per distance, used by the stand-alone PDFs
            unit = {2: 1.5, 3: 1.5, 4: 1.2, 5: 1.05, 6: 0.95}[d]
            fonts = {2: (r'\normalsize ', r'\scriptsize '), 3: (r'\normalsize ', r'\scriptsize '),
                     4: (r'\small ', r'\tiny '), 5: (r'\footnotesize ', r'\tiny '),
                     6: (r'\scriptsize ', r'\tiny ')}[d]
            with open(os.path.join(HERE, 'figures', 'folded-d%d.tex' % d), 'w') as fh:
                fh.write(tikz_folded(lay, unit, fonts, lw={2: 0.85, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.7}[d],
                                     node_r=0.19 if d <= 3 else 0.215, dia=0.21 if d <= 3 else 0.25))
        if d <= 5:
            # the family at one common scale, for the figure in the note
            with open(os.path.join(HERE, 'figures', 'family-d%d.tex' % d), 'w') as fh:
                fh.write(tikz_folded(lay, 1.0, (r'\footnotesize ', r'\tiny '), lw=0.75))
        if d == 3:
            with open(os.path.join(HERE, 'figures', 'unfolded-d3.tex'), 'w') as fh:
                fh.write(tikz_unfolded(lay, 1.25, (r'\normalsize ', r'\scriptsize ')))
            with open(os.path.join(HERE, 'figures', 'folded-d3-main.tex'), 'w') as fh:
                fh.write(tikz_folded(lay, 1.45, (r'\normalsize ', r'\scriptsize '), lw=0.85, node_r=0.19, dia=0.21))
    with open(os.path.join(HERE, 'figures', 'tile.tex'), 'w') as fh:
        fh.write(tikz_tile())
    with open(os.path.join(HERE, 'data', 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    # check tables for the document (d = 2, 3)
    for d in (2, 3):
        lay = build(d)
        rows = []
        for a in lay['anc']:
            rows.append(r'%s & $%s$ & $(%s,\,%s)$ & %s \\' % (
                a['layer'], a['type'], f(a['pos'][0]), f(a['pos'][1]), ', '.join(str(s) for s in a['support'])))
        with open(os.path.join(HERE, 'figures', 'checks-d%d.tex' % d), 'w') as fh:
            fh.write('\n'.join(rows) + '\n')
    print('all checks passed for d = 2..%d' % dmax)


if __name__ == '__main__':
    main()
