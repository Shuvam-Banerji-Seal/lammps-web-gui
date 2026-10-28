#!/usr/bin/env python3
"""Writes the B-basis potentials and active-set inverses of the pace/extrapolation oracle cases.

The files are made here, not taken from python-ace/pacemaker (those carry the Academic Software Licence):
coefficients and matrices come from numpy's RandomState with fixed seeds, the function lists from the coupling rules
of the B-basis .yaml format (rank 1: l = 0; rank 2: l1 = l2; rank 3: lint = [l3]; rank 4: lint = [L, L]; rank 5:
lint ends with l5; sum of l even). An .asi file is an uncompressed .npz (numpy.savez) with one square matrix per element:
the size is the number of functions of that element (linear grade) or that number times ndensity (non-linear grade).

    python3 scripts/oracle/pace-extrapolation-fixtures.py   # writes tests/oracle/pace_al_*.yaml and .asi
"""
import itertools
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests', 'oracle')


def functions(mu0, neigh, nradbase, nradmax, lmax, rng, ndensity, scale):
    """The nbody entries of one block: central mu0, neighbour species from neigh (the block's elements)."""
    out = []

    def add(types, nr, nl, lint=None):
        # the A's sum over some 30 neighbours: a rank-r product is damped by 30^-(r-1) to keep the forces moderate
        c = [float('%.12g' % x) for x in scale * 30.0 ** (1 - len(nr)) * rng.standard_normal(ndensity)]
        out.append((types, nr, nl, lint, c))

    pairs = [(n, l) for n in range(1, nradmax + 1) for l in range(lmax + 1)]
    for e in neigh:
        if e == mu0 and len(set(neigh)) > 1:
            continue
        for n in range(1, nradbase + 1):
            add([mu0, e], [n], [0])
    combos2 = [p for p in itertools.combinations_with_replacement(pairs, 2) if p[0][1] == p[1][1]]
    for sp in itertools.combinations_with_replacement(neigh, 2):
        for (a, b) in combos2:
            add([mu0, *sp], [a[0], b[0]], [a[1], b[1]])
    for sp in itertools.combinations_with_replacement(neigh, 3):
        for (a, b, c) in itertools.combinations_with_replacement(pairs, 3):
            l1, l2, l3 = a[1], b[1], c[1]
            if (l1 + l2 + l3) % 2 or not abs(l1 - l2) <= l3 <= l1 + l2 or a[0] + b[0] + c[0] > 4:
                continue
            add([mu0, *sp], [a[0], b[0], c[0]], [l1, l2, l3], [l3])
    for sp in itertools.combinations_with_replacement(neigh, 4):
        for ls in ([0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]):
            lo = max(abs(ls[0] - ls[1]), abs(ls[2] - ls[3]))
            for L in range(lo, min(ls[0] + ls[1], ls[2] + ls[3]) + 1):
                add([mu0, *sp], [1, 1, 1, 1], ls, [L, L])
    add([mu0, *([neigh[0]] * 5)], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0])
    return out


def radial_block(rng, nradmax, lmax, nradbase):
    crad = np.zeros((nradmax, lmax + 1, nradbase))
    for n in range(nradmax):
        for l in range(lmax + 1):
            crad[n, l] = 0.1 * rng.standard_normal(nradbase)
            crad[n, l, n] += 1.0
    return [[[round(float(x), 12) for x in row] for row in lv] for lv in crad]


def fmt_list(xs):
    return '[' + ', '.join(repr(x) for x in xs) + ']'


def write_yaml(path, blocks):
    lines = ['global:', '  DeltaSplineBins: 0.001', 'species:']
    for b in blocks:
        lines.append(f"  - speciesblock: {b['name']}")
        for k in ('ndensityi', 'npoti', 'parameters', 'rho_core_cut', 'drho_core_cut', 'nradmaxi', 'lmaxi', 'rcutij',
                  'dcutij', 'NameOfCutoffFunctionij', 'r_in', 'delta_in', 'inner_cutoff_type', 'nradbaseij', 'radbase',
                  'radparameters', 'radcoefficients', 'core-repulsion'):
            if k in b:
                v = b[k]
                lines.append(f'    {k}: {fmt_list(v) if isinstance(v, list) else v}')
        lines.append('    nbody:')
        for (types, nr, nl, lint, c) in b['nbody']:
            entry = f"      - {{type: {' '.join(types)}, nr: {fmt_list(nr)}, nl: {fmt_list(nl)}"
            if lint is not None:
                entry += f', lint: {fmt_list(lint)}'
            entry += f', c: {fmt_list(c)}}}'
            lines.append(entry)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_asi(path, sizes, seed):
    rng = np.random.RandomState(seed)
    mats = {el: (rng.standard_normal((n, n)) / n).astype('<f8') for el, n in sizes.items()}
    with open(path, 'wb') as f:
        np.savez(f, **mats)


def main():
    # single element, linear grade: ChebExpCos radial basis, distance inner cutoff with core repulsion
    rng = np.random.RandomState(20261029)
    cu = {
        'name': 'Cu', 'ndensityi': 2, 'npoti': 'FinnisSinclairShiftedScaled', 'parameters': [1, 1, 1, 0.5],
        'rho_core_cut': 100000, 'drho_core_cut': 250, 'nradmaxi': 2, 'lmaxi': 2, 'rcutij': 5.2, 'dcutij': 0.01,
        'NameOfCutoffFunctionij': 'cos', 'r_in': 1.6, 'delta_in': 0.2, 'inner_cutoff_type': 'distance',
        'nradbaseij': 4, 'radbase': 'ChebExpCos', 'radparameters': [5.25],
        'radcoefficients': radial_block(rng, 2, 2, 4), 'core-repulsion': [50.0, 4.0],
    }
    cu['nbody'] = functions('Cu', ['Cu'], 4, 2, 2, rng, 2, 0.05)
    write_yaml(os.path.join(OUT, 'pace_al_cu.yaml'), [cu])
    write_asi(os.path.join(OUT, 'pace_al_cu.asi'), {'Cu': len(cu['nbody'])}, 11)

    # two elements, non-linear grade (size = functions * ndensity): FinnisSinclair embedding, density inner cutoff,
    # ChebPow for the unlike pair whose radial basis is given in "Cu Ni" only ("Ni Cu" copies it)
    rng = np.random.RandomState(1029)
    base = {'npoti': 'FinnisSinclair', 'parameters': [1, 1, 1, 0.5], 'rho_core_cut': 200, 'drho_core_cut': 50,
            'ndensityi': 2}
    rad = lambda radbase, param: {
        'nradmaxi': 1, 'lmaxi': 1, 'rcutij': 4.8, 'dcutij': 0.01, 'NameOfCutoffFunctionij': 'cos', 'nradbaseij': 3,
        'radbase': radbase, 'radparameters': [param], 'radcoefficients': radial_block(rng, 1, 1, 3),
        'core-repulsion': [5.0, 3.0], 'inner_cutoff_type': 'density',
    }
    blocks = [
        {'name': 'Cu', **base, **rad('ChebExpCos', 5.25)},
        {'name': 'Ni', **base, **rad('ChebExpCos', 4.5)},
        {'name': 'Cu Ni', **rad('ChebPow', 2.0)},
        {'name': 'Ni Cu', 'inner_cutoff_type': 'density'},
    ]
    blocks[0]['nbody'] = functions('Cu', ['Cu'], 3, 1, 1, rng, 2, 0.05)
    blocks[1]['nbody'] = functions('Ni', ['Ni'], 3, 1, 1, rng, 2, 0.05)
    blocks[2]['nbody'] = [f for f in functions('Cu', ['Cu', 'Ni'], 3, 1, 1, rng, 2, 0.03) if 'Ni' in f[0][1:]][:24]
    blocks[3]['nbody'] = [f for f in functions('Ni', ['Cu', 'Ni'], 3, 1, 1, rng, 2, 0.03) if 'Cu' in f[0][1:]][:24]
    write_yaml(os.path.join(OUT, 'pace_al_cuni.yaml'), blocks)
    count = lambda el: sum(len(b['nbody']) for b in blocks if b['name'].split()[0] == el)
    write_asi(os.path.join(OUT, 'pace_al_cuni.asi'), {'Cu': 2 * count('Cu'), 'Ni': 2 * count('Ni')}, 12)


if __name__ == '__main__':
    main()
