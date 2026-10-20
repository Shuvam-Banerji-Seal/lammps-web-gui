import math, random, itertools, sys
random.seed(20261020)
a = 4.497                     # hexagonal a of ice Ih near 250 K (Angstrom)
c = a * math.sqrt(8.0 / 3.0)  # ideal c/a, so every O-O-O angle is tetrahedral (109.47 deg, the SPC/E H-O-H angle)
b = a * math.sqrt(3.0)
NX, NY, NZ = 3, 2, 2
# orthorhombic cell of ice Ih: 8 O; the hexagonal 4f sites of P63/mmc (1/3,2/3,z), (2/3,1/3,z+1/2), (2/3,1/3,-z), (1/3,2/3,1/2-z), z = 1/16
z0 = 1.0 / 16.0
hexO = [(1/3, 2/3, z0), (2/3, 1/3, 0.5 + z0), (2/3, 1/3, 1.0 - z0), (1/3, 2/3, 0.5 - z0)]
def hex2cart(f):
    x, y, z = f
    return (a * x - a / 2 * y, a * math.sqrt(3) / 2 * y, c * z)
# orthorhombic cell = hexagonal cell + its copy shifted by a1 (0..a in x, 0..b in y)
base = []
for f in hexO:
    for sh in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        p = hex2cart((f[0] + sh[0], f[1] + sh[1], f[2]))
        q = (p[0] % a, p[1] % b, p[2] % c)
        if not any(math.dist(q, r) < 1e-6 for r in base):
            base.append(q)
assert len(base) == 8, len(base)
L = (a * NX, b * NY, c * NZ)
O = []
for i, j, k in itertools.product(range(NX), range(NY), range(NZ)):
    for p in base:
        O.append((p[0] + i * a, p[1] + j * b, p[2] + k * c))
N = len(O)
def mi(d, l): return d - l * round(d / l)
def vec(i, j): return tuple(mi(O[j][d] - O[i][d], L[d]) for d in range(3))
nbr = [[] for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        if math.sqrt(sum(v * v for v in vec(i, j))) < 2.9:
            nbr[i].append(j); nbr[j].append(i)
assert all(len(n) == 4 for n in nbr), [len(n) for n in nbr]
oo = math.sqrt(sum(v * v for v in vec(0, nbr[0][0])))
# Eulerian circuit orientation (Hierholzer): every vertex gets in = out = 2, i.e. the ice rules
edges = {}
unused = {i: set(nbr[i]) for i in range(N)}
stack = [0]; path = []
while stack:
    v = stack[-1]
    if unused[v]:
        w = unused[v].pop(); unused[w].discard(v)
        stack.append(w)
    else:
        path.append(stack.pop())
donor = {}  # edge (min,max) -> donor O
for u, w in zip(path, path[1:]):
    donor[(min(u, w), max(u, w))] = u
assert len(donor) == 2 * N
def outs(i): return [j for j in nbr[i] if donor[(min(i, j), max(i, j))] == i]
assert all(len(outs(i)) == 2 for i in range(N))
def dipole():
    m = [0.0, 0.0, 0.0]
    for (u, w), d in donor.items():
        t = w if d == u else u
        v = vec(d, t)
        for k in range(3): m[k] += v[k]
    return m
def norm(m): return math.sqrt(sum(x * x for x in m))
# flip directed cycles (keeps in = out = 2) while the net dipole shrinks
best = norm(dipole())
for it in range(200000):
    if best < 1e-9: break
    start = random.randrange(N); seen = {start: 0}; walk = [start]
    while True:
        nxt = random.choice(outs(walk[-1]))
        if nxt in seen:
            cyc = walk[seen[nxt]:] + [nxt]; break
        seen[nxt] = len(walk); walk.append(nxt)
    for u, w in zip(cyc, cyc[1:]): donor[(min(u, w), max(u, w))] = w
    m = norm(dipole())
    if m <= best: best = m
    else:
        for u, w in zip(cyc, cyc[1:]): donor[(min(u, w), max(u, w))] = u
assert all(len(outs(i)) == 2 for i in range(N))
print('waters', N, 'O-O', round(oo, 4), 'net dipole (sum of O->O bond vectors)', best, file=sys.stderr)
ROH = 1.0
atoms = []
for i in range(N):
    atoms.append(('O', O[i]))
    for j in outs(i):
        v = vec(i, j); r = math.sqrt(sum(x * x for x in v))
        atoms.append(('H', tuple(O[i][d] + ROH * v[d] / r for d in range(3))))
# check H-O-H angles
for i in range(N):
    h1, h2 = atoms[3 * i + 1][1], atoms[3 * i + 2][1]
    o = atoms[3 * i][1]
    v1 = [h1[d] - o[d] for d in range(3)]; v2 = [h2[d] - o[d] for d in range(3)]
    ang = math.degrees(math.acos(sum(x * y for x, y in zip(v1, v2)) / (norm(v1) * norm(v2))))
    assert abs(ang - 109.4712) < 0.01, ang
rho = N * 18.01528 / 0.602214076 / (L[0] * L[1] * L[2])
print('box', [round(x, 4) for x in L], 'density', round(rho, 4), file=sys.stderr)
out = []
out.append('# Ice Ih, proton ordered (ice rules satisfied, zero net dipole), %d water molecules in a P1 cell' % N)
out.append('# of %dx%dx%d orthorhombic ice Ih cells (a = %.3f A, ideal c/a, O-O = %.4f A). The H atoms sit 1.0 A from' % (NX, NY, NZ, a, oo))
out.append('# their O along the O-O bonds (H-O-H = 109.47 deg, the SPC/E geometry). Each molecule is listed O, H, H with')
out.append('# whole-molecule coordinates (some H fractions lie just outside 0..1). Generated for the LAMMPS Web GUI examples.')
out.append('data_ice_Ih_proton_ordered')
out.append("_chemical_formula_sum            'H2 O'")
out.append('_cell_length_a                   %.6f' % L[0])
out.append('_cell_length_b                   %.6f' % L[1])
out.append('_cell_length_c                   %.6f' % L[2])
out.append('_cell_angle_alpha                90.0')
out.append('_cell_angle_beta                 90.0')
out.append('_cell_angle_gamma                90.0')
out.append("_symmetry_space_group_name_H-M   'P 1'")
out.append('')
out.append('loop_')
out.append('_atom_site_label')
out.append('_atom_site_type_symbol')
out.append('_atom_site_fract_x')
out.append('_atom_site_fract_y')
out.append('_atom_site_fract_z')
cnt = {'O': 0, 'H': 0}
for el, p in atoms:
    cnt[el] += 1
    out.append('%s%d %s %.6f %.6f %.6f' % (el, cnt[el], el, p[0] / L[0], p[1] / L[1], p[2] / L[2]))
open('ice_ih.cif', 'w').write('\n'.join(out) + '\n')
