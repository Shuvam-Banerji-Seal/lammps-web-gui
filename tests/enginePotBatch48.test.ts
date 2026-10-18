import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { PairPedone } from '../src/engine/force/pair/pedone';
import type { EngineEvent } from '../src/engine/types';

/*
 * Unit checks for the batch-48 pair styles: pedone, smatb (smatb/single) and edip (edip/multi).
 * Energies are recomputed here from the documented formulas (docs.lammps.org/pair_pedone.html,
 * pair_smatb.html, pair_edip.html) with the parameters written in each test; forces are checked
 * against central finite differences of the engine's own energy.
 * Native agreement (measured dimers, fixtures) is in tests/oracle/w48pb_*.in.
 */

interface Atom { type: number; x: [number, number, number] }
interface Run { pe: number; press: number[]; forces: number[][]; error?: string }

/** Runs `run 0` for the given atoms (no periodic images, non-periodic box) and returns pe and forces by id. */
interface EvalOpts { boundary?: string; box?: number; extra?: string[]; files?: Record<string, string>; ntypes?: number }

const evaluate = async (pairLines: string[], atoms: Atom[], opts: EvalOpts = {}): Promise<Run> => {
  const boundary = opts.boundary ?? 'f f f';
  const L = opts.box ?? 60;
  const extra = opts.extra ?? [];
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  for (const [n, t] of Object.entries(opts.files ?? {})) session.addFile(n, t);
  const ntypes = opts.ntypes ?? Math.max(...atoms.map((a) => a.type));
  const lines = [
    'units metal', 'atom_style atomic', `boundary ${boundary}`,
    boundary === 'p p p' ? `region box block 0 ${L} 0 ${L} 0 ${L}` : `region box block -60 60 -60 60 -60 60`,
    `create_box ${ntypes} box`,
    ...atoms.map((a) => `create_atoms ${a.type} single ${a.x.join(' ')} units box`),
    'mass * 1',
    ...extra,
    ...pairLines,
    'thermo_style custom step pe press', 'thermo_modify format float %.17g norm no',
    'run 0', 'write_dump all custom f48.dump id fx fy fz modify format float %.17g sort id',
  ];
  try {
    await session.execute(`${lines.join('\n')}\n`);
  } catch (e) {
    return { pe: NaN, press: [], forces: [], error: e instanceof Error ? e.message : String(e) };
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && 'message' in err) return { pe: NaN, press: [], forces: [], error: String(err.message) };
  const row = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').at(-1)!;
  const dump = files.get('f48.dump')!.trim().split('\n');
  const body = dump.slice(dump.findIndex((l) => l.startsWith('ITEM: ATOMS')) + 1).map((l) => l.trim().split(/\s+/).map(Number));
  const press = [row.row.press as number];
  return { pe: row.row.pe as number, press, forces: body.map((r) => [r[1], r[2], r[3]]) };
};

/** Central finite-difference force on every atom, from the engine's own energy. */
const fdForces = async (pairLines: string[], atoms: Atom[], h = 1e-5, files?: Record<string, string>, ntypes?: number): Promise<number[][]> => {
  const out: number[][] = [];
  for (let a = 0; a < atoms.length; a++) {
    const row: number[] = [];
    for (let c = 0; c < 3; c++) {
      const plus = atoms.map((at, k) => (k === a ? { ...at, x: at.x.map((v, d) => (d === c ? v + h : v)) as Atom['x'] } : at));
      const minus = atoms.map((at, k) => (k === a ? { ...at, x: at.x.map((v, d) => (d === c ? v - h : v)) as Atom['x'] } : at));
      const ep = (await evaluate(pairLines, plus, { files, ntypes })).pe;
      const em = (await evaluate(pairLines, minus, { files, ntypes })).pe;
      row.push(-(ep - em) / (2 * h));
    }
    out.push(row);
  }
  return out;
};

const expectForcesMatch = async (pairLines: string[], atoms: Atom[], rel = 1e-6, files?: Record<string, string>, ntypes?: number): Promise<void> => {
  const run = await evaluate(pairLines, atoms, { files, ntypes });
  expect(run.error).toBeUndefined();
  const fd = await fdForces(pairLines, atoms, 1e-5, files, ntypes);
  for (let a = 0; a < atoms.length; a++) {
    for (let c = 0; c < 3; c++) {
      expect(Math.abs(run.forces[a][c] - fd[a][c])).toBeLessThanOrEqual(rel * Math.max(1, Math.abs(fd[a][c])));
    }
  }
};

/* ---------- pedone ---------- */

/** Pedone energy of one pair (doc formula, B0 = the fourth coefficient). */
const pedoneE = (d0: number, a: number, r0: number, b0: number, r: number): number =>
  d0 * (Math.exp(-2 * a * (r - r0)) - 2 * Math.exp(-a * (r - r0))) + b0 / r ** 12;

/** Potential files of third_party/lammps/potentials (read at run time, as the oracle cases do). */
const potFile = (name: string): string => readFileSync(join(__dirname, '../third_party/lammps/potentials', name), 'utf8');

describe('pair_style pedone (batch 48)', () => {
  const PED = [
    'pair_style pedone 15.0',
    'pair_coeff 1 1 0.030211 2.241334 2.923245 5.0 5.0',
    'pair_coeff 2 2 0.042395 1.379316 3.618701 22.0',
    'pair_coeff 1 2 0.030211 2.241334 2.923245 5.0',
  ];

  it('the energy of a dimer equals the documented formula, and is zero at and beyond the pair cutoff', async () => {
    for (const r of [1.5, 2.0, 2.5, 3.0, 3.5, 4.9]) {
      const run = await evaluate(PED, [{ type: 1, x: [0, 0, 0] }, { type: 2, x: [r, 0, 0] }]);
      expect(run.pe).toBeCloseTo(pedoneE(0.030211, 2.241334, 2.923245, 5.0, r), 12);
    }
    // the 1-2 pair has no own cutoff, so the global 15.0 applies
    const far = await evaluate(PED, [{ type: 1, x: [0, 0, 0] }, { type: 2, x: [5.1, 0, 0] }]);
    expect(far.pe).toBeCloseTo(pedoneE(0.030211, 2.241334, 2.923245, 5.0, 5.1), 12);
    // a pair with its own cutoff 5.0 is zero beyond it (no shift unless pair_modify shift yes)
    const cut = await evaluate(
      ['pair_style pedone 15.0', 'pair_coeff * * 0.030211 2.241334 2.923245 5.0 5.0'],
      [{ type: 1, x: [0, 0, 0] }, { type: 1, x: [5.1, 0, 0] }],
    );
    expect(cut.pe).toBe(0);
  });

  it('forces agree with central finite differences of the energy (three types of pairs)', async () => {
    await expectForcesMatch(PED, [
      { type: 1, x: [0, 0, 0] },
      { type: 2, x: [2.2, 0.4, -0.3] },
      { type: 2, x: [0.5, 2.6, 0.7] },
      { type: 1, x: [-1.9, -0.8, 1.1] },
    ]);
  });

  it('refuses an unset I,J pair (pedone does not mix)', async () => {
    const run = await evaluate(
      ['pair_style pedone 15.0', 'pair_coeff 1 1 0.030211 2.241334 2.923245 5.0', 'pair_coeff 2 2 0.042395 1.379316 3.618701 22.0'],
      [{ type: 1, x: [0, 0, 0] }, { type: 2, x: [2.5, 0, 0] }],
    );
    expect(run.error).toMatch(/not set/);
  });

  it('refuses the tail option and wrong argument counts', async () => {
    const run = await evaluate(['pair_style pedone 15.0', 'pair_coeff * * 0.03 2.2 2.9 5.0', 'pair_modify tail yes'], [{ type: 1, x: [0, 0, 0] }, { type: 1, x: [2.5, 0, 0] }]);
    expect(run.error).toMatch(/tail/);
    const bad = new PairPedone();
    bad.allocate(1);
    expect(() => bad.coeff(['1', '1', '0.1', '2.0', '2.9'], { s: null, log: () => {}, readFile: () => '' })).toThrow(/usage: pair_coeff/);
  });
});

/* ---------- smatb and smatb/single ---------- */

/** Documented Au parameters of the smatb example (R0 p q A xi Rcs Rc). */
const AU = [2.88, 10.35, 4.178, 0.210, 1.818, 4.07293506, 4.9883063257983666];

/** The AgCu pancake of examples/PACKAGES/smtbq (AgCuPancake.data, 34 atoms, periodic 30 box, types 1 Ag and 2 Cu). */
const PANCAKE: Atom[] = [
  { type: 1, x: [11.8677744, 17.4748811, 16.8202155] },
  { type: 1, x: [14.4591543, 12.6388264, 17.3769114] },
  { type: 1, x: [19.2905996, 14.8698601, 15.9074284] },
  { type: 1, x: [13.6418392, 11.2583912, 15.0376329] },
  { type: 1, x: [16.5295136, 18.8875825, 16.3408808] },
  { type: 1, x: [11.4394217, 12.6680604, 13.934792] },
  { type: 1, x: [17.1772792, 13.4579369, 17.0971284] },
  { type: 1, x: [11.7477198, 12.5836832, 16.6835448] },
  { type: 1, x: [12.3254647, 15.127665, 12.9151285] },
  { type: 1, x: [11.5595413, 17.5592601, 14.0713777] },
  { type: 1, x: [18.9820568, 14.9536515, 13.1587506] },
  { type: 1, x: [14.5653354, 15.1189885, 11.12255] },
  { type: 1, x: [14.5797485, 17.250847, 17.5049696] },
  { type: 1, x: [13.9305528, 12.7833817, 12.6633235] },
  { type: 1, x: [12.8538576, 14.9830633, 17.6286053] },
  { type: 1, x: [16.7229337, 16.4525301, 12.4627193] },
  { type: 1, x: [14.0512024, 17.3954338, 12.7914031] },
  { type: 1, x: [18.2824041, 17.2316517, 14.6999911] },
  { type: 1, x: [16.2210303, 18.9719703, 13.5921161] },
  { type: 1, x: [17.2517138, 16.3080477, 17.1761679] },
  { type: 1, x: [16.0269821, 11.057668, 13.3712031] },
  { type: 1, x: [16.3354169, 10.9735982, 16.1199203] },
  { type: 1, x: [13.9149278, 18.8223918, 15.2392839] },
  { type: 1, x: [18.1136886, 12.5568867, 14.5759053] },
  { type: 1, x: [16.6484505, 13.6019749, 12.3834541] },
  { type: 1, x: [11.0468778, 15.1307677, 15.4471491] },
  { type: 1, x: [15.4347551, 14.8811145, 18.8773723] },
  { type: 2, x: [15.1502129, 14.9589199, 16.3394186] },
  { type: 2, x: [13.2837251, 13.9076768, 15.1588849] },
  { type: 2, x: [13.4425035, 16.3005712, 15.2145264] },
  { type: 2, x: [15.7534979, 16.8960911, 14.9738557] },
  { type: 2, x: [15.4967806, 13.024304, 14.8837423] },
  { type: 2, x: [14.8498337, 15.0410649, 13.6605697] },
  { type: 2, x: [17.0232044, 14.8712576, 14.7690776] }
];

const PANCAKE_LINES = [
  'pair_style smatb',
  'pair_coeff 1 1 2.89 10.85 3.18 0.1031 1.1895 4.08707719 5.0056268338740553',
  'pair_coeff 1 2 2.725 10.70 2.805 0.0977 1.2275 4.08707719 4.4340500673763259',
  'pair_coeff 2 2 2.56 10.55 2.43 0.0894 1.2799 3.62038672 4.4340500673763259',
];

/** Bulk fcc Au of examples/PACKAGES/smtbq/in.smatbBulkFCC: 8x8x8 cells, 2048 atoms, periodic. */
const bulkFcc = (a: number, n: number, s = 1): Atom[] => {
  const atoms: Atom[] = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
    for (const b of [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) {
      atoms.push({ type: 1, x: [(i + b[0]) * a * s, (j + b[1]) * a * s, (k + b[2]) * a * s] });
    }
  }
  return atoms;
};

/** Documented smatb energy per atom, E_i = sum alpha - sqrt(sum Xi^2), for a list of neighbour distances (independent of the engine). */
const smatbEi = (C: number[], rs: number[]): number => {
  const [R0, p, q, A, xi, Rsc, Rc] = C;
  const ea = A * Math.exp(-p * (Rsc / R0 - 1)), ex = xi * Math.exp(-q * (Rsc / R0 - 1));
  // polynomial c3 d^3 + c4 d^4 + c5 d^5, d = r - Rc, matching value, slope and curvature at Rsc (Cramer's rule)
  const d0 = Rsc - Rc;
  const M = [[d0 ** 3, d0 ** 4, d0 ** 5], [3 * d0 ** 2, 4 * d0 ** 3, 5 * d0 ** 4], [6 * d0, 12 * d0 ** 2, 20 * d0 ** 3]];
  const det = (m: number[][]) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const solve = (b: number[]) => {
    const D = det(M);
    return [0, 1, 2].map((c) => det(M.map((row, i) => row.map((v, k) => (k === c ? b[i] : v)))) / D);
  };
  const a = solve([ea, (-p / R0) * ea, (p / R0) ** 2 * ea]);
  const x = solve([ex, (-q / R0) * ex, (q / R0) ** 2 * ex]);
  let sa = 0, sx = 0;
  for (const r of rs) {
    if (r >= Rc) continue;
    let al: number, xv: number;
    if (r < Rsc) {
      al = A * Math.exp(-p * (r / R0 - 1));
      xv = xi * Math.exp(-q * (r / R0 - 1));
    } else {
      const d = r - Rc;
      al = a[0] * d ** 3 + a[1] * d ** 4 + a[2] * d ** 5;
      xv = x[0] * d ** 3 + x[1] * d ** 4 + x[2] * d ** 5;
    }
    sa += al;
    sx += xv * xv;
  }
  return sa - Math.sqrt(sx);
};

describe('pair_style smatb and smatb/single (batch 48)', () => {
  const SM = ['pair_style smatb', `pair_coeff 1 1 ${AU.join(' ')}`];

  it('the bulk fcc example reproduces the native step-0 E_pair of its log (-7800.9629)', async () => {
    const run = await evaluate(SM, bulkFcc(4.0782, 8), { boundary: 'p p p', box: 4.0782 * 8 });
    expect(run.error).toBeUndefined();
    expect(run.pe).toBeCloseTo(-7800.9629, 3);
  });

  it('the AgCu pancake reproduces the native step-0 E_pair of its log (-90.165962)', async () => {
    const run = await evaluate(PANCAKE_LINES, PANCAKE, { boundary: 'p p p', box: 30 });
    expect(run.error).toBeUndefined();
    expect(run.pe).toBeCloseTo(-90.165962, 5);
  });

  it('the dimer energy is 2 (alpha - Xi) at a distance in the exponential zone', async () => {
    const dimer = await evaluate(SM, [{ type: 1, x: [0, 0, 0] }, { type: 1, x: [2.9, 0.1, 0] }]);
    expect(dimer.error).toBeUndefined();
    expect(dimer.pe).toBeCloseTo(2 * smatbEi(AU, [Math.hypot(2.9, 0.1)]), 10);
  });

  it('forces agree with finite differences for a mixed-type cluster', async () => {
    const atoms: Atom[] = [
      { type: 1, x: [0, 0, 0] }, { type: 2, x: [2.6, 0.3, -0.2] }, { type: 1, x: [0.4, 2.7, 0.5] },
      { type: 2, x: [-0.6, -0.5, 2.9] }, { type: 1, x: [3.3, 3.1, 2.4] }, { type: 2, x: [4.3, 0.8, 1.6] },
    ];
    await expectForcesMatch(PANCAKE_LINES, atoms, 1e-6);
  });

  it('forces agree with finite differences in the polynomial zone (R_cs < r < R_c)', async () => {
    // distances 2.9 (exponential), 4.2 and 4.7 (polynomial zone), 4.95 (close to Rc)
    const atoms: Atom[] = [{ type: 1, x: [0, 0, 0] }, { type: 1, x: [4.2, 0, 0] }, { type: 1, x: [0.6, 4.7, 0.5] }, { type: 1, x: [-0.8, 0.5, -4.95] }];
    await expectForcesMatch(SM, atoms, 1e-6);
  });

  it('the virial pressure equals -dE/dV of a homogeneous scaling of the periodic bulk fcc cell', async () => {
    const a = 4.0782, n = 2, h = 1e-5;
    const energyAt = async (s: number) => (await evaluate(SM, bulkFcc(a, n, s), { boundary: 'p p p', box: a * n * s })).pe;
    const run = await evaluate(SM, bulkFcc(a, n), { boundary: 'p p p', box: a * n });
    const dEds = (await energyAt(1 + h) - (await energyAt(1 - h))) / (2 * h);
    const V = (a * n) ** 3;
    const bar = 1.602176634e6; // 1 eV/Angstrom^3 in bar
    const pFd = (-dEds / (3 * V)) * bar;
    expect(run.press[0]).toBeCloseTo(pFd, 0);
    expect(Math.abs(run.press[0] - pFd) / Math.abs(pFd)).toBeLessThan(1e-5);
  });

  it('refuses unset I,J pairs (smatb does not mix) and a bad cutoff order', async () => {
    const run = await evaluate(['pair_style smatb', `pair_coeff 1 1 ${AU.join(' ')}`, 'pair_coeff 2 2 2.56 10.55 2.43 0.0894 1.2799 3.62038672 4.4340500673763259'], [{ type: 1, x: [0, 0, 0] }, { type: 2, x: [2.5, 0, 0] }]);
    expect(run.error).toMatch(/not set/);
    const bad = await evaluate(['pair_style smatb', 'pair_coeff 1 1 2.88 10.35 4.178 0.210 1.818 5.0 4.9'], [{ type: 1, x: [0, 0, 0] }, { type: 1, x: [2.5, 0, 0] }]);
    expect(bad.error).toMatch(/Rcs < Rc/);
  });
});

/* ---------- edip (single element, Si.edip) ---------- */

describe('pair_style edip (batch 48)', () => {
  const EDIP = ['pair_style edip', 'pair_coeff * * Si.edip Si'];
  const SIEDIP = { 'Si.edip': potFile('Si.edip') };
  const ev = (atoms: Atom[]) => evaluate(EDIP, atoms, { files: SIEDIP });

  it('the dimer energy is 2 phi2(r, f(r)) (Justo form with the coordination x = (r - c)/(a - c))', async () => {
    // native (black box) with Si.edip: 2.2 -3.06777893228284, 2.6 -2.50756381140518, 2.8 -1.38372008298309, 3.0 -0.0774444986334451
    const native: [number, number][] = [[1.5, 0.147094019858047], [2.2, -3.06777893228284], [2.4, -3.03176422024769], [2.6, -2.50756381140518], [2.8, -1.38372008298309], [3.0, -0.0774444986334451]];
    for (const [r, e] of native) {
      const run = await ev([{ type: 1, x: [0, 0, 0] }, { type: 1, x: [r, 0, 0] }]);
      expect(run.error).toBeUndefined();
      expect(Math.abs(run.pe - e)).toBeLessThanOrEqual(1e-9 * Math.abs(e));
    }
  });

  it('the three-body energies of trimers match native values (angles 60 to 150 degrees)', async () => {
    // native (black box) with Si.edip, trimers with the centre at the origin and legs r at angle th
    const cases: [number, number, number][] = [
      [2.4, 60, -7.50997133388094], [2.4, 100, -5.81757639227646], [2.4, 109.4712206, -5.70963080000977],
      [2.4, 120, -5.53506700815967], [2.4, 150, -4.90668441989243], [2.7, 109.4712206, -3.93044873637878], [2.9, 130, -1.17791749690618],
    ];
    for (const [r, th, e] of cases) {
      const t = (th * Math.PI) / 180;
      const run = await ev([{ type: 1, x: [0, 0, 0] }, { type: 1, x: [r, 0, 0] }, { type: 1, x: [r * Math.cos(t), r * Math.sin(t), 0] }]);
      expect(run.error).toBeUndefined();
      expect(Math.abs(run.pe - e)).toBeLessThanOrEqual(1e-8 * Math.abs(e));
    }
  });

  it('forces agree with finite differences to the grid accuracy (dimer in the coordination zone, a 6-atom cluster)', async () => {
    // native pair_edip.cpp tabulates the functions and their derivatives separately on a grid (initGrids, linear
    // interpolation), so its forces match the energy's finite difference only to about 2e-4; the port keeps the grids
    await expectForcesMatch(EDIP, [{ type: 1, x: [0, 0, 0] }, { type: 1, x: [2.8, 0, 0] }], 1e-3, SIEDIP);
    await expectForcesMatch(EDIP, [
      { type: 1, x: [0, 0, 0] }, { type: 1, x: [2.4, 0.3, 0.1] }, { type: 1, x: [-0.5, 2.6, 0.4] },
      { type: 1, x: [0.2, -0.4, 2.7] }, { type: 1, x: [2.9, 2.5, 2.3] }, { type: 1, x: [-2.2, -1.9, -1.6] },
    ], 1e-3, SIEDIP);
  });

  it('refuses several element names and a missing entry; accepts pair_modify shift as native does', async () => {
    const multi = await evaluate(['pair_style edip', 'pair_coeff * * Si.edip Si C'], [{ type: 1, x: [0, 0, 0] }, { type: 2, x: [2.4, 0, 0] }], { files: SIEDIP });
    expect(multi.error).toMatch(/single element/);
    const none = await evaluate(['pair_style edip', 'pair_coeff * * Si.edip C'], [{ type: 1, x: [0, 0, 0] }, { type: 1, x: [2.4, 0, 0] }], { files: SIEDIP });
    expect(none.error).toMatch(/Potential file is missing an entry for: C C C/);
    const shift = await evaluate([...EDIP, 'pair_modify shift yes'], [{ type: 1, x: [0, 0, 0] }, { type: 1, x: [2.4, 0, 0] }], { files: SIEDIP });
    expect(shift.error).toBeUndefined();
  });
});

/* ---------- edip/multi (SiC.edip, Si and C) ---------- */

describe('pair_style edip/multi (batch 48)', () => {
  const MULTI = ['pair_style edip/multi', 'pair_coeff * * SiC.edip Si C'];
  const SIC = { 'SiC.edip': potFile('SiC.edip') };
  const T = (name: string): number => (name === 'Si' ? 1 : 2);

  it('dimers match native values (Si-C, Si-Si) with the (E_i, E_j, E_j) entries', async () => {
    // native (black box) with SiC.edip: Si-C 1.9 -4.73328422908288, 2.2 -3.0356442409666, 2.5 -6.4751951184357e-05;
    // Si-Si 2.2 -3.09890240983778, 2.5 -2.85923724377621, 2.9 -0.00509466465429816
    const cases: [string, string, number, number][] = [
      ['Si', 'C', 1.9, -4.73328422908288], ['Si', 'C', 2.2, -3.0356442409666], ['Si', 'C', 2.5, -6.4751951184357e-5],
      ['Si', 'Si', 2.2, -3.09890240983778], ['Si', 'Si', 2.5, -2.85923724377621], ['Si', 'Si', 2.9, -0.00509466465429816],
    ];
    for (const [a, b, r, e] of cases) {
      const run = await evaluate(MULTI, [{ type: T(a), x: [0, 0, 0] }, { type: T(b), x: [r, 0, 0] }], { files: SIC, ntypes: 2 });
      expect(run.error).toBeUndefined();
      expect(Math.abs(run.pe - e)).toBeLessThanOrEqual(1e-9 * Math.max(Math.abs(e), 1e-3));
    }
  });

  it('trimers with mixed centre and neighbour types match native values (three-body entries (E_i, E_j, E_k))', async () => {
    // native (black box) with SiC.edip: [centre, leg j, leg k, r1, r2, angle, energy]
    const cases: [string, string, string, number, number, number, number][] = [
      ['Si', 'Si', 'C', 2.3, 2.4, 109.47, -3.58115509746697], ['Si', 'C', 'Si', 2.3, 2.4, 109.47, -4.65657280104949],
      ['C', 'Si', 'Si', 1.9, 1.9, 109.47, -8.80322691847865], ['Si', 'C', 'C', 2.2, 2.6, 120, -3.0356442409666],
      ['C', 'C', 'Si', 2.0, 2.2, 100, -4.31751463506423],
    ];
    for (const [c, a, b, r1, r2, th, e] of cases) {
      const t = (th * Math.PI) / 180;
      const run = await evaluate(MULTI, [
        { type: T(c), x: [0, 0, 0] }, { type: T(a), x: [r1, 0, 0] }, { type: T(b), x: [r2 * Math.cos(t), r2 * Math.sin(t), 0] },
      ], { files: SIC, ntypes: 2 });
      expect(run.error).toBeUndefined();
      expect(Math.abs(run.pe - e)).toBeLessThanOrEqual(1e-9 * Math.abs(e));
    }
  });

  it('forces agree with finite differences for a mixed cluster', async () => {
    await expectForcesMatch(MULTI, [
      { type: 1, x: [0, 0, 0] }, { type: 2, x: [2.2, 0.4, 0.1] }, { type: 1, x: [-0.5, 2.5, 0.3] },
      { type: 2, x: [0.3, -0.6, 2.4] }, { type: 1, x: [2.6, 2.4, 2.2] },
    ], 1e-6, SIC, 2);
  });

  it('refuses a missing entry (e.g. C C C is in the file but Ge is not)', async () => {
    const run = await evaluate(['pair_style edip/multi', 'pair_coeff * * SiC.edip Si Ge'], [{ type: 1, x: [0, 0, 0] }, { type: 2, x: [2.2, 0, 0] }], { files: SIC });
    expect(run.error).toMatch(/Potential file is missing an entry for/);
  });
});
