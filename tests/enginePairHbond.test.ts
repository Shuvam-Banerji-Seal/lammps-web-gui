import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError, EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * pair_style hbond/dreiding/* (docs.lammps.org/pair_hbond_dreiding.html): the
 * energy and force consistency, the measured angle factor and cutoffs, the
 * coefficient rules and the error paths. Agreement with native LAMMPS on a
 * whole system is covered by the w14hbond_* oracle cases.
 */

type Atom = { mol: number; type: number; x: number; y: number; z: number };

/** A donor D (type 1) bonded to hydrogen H (type 2) and an acceptor A (type 3), in a box of length 60, non-periodic. */
const dataText = (atoms: Atom[], bonds: [number, number][], ntypes = 3, box = 60): string => {
  const lines = [
    'hbond test', '', `${atoms.length} atoms`, `${bonds.length} bonds`, `${ntypes} atom types`, '1 bond types', '',
    `0 ${box} xlo xhi`, `0 ${box} ylo yhi`, `0 ${box} zlo zhi`, '', 'Masses', '',
  ];
  for (let t = 1; t <= ntypes; t++) lines.push(`${t} 1.0`);
  lines.push('', 'Atoms # full', '');
  atoms.forEach((a, i) => lines.push(`${i + 1} ${a.mol} ${a.type} 0.0 ${a.x} ${a.y} ${a.z}`));
  lines.push('', 'Bonds', '');
  bonds.forEach(([a, b], k) => lines.push(`${k + 1} 1 ${a} ${b}`));
  return lines.join('\n') + '\n';
};

interface Result { pe: number; forces: number[][] }

/** Runs pair commands on the system; returns evdwl and the forces of every atom (bonds are switched off). */
const evaluate = async (atoms: Atom[], bonds: [number, number][], pairCmds: string, box = 60, periodic = false): Promise<Result> => {
  const rows: ThermoRow[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); },
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  session.addFile('probe.data', dataText(atoms, bonds, 4, box));
  await session.execute([
    'units real', 'atom_style full', `boundary ${periodic ? 'p p p' : 'f f f'}`,
    'read_data probe.data', 'bond_style harmonic', 'bond_coeff 1 0.0 1.0',
    pairCmds,
    'thermo_style custom step evdwl', 'thermo_modify format float %.17g', 'run 0',
    'write_dump all custom probe.dump id fx fy fz modify format float %.17g sort id',
  ].join('\n'));
  const dump = files.get('probe.dump') ?? '';
  const lines = dump.split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const forces = lines.slice(k + 1).filter((l) => l.trim()).map((l) => l.trim().split(/\s+/).slice(1).map(Number));
  return { pe: rows[0].evdwl, forces };
};

/** A donor-H...acceptor triple with a generic geometry (no special symmetry). */
const triple = (): Atom[] => [
  { mol: 1, type: 1, x: 30.0, y: 30.0, z: 30.0 },
  { mol: 1, type: 2, x: 30.9, y: 30.4, z: 29.7 },
  { mol: 2, type: 3, x: 32.5, y: 31.1, z: 31.9 },
];
const TRIPLE_BONDS: [number, number][] = [[1, 2]];

/** Finite-difference check: the analytic forces equal -dE/dx for every atom and component. */
const fdCheck = async (atoms: Atom[], bonds: [number, number][], pairCmds: string, box = 60, h = 1e-6, periodic = false) => {
  const base = await evaluate(atoms, bonds, pairCmds, box, periodic);
  let worst = 0;
  for (let i = 0; i < atoms.length; i++) {
    for (let c = 0; c < 3; c++) {
      const key = c === 0 ? 'x' : c === 1 ? 'y' : 'z';
      const plus = atoms.map((a, j) => (j === i ? { ...a, [key]: a[key] + h } : a));
      const minus = atoms.map((a, j) => (j === i ? { ...a, [key]: a[key] - h } : a));
      const ep = (await evaluate(plus, bonds, pairCmds, box, periodic)).pe;
      const em = (await evaluate(minus, bonds, pairCmds, box, periodic)).pe;
      const fd = -(ep - em) / (2 * h);
      const an = base.forces[i][c];
      worst = Math.max(worst, Math.abs(fd - an) / Math.max(1, Math.abs(fd)));
    }
  }
  return worst;
};

const LJ = (eps: number, sig: number, n: number, rin: number, rout: number, tc: number) =>
  `pair_style hybrid/overlay lj/cut 10.0 hbond/dreiding/lj ${n} ${rin} ${rout} ${tc}\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/lj 2 i ${eps} ${sig}`;

describe('hbond/dreiding: forces are the gradient of the energy', () => {
  it('lj, switching band, generic angle (finite differences)', async () => {
    const worst = await fdCheck(triple(), TRIPLE_BONDS, LJ(1.3, 2.1, 4, 2.5, 4.0, 0.0));
    expect(worst).toBeLessThan(1e-5);
  });

  it('morse with n = 2 and an angle cutoff (finite differences)', async () => {
    const cmds = 'pair_style hybrid/overlay lj/cut 10.0 hbond/dreiding/morse 2 2.5 4.0 60.0\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/morse 2 i 1.5 1.2 2.5';
    const worst = await fdCheck(triple(), TRIPLE_BONDS, cmds);
    expect(worst).toBeLessThan(1e-5);
  });

  it('angleoffset variants (finite differences)', async () => {
    const lj = 'pair_style hybrid/overlay lj/cut 10.0 hbond/dreiding/lj/angleoffset 1 2.5 4.0 60.0 150.0\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/lj/angleoffset 2 i 1.3 2.1';
    const morse = 'pair_style hybrid/overlay lj/cut 10.0 hbond/dreiding/morse/angleoffset 1 2.5 4.0 60.0 150.0\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/morse/angleoffset 2 i 1.5 1.2 2.5';
    expect(await fdCheck(triple(), TRIPLE_BONDS, lj)).toBeLessThan(1e-5);
    expect(await fdCheck(triple(), TRIPLE_BONDS, morse)).toBeLessThan(1e-5);
  });

  it('periodic box: a donor-hydrogen bond and the acceptor across the boundary (finite differences)', async () => {
    const atoms: Atom[] = [
      { mol: 1, type: 1, x: 0.4, y: 5.0, z: 5.0 },
      { mol: 1, type: 2, x: 15.3, y: 5.2, z: 4.9 },
      { mol: 2, type: 3, x: 13.9, y: 5.6, z: 5.4 },
    ];
    const cmds = LJ(1.3, 2.1, 2, 2.5, 4.0, 0.0);
    // the box is 16 long in x: the hydrogen is at image x = -0.7, the acceptor at x = -2.1
    expect(await fdCheck(atoms, TRIPLE_BONDS, cmds, 16, 1e-6, true)).toBeLessThan(1e-5);
  });
});

describe('hbond/dreiding: measured with native LAMMPS (black box)', () => {
  // donor at (-1,0,0) from hydrogen at the origin, acceptor placed at distance 3.0 from the donor and angle theta at H
  const geom = (theta: number, r: number): Atom[] => {
    const a = (theta * Math.PI) / 180;
    const c = -Math.cos(a), s = Math.sin(a);
    const b = 2 * c, cc = 1 - r * r;
    const R = (-b + Math.sqrt(b * b - 4 * cc)) / 2;
    return [
      { mol: 1, type: 1, x: 29.0, y: 30.0, z: 30.0 },
      { mol: 1, type: 2, x: 30.0, y: 30.0, z: 30.0 },
      { mol: 2, type: 3, x: 30 + R * c, y: 30 + R * s, z: 30 },
    ];
  };
  const radial = (r: number, eps: number, sig: number) => eps * (5 * (sig / r) ** 12 - 6 * (sig / r) ** 10);

  it('lj, n = 1, theta = 120: energy matches the measured value', async () => {
    const res = await evaluate(geom(120, 3.0), [[1, 2]], LJ(1.0, 2.0, 1, 4.0, 5.0, 0.0));
    // Measured with native LAMMPS (black box): E = 0.0327562231743516 for this geometry
    expect(res.pe).toBeCloseTo(0.0327562231743516, 12);
    expect(res.pe).toBeCloseTo(radial(3.0, 1, 2) * Math.cos((120 * Math.PI) / 180), 12);
  });

  it('angle cutoff: zero below the cutoff, nonzero at and above it', async () => {
    const below = await evaluate(geom(89, 3.0), [[1, 2]], LJ(1.0, 2.0, 1, 4.0, 5.0, 90.0));
    const above = await evaluate(geom(91, 3.0), [[1, 2]], LJ(1.0, 2.0, 1, 4.0, 5.0, 90.0));
    // Measured with native LAMMPS (black box): E = 0.00114334984038608 at theta = 91 with the 90 degree cutoff
    expect(below.pe).toBe(0);
    expect(above.pe).toBeCloseTo(0.00114334984038608, 12);
  });

  it('morse, n = 2, switching band: matches the measured value', async () => {
    const cmds = 'pair_style hybrid/overlay lj/cut 10.0 hbond/dreiding/morse 2 2.5 4.0 60\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/morse 2 i 1.5 1.2 2.5';
    const res = await evaluate(geom(130, 3.0), [[1, 2]], cmds);
    // Measured with native LAMMPS (black box): E = -0.397947176310608
    expect(res.pe).toBeCloseTo(-0.397947176310608, 12);
  });

  it('angleoffset: factor (cos(theta - theta_eq + 180))^n, measured at theta = 60 with theta_eq = 166.6', async () => {
    const cmds = 'pair_style hybrid/overlay lj/cut 10.0 hbond/dreiding/lj/angleoffset 1 4.0 5.0 0.0 166.6\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/lj/angleoffset 2 i 1.0 2.0';
    const res = await evaluate(geom(60, 3.0), [[1, 2]], cmds);
    // Measured with native LAMMPS (black box): E / radial = 0.28568836740478126 at theta = 60
    expect(res.pe / radial(3.0, 1, 2)).toBeCloseTo(0.28568836740478126, 10);
  });

  it('two hydrogens of the same type on one donor add up (energy is the sum)', async () => {
    // the geometry of the native probe: donor at (29,30,30), hydrogens at (30,30,30) and D + (-0.3, 0.9, 0.4)
    const d = { x: 29.0, y: 30.0, z: 30.0 };
    const atoms: Atom[] = [
      { mol: 1, type: 1, x: d.x, y: d.y, z: d.z },
      { mol: 1, type: 2, x: 30.0, y: 30.0, z: 30.0 },
      { mol: 1, type: 2, x: d.x - 0.3, y: d.y + 0.9, z: d.z + 0.4 },
      { mol: 2, type: 3, x: geom(170, 3.0)[2].x, y: geom(170, 3.0)[2].y, z: geom(170, 3.0)[2].z },
    ];
    const one = await evaluate(atoms.filter((_, i) => i !== 2), [[1, 2]], LJ(1.0, 2.0, 1, 4.0, 5.0, 0.0));
    const two = await evaluate(atoms, [[1, 2], [1, 3]], LJ(1.0, 2.0, 1, 4.0, 5.0, 0.0));
    // Measured with native LAMMPS (black box): one hydrogen 0.0645171650830171, both 0.0333671881230873 (count 2)
    expect(one.pe).toBeCloseTo(0.0645171650830171, 12);
    expect(two.pe).toBeCloseTo(0.0333671881230873, 12);
  });

  it('special_bonds weight multiplies the pair energy (0.5 halves it, 0.0 excludes it)', async () => {
    const atoms: Atom[] = [
      { mol: 1, type: 1, x: 29.0, y: 30.0, z: 30.0 },
      { mol: 1, type: 2, x: 30.0, y: 30.0, z: 30.0 },
      { mol: 1, type: 3, x: 30 + 1.5, y: 30 + 2.0, z: 30.0 },
    ];
    // acceptor and donor are bonded (1-2 pair): the special weight decides whether and how much it counts
    // default special_bonds weights are 0.0 (the pair is excluded), so the weight 1.0 is set explicitly
    const full = await evaluate(atoms, [[1, 2], [1, 3]], 'special_bonds lj/coul 1.0 1.0 1.0\n' + LJ(1.0, 2.0, 1, 4.0, 5.0, 0.0));
    const dflt = await evaluate(atoms, [[1, 2], [1, 3]], LJ(1.0, 2.0, 1, 4.0, 5.0, 0.0));
    expect(dflt.pe).toBe(0);
    const half = await evaluate(atoms, [[1, 2], [1, 3]], 'special_bonds lj/coul 0.5 0.5 0.5\n' + LJ(1.0, 2.0, 1, 4.0, 5.0, 0.0));
    const none = await evaluate(atoms, [[1, 2], [1, 3]], 'special_bonds lj/coul 0.0 0.0 0.0\n' + LJ(1.0, 2.0, 1, 4.0, 5.0, 0.0));
    expect(half.pe).toBeCloseTo(0.5 * full.pe, 12);
    expect(none.pe).toBe(0);
  });

  it('a repeated pair_coeff for the same hydrogen type replaces the earlier one', async () => {
    const replaced = await evaluate(triple(), TRIPLE_BONDS, LJ(1.0, 2.0, 1, 4.0, 5.0, 0.0) + '\npair_coeff 1 3 hbond/dreiding/lj 2 i 2.0 2.0');
    const single = await evaluate(triple(), TRIPLE_BONDS, LJ(2.0, 2.0, 1, 4.0, 5.0, 0.0));
    expect(replaced.pe).toBeCloseTo(single.pe, 12);
  });
});

describe('hbond/dreiding: argument and option errors', () => {
  const run = async (text: string) => {
    let error: EngineError | null = null;
    const session = new Session({ emit: () => {}, writeFile: () => {} });
    session.addFile('probe.data', dataText(triple(), TRIPLE_BONDS));
    try {
      await session.execute(['units real', 'atom_style full', 'boundary f f f', 'read_data probe.data', text, 'run 0'].join('\n'));
    } catch (e) {
      error = e as EngineError;
    }
    return error;
  };
  const base = 'pair_style hybrid/overlay lj/cut 10.0 hbond/dreiding/lj';

  it('pair_style needs N, inner, outer, angle cutoff (plus equilibrium angle for angleoffset)', async () => {
    expect(await run(`${base} 4 3.0 4.5`)).not.toBeNull();
    expect(await run(`${base} 4 3.0 4.5 90.0 170.0`)).not.toBeNull();
    expect(await run('pair_style hybrid/overlay lj/cut 10.0 hbond/dreiding/lj/angleoffset 4 3.0 4.5 90.0')).not.toBeNull();
  });

  it('inner cutoff must be below the outer cutoff; N must be an integer', async () => {
    expect(await run(`${base} 4 4.5 3.0 90.0\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/lj 2 i 1.0 2.0`)).not.toBeNull();
    expect(await run(`${base} 1.5 3.0 4.5 90.0\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/lj 2 i 1.0 2.0`)).not.toBeNull();
  });

  it('pair_modify shift is not supported', async () => {
    const err = await run(`${base} 4 3.0 4.5 90.0\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/lj 2 i 1.0 2.0\npair_modify shift yes`);
    expect(err).not.toBeNull();
    expect(String(err?.message ?? '')).toMatch(/shift/);
  });

  it('the equilibrium angle is only accepted by the angleoffset styles', async () => {
    const plain = await run(`${base} 4 3.0 4.5 90.0\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/lj 2 i 1.0 2.0 4 3.0 4.5 90.0 170.0`);
    expect(plain).not.toBeNull();
    const offset = await run('pair_style hybrid/overlay lj/cut 10.0 hbond/dreiding/lj/angleoffset 4 3.0 4.5 90.0 170.0\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/lj/angleoffset 2 i 1.0 2.0 4 3.0 4.5 90.0 170.0');
    expect(offset).toBeNull();
  });

  it('the donor flag must be i or j', async () => {
    expect(await run(`${base} 4 3.0 4.5 90.0\npair_coeff * * lj/cut 0.0 1.0\npair_coeff 1 3 hbond/dreiding/lj 2 k 1.0 2.0`)).not.toBeNull();
  });
});
