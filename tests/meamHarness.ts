import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/** The synthetic potential files of tests/oracle (the inputs of the native-measured cases), by base name. */
export const oracleFiles = (...names: string[]): Record<string, string> =>
  Object.fromEntries(names.map((n) => [n, readFileSync(join(__dirname, 'oracle', n), 'utf8')]));

/*
 * Shared driver for the pair_style meam tests (tests/engineMeam*.test.ts, tests/enginePairMeam*.test.ts).
 * Every energy and force comes from the engine: a small LAMMPS input is run through a Session with the
 * potential files passed through the session's file map, exactly as a user would run it.
 *
 * Geometry: the atoms are placed with create_atoms single in a periodic cubic box of side `box` (default 60 A,
 * far larger than rc, so a cluster sees no periodic image). Coordinates are shifted by `offset` (default box / 2,
 * i.e. a cluster is centred in the box) and wrapped into [0, box), which leaves the energy and forces unchanged.
 * Crystals pass offset 0 and their cell length as box.
 */

export interface MeamSetup {
  /** Potential files by name, e.g. { 'lib.meam': '...', 'par.meam': '...' }. */
  files: Record<string, string>;
  /** Arguments of pair_coeff after "* *", e.g. "lib.meam Cu par.meam Cu". */
  coeff: string;
  /** Atom types (1-based), one per atom. */
  types: number[];
  /** Flat coordinates x0 y0 z0 x1 ... in A. */
  x: ArrayLike<number>;
  /** Number of atom types of the box (default: the largest type in `types`). */
  ntypes?: number;
  /** Cubic periodic box side in A (default 60). */
  box?: number;
  /** Shift added before wrapping into the box (default box / 2). */
  offset?: number;
  /** Extra input lines between create_box and pair_style (e.g. "neigh_modify ..."). */
  pre?: string[];
}

export interface MeamResult {
  /** Potential energy from thermo (metal units, eV). */
  pe: number;
  /** Per-atom forces in atom id order (eV/A). */
  F: number[][];
}

const wrap = (v: number, L: number): number => {
  const w = v - L * Math.floor(v / L);
  return w >= L ? 0 : w;
};

/** The LAMMPS input of a setup (one run 0). */
export const meamScript = (s: MeamSetup): string => {
  const L = s.box ?? 60;
  const off = s.offset ?? L / 2;
  const n = s.types.length;
  const nt = s.ntypes ?? s.types.reduce((m, t) => Math.max(m, t), 0);
  const lines = [
    'units metal',
    'atom_style atomic',
    'boundary p p p',
    `region box block 0 ${L} 0 ${L} 0 ${L}`,
    `create_box ${nt} box`,
  ];
  for (let i = 0; i < n; i++) {
    const p = [0, 1, 2].map((c) => wrap(Number(s.x[3 * i + c]) + off, L));
    lines.push(`create_atoms ${s.types[i]} single ${p[0]} ${p[1]} ${p[2]} units box`);
  }
  for (let t = 1; t <= nt; t++) lines.push(`mass ${t} 1.0`);
  lines.push(...(s.pre ?? []));
  lines.push(`pair_style meam`, `pair_coeff * * ${s.coeff}`);
  lines.push('thermo_style custom step pe', 'thermo_modify format float %.17g', 'run 0');
  lines.push('write_dump all custom meam_forces.dump id fx fy fz modify format float %.17g sort id');
  return lines.join('\n') + '\n';
};

/** Runs a setup through the engine; resolves to the energy and per-atom forces, or rejects with the engine error. */
export const runMeam = async (s: MeamSetup): Promise<MeamResult> => {
  const events: EngineEvent[] = [];
  const written = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (name, text) => written.set(name, text),
  });
  for (const [name, text] of Object.entries(s.files)) session.addFile(name, text);
  await session.execute(meamScript(s));
  const rows = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row as ThermoRow);
  const lines = (written.get('meam_forces.dump') ?? '').trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const F = lines.slice(k + 1).map((l) => l.trim().split(/\s+/).slice(1).map(Number));
  return { pe: Number(rows[rows.length - 1].pe), F };
};

/** Energy only (the finite-difference probes). */
export const meamPe = async (s: MeamSetup): Promise<number> => (await runMeam(s)).pe;

/**
 * Worst relative error of the engine forces against central finite differences of the engine energy,
 * over every coordinate. |F| is compared with max(|fd|, floor) so that near-zero components do not dominate.
 */
export const worstForceError = async (s: MeamSetup, h = 1e-5, floor = 1e-2): Promise<number> => {
  const { F } = await runMeam(s);
  let worst = 0;
  for (let q = 0; q < s.x.length; q++) {
    const xp = Array.from(s.x, Number), xm = Array.from(s.x, Number);
    xp[q] += h;
    xm[q] -= h;
    const ep = await meamPe({ ...s, x: xp });
    const em = await meamPe({ ...s, x: xm });
    const fd = -(ep - em) / (2 * h);
    const err = Math.abs(F[Math.floor(q / 3)][q % 3] - fd) / Math.max(Math.abs(fd), floor);
    if (err > worst) worst = err;
  }
  return worst;
};

/** Deterministic LCG in [0, 1). */
export const lcg = (seed: number): (() => number) => {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
};

/** Flat coordinates with a deterministic displacement of up to amp per component (in input order). */
export const perturbed = (x: ArrayLike<number>, amp: number, seed: number): number[] => {
  const R = lcg(seed);
  return Array.from(x, (v) => v + (R() * 2 - 1) * amp);
};

/** Cubic cell of a Bravais+basis structure: n x n x n conventional cells of side a (flat coordinates). */
export const cellAtoms = (basis: number[][], a: number, n: number): number[] => {
  const p: number[] = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < n; k++)
        for (const b of basis) p.push((i + b[0]) * a, (j + b[1]) * a, (k + b[2]) * a);
  return p;
};

/** Fcc conventional-cell atoms (n x n x n cells of side a), each coordinate displaced by up to amp. */
export const fccCellSetup = (a: number, n: number, amp: number, seed: number): { x: number[]; types: number[]; box: number } => {
  const x = perturbed(cellAtoms([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], a, n), amp, seed);
  return { x, types: Array.from({ length: x.length / 3 }, () => 1), box: n * a };
};

/** Flat coordinates of points with a deterministic displacement of up to amp per component. */
export const jittered = (pts: number[][], amp: number, seed: number): number[] => perturbed(pts.flat(), amp, seed);

/** Relative difference |a - b| / |b|. */
export const relErr = (a: number, b: number): number => Math.abs(a - b) / Math.abs(b);
