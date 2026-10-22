import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { solveBanded } from '../src/engine/fix/ilves';
import type { EngineError, EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix ilves (src/engine/fix/ilves.ts): parsing, refusals and the constraints after each step. The numbers in the
 * expectations are native LAMMPS (black box) values measured on the probes described in the fix header.
 */

// two atoms of one bond stretched from 1.2 to r0 = 1.0, mass 12, no velocity
const DIMER = `LAMMPS data

2 atoms
1 bonds
1 atom types
1 bond types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 12.0

Atoms # bond

1 1 1 5.0 5.0 5.0
2 1 1 6.2 5.0 5.0

Bonds

1 1 1 2
`;

// four atoms: centre 1 bonded to 2, 3, 4 with masses 12 and 1, velocities given; angles from ANGLES
const STAR = (angles: string) => `LAMMPS data

4 atoms
3 bonds
${angles.split('\n').filter(Boolean).length} angles
2 atom types
1 bond types
1 angle types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 12.0
2 1.0

Atoms # angle

1 1 1 5.0 5.0 5.0
2 1 2 6.1 5.0 5.2
3 1 2 4.9 6.1 5.1
4 1 2 4.8 4.6 6.2

Velocities

1 0.005 -0.004 0.002
2 0.010 0.0 -0.020
3 -0.008 0.012 0.0
4 0.0 -0.010 0.015

Bonds

1 1 1 2
2 1 1 3
3 1 1 4

Angles

${angles}
`;

// three atoms: centre 2 (type 2, mass 16) bonded to 1 and 3, angle 1-2-3 (the probe of the fix header)
const ANGLE3 = `LAMMPS data

3 atoms
2 bonds
1 angles
2 atom types
1 bond types
1 angle types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 12.0
2 16.0

Atoms # angle

1 1 1 5.0 5.0 5.0
2 1 2 6.3 5.0 5.0
3 1 1 6.3 6.1 5.0

Velocities

1 0.0 0.0 0.0
2 0.0 0.0 0.0
3 0.0 0.0 0.0

Bonds

1 1 1 2
2 1 2 3

Angles

1 1 1 2 3
`;

const BASE = (data: string, extra: string, body: string) => `units real
atom_style ${data.includes('Angles') ? 'angle' : 'bond'}
read_data probe.data
pair_style zero 3.0
pair_coeff * *
bond_style harmonic
bond_coeff 1 100.0 1.0
${data.includes('Angles') ? 'angle_style harmonic\nangle_coeff 1 50.0 109.47' : ''}
neighbor 0.3 bin
timestep 1.0
fix 1 all nve
${extra}
${body}`;

/** Runs one input on a data file; returns the thermo rows, the dumped atoms (write_dump) and any engine error. */
const run = async (data: string, input: string) => {
  const rows: ThermoRow[] = [];
  const files = new Map<string, string>();
  const s = new Session({
    emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); },
    writeFile: (name, text, append) => files.set(name, (append ? files.get(name) ?? '' : '') + text),
  });
  s.addFile('probe.data', data);
  try {
    await s.execute(`${input}\nwrite_dump all custom probe.dump id type x y z vx vy vz fx fy fz modify format float %.17g sort id\n`);
    return { rows, error: null as EngineError | null, atoms: parseAtoms(files.get('probe.dump') ?? '') };
  } catch (e) {
    return { rows, error: e as EngineError, atoms: [] as number[][] };
  }
};

/** Rows of id-sorted atoms: [id, x, y, z, vx, vy, vz, fx, fy, fz]. */
const parseAtoms = (text: string): number[][] => {
  const lines = text.split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  if (k < 0) return [];
  return lines.slice(k + 1).filter((l) => l.trim()).map((l) => l.trim().split(/\s+/).map(Number));
};

describe('fix ilves parsing', () => {
  const bad = (fix: string) => run(DIMER, BASE(DIMER, fix, 'run 0'));

  it('needs the tol, iter and N arguments and a selector', async () => {
    expect((await bad('fix c all ilves 1e-4 25')).error?.message).toMatch(/Illegal fix ilves command: missing argument/);
    expect((await bad('fix c all ilves 1e-4 25 0')).error?.message).toMatch(/Illegal fix ilves command: missing argument/);
    expect((await bad('fix c all ilves 1e-4 25 0 mode fixed')).error?.message).toMatch(/no bond, angle, atom type or mass selected/);
    expect((await bad('fix c all ilves 0 25 0 b 1')).error?.message).toMatch(/tol must be > 0/);
    expect((await bad('fix c all ilves 1e-4 0 0 b 1')).error?.message).toMatch(/iter >= 1/);
  });

  it('rejects a selector value before b, a, t or m', async () => {
    expect((await bad('fix c all ilves 1e-4 25 0 1 b 1')).error?.message).toMatch(/must follow a constraint keyword/);
  });

  it('rejects unknown keyword values', async () => {
    expect((await bad('fix c all ilves 1e-4 25 0 b 1 mode sometimes')).error?.message).toMatch(/mode must be converge or fixed/);
    expect((await bad('fix c all ilves 1e-4 25 0 b 1 linearangle bend 170')).error?.message).toMatch(/linearangle style must be error, skip or restrain/);
    expect((await bad('fix c all ilves 1e-4 25 0 b 1 kbond -2')).error?.message).toMatch(/kbond must be > 0/);
    expect((await bad('fix c all ilves 1e-4 25 0 b 1 store maybe')).error?.message).toMatch(/store must be yes or no/);
  });

  it('allows one fix ilves per simulation', async () => {
    const input = BASE(DIMER, 'fix c all ilves 1e-4 25 0 b 1\nfix d all ilves 1e-4 25 0 b 1', 'run 0');
    expect((await run(DIMER, input)).error?.message).toMatch(/Only one fix ilves/);
  });

  it('minimization: each bond constraint is a restraint with 2000 * 5e5 * boltz (1e9 * boltz), energy in pe', async () => {
    // fix_ilves.cpp min_post_force: k_bond = kbond (default 5e5*boltz) times KBOND_MIN_FACTOR (2e3); the DIMER bond is
    // stretched from 1.0 to 1.2 and the bond itself is switched off, so step 0 has pe = k (r - r0)^2 = 0.04 k alone
    const { rows, error } = await run(DIMER, BASE(DIMER, 'fix c all ilves 1e-4 25 0 b 1\nthermo_style custom step pe', 'minimize 0 0 1 10'));
    expect(error).toBeNull();
    const k = 1e9 * UNIT_SYSTEMS.real.boltz;
    expect(Math.abs(rows[0].pe - 0.04 * k) / (0.04 * k)).toBeLessThan(1e-12);
  });

  it('refuses a near-linear angle type with the default error style', async () => {
    const angle = STAR('1 1 2 1 3');
    const input = BASE(angle, 'fix c all ilves 1e-12 50 0 b 1 a 1 linearangle error 110', 'run 0').replace('angle_coeff 1 50.0 109.47', 'angle_coeff 1 50.0 176.0');
    expect((await run(angle, input)).error?.message).toMatch(/angle type 1 has an equilibrium angle at or above the linearangle threshold of 110/);
  });
});

describe('fix ilves dynamics (native values)', () => {
  it('gives the setup force of a stretched dimer: 1.2 / (dt^2 ftm2v) = 2868.0688 (native)', async () => {
    const { atoms, error } = await run(DIMER, BASE(DIMER, 'fix c all ilves 1e-12 50 0 b 1', 'run 0'));
    expect(error).toBeNull();
    expect(atoms[0][8]).toBeCloseTo(2868.06883384018, 6);
    expect(atoms[1][8]).toBeCloseTo(-2868.06883384018, 6);
    // setup coordinates are not moved
    expect(atoms[1][2]).toBeCloseTo(6.2, 12);
  });

  it('stops after the native number of Newton steps at tol 1e-4 (two steps: 2866.1097704291; three: 2868.0687)', async () => {
    const two = await run(DIMER, BASE(DIMER, 'fix c all ilves 1e-4 2 0 b 1 mode fixed', 'run 0'));
    expect(two.atoms[0][8]).toBeCloseTo(2866.1097704291, 6);
    const conv = await run(DIMER, BASE(DIMER, 'fix c all ilves 1e-4 25 0 b 1', 'run 0'));
    expect(conv.atoms[0][8]).toBeCloseTo(2868.06870004265, 6);
  });

  it('reproduces the native unconverged velocity sweeps of a star with three angles (tol 1e-12, iter 50)', async () => {
    const angles = '1 1 2 1 3\n2 1 3 1 4\n3 1 2 1 4';
    const star = STAR(angles);
    const input = BASE(star, 'fix c all ilves 1e-12 50 0 b 1 a 1', 'run 0');
    const { atoms, error } = await run(star, input);
    expect(error).toBeNull();
    // native dump after setup: v1 = (-63.263448, -66.576655, -128.578187) (the Jacobi sweeps diverge here)
    expect(atoms[0][5]).toBeCloseTo(-63.263448, 4);
    expect(atoms[0][6]).toBeCloseTo(-66.576655, 4);
    expect(atoms[0][7]).toBeCloseTo(-128.578187, 4);
  });

  it('restrain: the A-C restraint energy (kbond 2000: 175.046181942578; default 5e5 kB: 86.9632363914275) is in pe', async () => {
    // three atoms (5,5,5) (6.3,5,5) (6.3,6.1,5), angle 176 degrees: the native probe values
    const molecule = ANGLE3;
    const body = 'thermo_style custom step pe ebond eangle f_c\nthermo 1\nrun 0';
    const withKb = await run(molecule, BASE(molecule, 'fix c all ilves 1e-12 50 0 b 1 a 1 linearangle restrain 170 kbond 2000', body).replace('angle_coeff 1 50.0 109.47', 'angle_coeff 1 50.0 176.0'));
    expect(withKb.error).toBeNull();
    expect(withKb.rows[0].pe).toBeCloseTo(175.046181942578, 6);
    expect(withKb.rows[0].eangle).toBeCloseTo(0, 12);
    const dflt = await run(molecule, BASE(molecule, 'fix c all ilves 1e-12 50 0 b 1 a 1 linearangle restrain 170', body).replace('angle_coeff 1 50.0 109.47', 'angle_coeff 1 50.0 176.0'));
    expect(dflt.rows[0].pe).toBeCloseTo(86.9632363914275, 6);
  });

  it('keeps the constraints inside tol after each step of a bond-only run', async () => {
    // a 3-atom chain stretched by 0.3 and 0.1 Angstrom: from step 2 on the bonds are within tol of their length
    const chain = `LAMMPS data

3 atoms
2 bonds
1 atom types
1 bond types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 12.0

Atoms # bond

1 1 1 5.0 5.0 5.0
2 1 1 6.3 5.0 5.0
3 1 1 6.3 6.1 5.0

Velocities

1 0.01 0.02 0.0
2 0.0 0.0 0.03
3 0.02 -0.01 0.0

Bonds

1 1 1 2
2 1 2 3
`;
    const tol = 1e-6;
    // measured with native LAMMPS (black box): the positions after step 1 are not yet constrained (a stretched dimer
    // is at 1.10 after step 1 and at 1.00 after step 2), so the check starts at step 2
    for (let n = 2; n <= 10; n++) {
      const input = BASE(chain, `fix c all ilves ${tol} 25 0 b 1`, `run ${n}`);
      const { atoms, error } = await run(chain, input);
      expect(error).toBeNull();
      const dist = (a: number[], b: number[]) => Math.hypot(a[2] - b[2], a[3] - b[3], a[4] - b[4]);
      // positions after the step: the bonds are the predicted ones, so their error is tol-level
      expect(Math.abs(dist(atoms[0], atoms[1]) - 1.0)).toBeLessThan(1e-4);
      expect(Math.abs(dist(atoms[1], atoms[2]) - 1.0)).toBeLessThan(1e-4);
    }
  });
});

describe('solveBanded', () => {
  it('solves a banded system with partial pivoting (residual 1e-12)', () => {
    let seed = 7;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) - 0.5;
    for (const [n, bw] of [[1, 0], [5, 1], [12, 2], [30, 3]] as const) {
      const entries: { row: number; col: number; v: number }[] = [];
      const dense = Array.from({ length: n }, () => new Array(n).fill(0));
      for (let i = 0; i < n; i++) for (let j = Math.max(0, i - bw); j <= Math.min(n - 1, i + bw); j++) {
        const v = i === j ? 2 + rnd() : rnd();
        entries.push({ row: i, col: j, v });
        dense[i][j] = v;
      }
      const rhs = Array.from({ length: n }, () => rnd());
      const x = solveBanded(n, bw, bw, entries, rhs);
      expect(x).not.toBeNull();
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += dense[i][j] * x![j];
        expect(Math.abs(s - rhs[i])).toBeLessThan(1e-12);
      }
    }
  });

  it('returns null for a singular matrix', () => {
    expect(solveBanded(2, 1, 1, [{ row: 0, col: 0, v: 1 }, { row: 0, col: 1, v: 1 }, { row: 1, col: 0, v: 1 }, { row: 1, col: 1, v: 1 }], [1, 1])).toBeNull();
  });
});
