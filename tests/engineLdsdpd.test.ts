import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { LddIndicatorDpd, LddIndicatorLucy } from '../src/engine-gpl/ldd/indicators';

// Pair styles local/density, ldd and sdpd/taitwater/isothermal (src/engine-gpl/localdensity, ldd, sdpd). The parity
// cases tests/oracle/w112ld_*.in and w112sdpd_*.in compare whole runs with native LAMMPS; these tests cover the
// refusals and the normalisation of the ldd weight functions.

const newSession = () => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  return { session, events };
};

/** The error a script stops with ('' when it runs through). */
const errorOf = async (session: Session, events: EngineEvent[], script: string): Promise<string> => {
  try {
    await session.execute(script);
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
  const e = events.find((ev) => ev.kind === 'error');
  return e && 'message' in e ? String(e.message) : '';
};

const BOX = `units lj
atom_style atomic
region box block 0 5 0 5 0 5
create_box 2 box
create_atoms 1 random 10 1 box
create_atoms 2 random 10 2 box
mass * 1.0
`;

/** Numerical integral of f over the sphere (dim 3) or disk (dim 2) of radius rc. */
const integrate = (f: (r: number) => number, rc: number, dim: 2 | 3): number => {
  const n = 20000;
  let sum = 0;
  for (let k = 0; k < n; k++) {
    const r = (k + 0.5) * rc / n;
    sum += (dim === 3 ? 4 * Math.PI * r * r : 2 * Math.PI * r) * f(r) * rc / n;
  }
  return sum;
};

describe('ldd weight functions', () => {
  it('are normalised to one over the sphere (3d) and the disk (2d)', () => {
    const rc = 2.5;
    expect(integrate((r) => new LddIndicatorLucy(0, rc, 3).w(r), rc, 3)).toBeCloseTo(1, 5);
    expect(integrate((r) => new LddIndicatorDpd(0, rc, 3).w(r), rc, 3)).toBeCloseTo(1, 5);
    expect(integrate((r) => new LddIndicatorLucy(0, rc, 2).w(r), rc, 2)).toBeCloseTo(1, 5);
    expect(integrate((r) => new LddIndicatorDpd(0, rc, 2).w(r), rc, 2)).toBeCloseTo(1, 5);
  });

  it('vanish at the cutoff with their derivative (lucy) and the derivative matches a finite difference', () => {
    const w = new LddIndicatorLucy(0, 3, 3);
    expect(w.w(3)).toBe(0);
    expect(w.wp(3)).toBe(0);
    const h = 1e-6, r = 1.1;
    expect((w.w(r + h) - w.w(r - h)) / (2 * h)).toBeCloseTo(w.wp(r), 6);
  });
});

describe('pair local/density', () => {
  it('refuses pair_style arguments and a pair_coeff that is not * * <table>', async () => {
    const { session, events } = newSession();
    expect(await errorOf(session, events, `${BOX}pair_style local/density 3.0\n`)).toMatch(/Illegal pair_style command/);
    const s2 = newSession();
    expect(await errorOf(s2.session, s2.events, `${BOX}pair_style local/density\npair_coeff 1 1 ld.table\n`)).toMatch(/Incorrect args for pair coefficients/);
  });

  it('stops with a named error when the table file cannot be read', async () => {
    const { session, events } = newSession();
    const msg = await errorOf(session, events, `${BOX}pair_style local/density\npair_coeff * * missing_table.txt\nrun 0\n`);
    expect(msg).toMatch(/missing_table\.txt/);
  });
});

describe('pair ldd', () => {
  it('names an unsupported indicator and potential', async () => {
    const shell = newSession();
    const file = 'A A indicator shell 0.0 2.0 self no potential quadratic 1 0 0\n';
    shell.session.addFile('ldd_shell.ldd', file);
    expect(await errorOf(shell.session, shell.events, `${BOX}pair_style ldd\npair_coeff * * ldd_shell.ldd A B\n`)).toMatch(/'shell'/);
    const constant = newSession();
    constant.session.addFile('ldd_const.ldd', 'A A indicator dpd 0.0 2.0 self no potential constant 1\n');
    expect(await errorOf(constant.session, constant.events, `${BOX}pair_style ldd\npair_coeff * * ldd_const.ldd A B\n`)).toMatch(/'constant'/);
  });

  it('requires an entry for every ordered species pair', async () => {
    const { session, events } = newSession();
    session.addFile('ldd_one.ldd', 'A A indicator dpd 0.0 2.0 self no potential quadratic 1 0 0\n');
    expect(await errorOf(session, events, `${BOX}pair_style ldd\npair_coeff * * ldd_one.ldd A B\n`)).toMatch(/Missing entry for species pair A B/);
  });

  it('runs a two-species system with the dpd indicator and the quadratic potential', async () => {
    const { session, events } = newSession();
    session.addFile('ldd_full.ldd', [
      'A A indicator dpd 0.0 2.0 self no potential quadratic 0.1 0.0 0.0',
      'A B indicator dpd 0.0 2.0 self no potential quadratic 0.1 0.0 0.0',
      'B A indicator dpd 0.0 2.0 self no potential quadratic 0.1 0.0 0.0',
      'B B indicator dpd 0.0 2.0 self no potential quadratic 0.1 0.0 0.0',
    ].join('\n') + '\n');
    expect(await errorOf(session, events, `${BOX}pair_style ldd\npair_coeff * * ldd_full.ldd A B\nrun 2\n`)).toBe('');
  });
});

describe('pair sdpd/taitwater/isothermal', () => {
  it('checks its settings', async () => {
    const a = newSession();
    expect(await errorOf(a.session, a.events, 'units lj\natom_style sph\npair_style sdpd/taitwater/isothermal 1.0\n')).toMatch(/Illegal number of arguments/);
    const b = newSession();
    expect(await errorOf(b.session, b.events, 'units lj\natom_style sph\npair_style sdpd/taitwater/isothermal -1.0 0.5\n')).toMatch(/Temperature must be positive/);
  });

  it('runs two sph particles with thermal forces', async () => {
    const { session, events } = newSession();
    const script = `units lj
atom_style sph
dimension 2
boundary f f p
region box block -5 5 -5 5 -0.5 0.5
create_box 1 box
create_atoms 1 single 0.0 0.0 0.0 units box
create_atoms 1 single 0.5 0.0 0.0 units box
mass * 1.0
set group all sph/rho 1.0
pair_style sdpd/taitwater/isothermal 1.0 0.5 7
pair_coeff * * 1.0 10.0 1.0
fix 1 all sph
run 2
`;
    expect(await errorOf(session, events, script)).toBe('');
  });
});

describe('the DPD-SMOOTH example fixes are not in the engine', () => {
  it('names the variable style of fix meso/move in its error', async () => {
    const { session, events } = newSession();
    const base = `units lj
atom_style sph
dimension 2
boundary f f p
region box block -5 5 -5 5 -0.5 0.5
create_box 1 box
create_atoms 1 single 0.0 0.0 0.0 units box
mass * 1.0
set group all sph/rho 1.0
`;
    expect(await errorOf(session, events, `${base}fix 3 all meso/move variable v_a NULL NULL NULL NULL NULL\n`)).toMatch(/meso\/move/);
  });
});
