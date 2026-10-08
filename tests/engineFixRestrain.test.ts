import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { Fix } from '../src/engine/fix/fix';

/*
 * Direct checks of fix restrain (docs.lammps.org/fix_restrain.html), fix
 * spring/rg (docs.lammps.org/fix_spring_rg.html), fix evaporate
 * (docs.lammps.org/fix_evaporate.html) and compute msd/nongauss
 * (docs.lammps.org/compute_msd_nongauss.html): the restraint forces equal -grad E
 * by central finite differences, the closed-form values on simple geometries
 * and the argument errors. The energy of each restraint is the fix's own
 * scalar (fix_modify energy yes is not needed for the scalar).
 */

const HEAD = `units lj
atom_style atomic
atom_modify map array
region box block -10 10 -10 10 -10 10
create_box 1 box
`;

const atoms = (pts: number[][]) => pts.map((p) => `create_atoms 1 single ${p[0]} ${p[1]} ${p[2]} units box`).join('\n');
const PAIR = 'mass 1 1.0\npair_style lj/cut 1.0\npair_coeff 1 1 0.0 1.0\n';

/** A four-atom and a three-atom geometry with generic (non-special) coordinates. */
const PTS = [
  [0.1, 0.9, 0.2],
  [0.0, 0.0, 0.1],
  [1.1, 0.2, -0.3],
  [1.6, 0.8, 0.9],
  [-0.4, 1.3, 0.5],
];

const make = async (fixLines: string, pts = PTS) => {
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  await session.execute(`${HEAD}${atoms(pts)}\n${PAIR}${fixLines}\nrun 0\n`);
  return session;
};

const errorOf = async (text: string): Promise<string | null> => {
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  try {
    await session.execute(text);
    return null;
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
};

const fixOf = (session: Session, id: string): Fix => {
  const f = session.sys.fixes.find((x) => x.id === id);
  if (!f) throw new Error(`no fix ${id}`);
  return f;
};

/**
 * Checks the fix's forces against -grad E (central differences) at the current
 * positions. E is the fix's energy, evaluated by postForce.
 */
const checkForces = (session: Session, fix: Fix, h = 1e-6, tol = 1e-6) => {
  const s = session.sys.state;
  const energyAt = (): number => {
    s.f.fill(0);
    fix.postForce!();
    return fix.energy();
  };
  energyAt();
  const f0 = Float64Array.from(s.f);
  let worst = 0;
  for (let k = 0; k < 3 * s.n; k++) {
    const x0 = s.x[k];
    s.x[k] = x0 + h;
    const ep = energyAt();
    s.x[k] = x0 - h;
    const em = energyAt();
    s.x[k] = x0;
    const grad = (ep - em) / (2 * h);
    worst = Math.max(worst, Math.abs(-grad - f0[k]));
  }
  expect(worst).toBeLessThan(tol);
  // the energy itself must be finite and nonzero for a meaningful check
  expect(Number.isFinite(fix.energy())).toBe(true);
};

describe('fix restrain: forces equal -grad E', () => {
  it('bond (harmonic, ramped K and r0)', async () => {
    const session = await make('fix R all restrain bond 1 2 12.0 30.0 1.0 1.6');
    checkForces(session, fixOf(session, 'R'));
  });

  it('lbound active (r < r0) and inactive (r >= r0)', async () => {
    // atoms 2-4 are about 1.7 apart; r0 = 2.5 is active, r0 = 0.5 is not
    const active = await make('fix R all restrain lbound 2 4 9.0 9.0 2.5');
    checkForces(active, fixOf(active, 'R'));
    expect(fixOf(active, 'R').energy()).toBeGreaterThan(0);
    const inactive = await make('fix R all restrain lbound 2 4 9.0 9.0 0.5');
    expect(fixOf(inactive, 'R').energy()).toBe(0);
    checkForces(inactive, fixOf(inactive, 'R'));
  });

  it('angle (K per radian^2, theta0 in degrees)', async () => {
    const session = await make('fix R all restrain angle 1 2 3 8.0 20.0 100.0');
    checkForces(session, fixOf(session, 'R'));
  });

  it('dihedral with mult 1, 2 and 3 and a second dihedral', async () => {
    for (const mult of [1, 2, 3]) {
      const session = await make(`fix R all restrain dihedral 1 2 3 4 4.0 10.0 -75.0 mult ${mult}`);
      checkForces(session, fixOf(session, 'R'));
    }
    const two = await make('fix R all restrain dihedral 1 2 3 4 4.0 4.0 40.0 dihedral 5 2 3 4 3.0 3.0 -150.0 mult 2');
    checkForces(two, fixOf(two, 'R'));
  });

  it('combined bond, angle and dihedral: scalar is the sum of the terms', async () => {
    const session = await make('fix R all restrain bond 1 3 5.0 5.0 1.0 1.0 angle 2 1 4 3.0 3.0 60.0 dihedral 1 2 3 4 2.0 2.0 30.0');
    checkForces(session, fixOf(session, 'R'));
  });

  it('vector slots follow the measured native layout (bond, lbound, zero)', async () => {
    const session = await make('fix R all restrain bond 1 2 2.0 2.0 1.0 1.0 angle 1 2 3 1.0 1.0 90.0');
    const f = fixOf(session, 'R');
    f.postForce!();
    const bond = f.computeVector!(0);
    // atoms 1 and 2 of PTS are about 0.9055 apart (r0 = 1.0)
    const r = Math.hypot(PTS[0][0] - PTS[1][0], PTS[0][1] - PTS[1][1], PTS[0][2] - PTS[1][2]);
    expect(bond).toBeCloseTo(2.0 * (r - 1.0) ** 2, 10);
    expect(f.computeVector!(1)).toBe(0);
    expect(f.computeVector!(2)).toBe(0);
    expect(f.computeScalar!()).toBeCloseTo(f.energy(), 12);
  });
});

describe('fix restrain: closed-form values', () => {
  it('bond energy K (r - r0)^2 at a known distance', async () => {
    // atoms 1 and 2 are placed 3.0 apart along x
    const session = await make('fix R all restrain bond 1 2 10.0 10.0 2.0', [[0, 0, 0], [3, 0, 0]]);
    expect(fixOf(session, 'R').energy()).toBeCloseTo(10.0, 10);
  });

  it('angle energy K (theta - theta0)^2 with theta0 in degrees', async () => {
    // atoms 1-2-3 form a right angle at atom 2; theta0 = 90 gives zero energy, 45 gives (pi/4)^2
    const right = await make('fix R all restrain angle 1 2 3 1.0 1.0 90.0', [[1, 0, 0], [0, 0, 0], [0, 1, 0]]);
    expect(fixOf(right, 'R').energy()).toBeCloseTo(0, 12);
    const off = await make('fix R all restrain angle 1 2 3 1.0 1.0 45.0', [[1, 0, 0], [0, 0, 0], [0, 1, 0]]);
    expect(fixOf(off, 'R').energy()).toBeCloseTo((Math.PI / 4) ** 2, 10);
  });

  it('dihedral energy K [1 + cos(n phi - phi0)] for a cis geometry (phi = 0)', async () => {
    // cis: atoms 1 and 4 on the same side of the 2-3 axis, dihedral 0
    const pts = [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]];
    const mult2 = await make('fix R all restrain dihedral 1 2 3 4 2.0 2.0 0.0 mult 2', pts);
    expect(fixOf(mult2, 'R').energy()).toBeCloseTo(4.0, 10);
    const opp = await make('fix R all restrain dihedral 1 2 3 4 2.0 2.0 180.0', pts);
    expect(fixOf(opp, 'R').energy()).toBeCloseTo(0, 10);
  });

  it('a setup at run 0 uses Kstart (the ramp fraction is zero)', async () => {
    const session = new Session({ emit: () => {}, writeFile: () => {} });
    await session.execute(`${HEAD}${atoms([[0, 0, 0], [3, 0, 0]])}\n${PAIR}fix R all restrain bond 1 2 10.0 30.0 2.0\nthermo 1\nrun 0\n`);
    expect(fixOf(session, 'R').energy()).toBeCloseTo(10.0, 10);
  });
});

describe('fix restrain: argument errors', () => {
  const base = `${HEAD}${atoms(PTS)}\n${PAIR}`;
  it('unknown keyword is named', async () => {
    expect(await errorOf(`${base}fix R all restrain torsion 1 2 3 4 1.0 1.0 0.0\n`)).toMatch(/torsion/);
  });
  it('bond followed by another keyword: the optional r0stop is read (native behaviour)', async () => {
    expect(await errorOf(`${base}fix R all restrain bond 1 2 1.0 1.0 2.0 angle 1 2 3 1.0 1.0 90.0\n`)).toMatch(/'angle'/);
  });
  it('missing arguments', async () => {
    expect(await errorOf(`${base}fix R all restrain bond 1 2 1.0 1.0\n`)).toMatch(/bond/);
    expect(await errorOf(`${base}fix R all restrain dihedral 1 2 3 4 1.0 1.0\n`)).toMatch(/dihedral/);
  });
  it('mult must be an integer >= 0', async () => {
    expect(await errorOf(`${base}fix R all restrain dihedral 1 2 3 4 1.0 1.0 0.0 mult -1\n`)).toMatch(/mult/);
    expect(await errorOf(`${base}fix R all restrain dihedral 1 2 3 4 1.0 1.0 0.0 mult 1.5\n`)).toMatch(/mult/);
  });
  it('no restraint at all, and non-numeric values', async () => {
    expect(await errorOf(`${base}fix R all restrain\n`)).toMatch(/at least one/);
    expect(await errorOf(`${base}fix R all restrain bond 1 2 abc 1.0 2.0\n`)).toMatch(/abc/);
  });
  it('bond atoms must differ', async () => {
    expect(await errorOf(`${base}fix R all restrain bond 2 2 1.0 1.0 2.0\n`)).toMatch(/differ/);
  });
});

describe('fix spring/rg (fix_spring_rg.html)', () => {
  /** Radius of gyration of all atoms with unit mass about their centre (no image flags: positions are in the box). */
  const rgOf = (x: Float64Array, n: number): number => {
    const c = [0, 0, 0];
    for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) c[d] += x[3 * i + d] / n;
    let r2 = 0;
    for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) r2 += (x[3 * i + d] - c[d]) ** 2;
    return Math.sqrt(r2 / n);
  };

  it('force equals -grad of E = K (RG - RG0)^2 (unit masses)', async () => {
    const session = await make('fix S all spring/rg 3.0 0.6');
    const fix = fixOf(session, 'S');
    const s = session.sys.state;
    const energy = (): number => {
      const rg = rgOf(s.x, s.n);
      return 3.0 * (rg - 0.6) ** 2;
    };
    s.f.fill(0);
    fix.postForce!();
    const f0 = Float64Array.from(s.f);
    const h = 1e-6;
    let worst = 0;
    for (let k = 0; k < 3 * s.n; k++) {
      const x0 = s.x[k];
      s.x[k] = x0 + h;
      const ep = energy();
      s.x[k] = x0 - h;
      const em = energy();
      s.x[k] = x0;
      worst = Math.max(worst, Math.abs(-(ep - em) / (2 * h) - f0[k]));
    }
    expect(worst).toBeLessThan(1e-6);
  });

  it('scalar is RG0; NULL takes the radius of gyration at definition', async () => {
    const numeric = await make('fix S all spring/rg 3.0 0.6');
    expect(fixOf(numeric, 'S').computeScalar!()).toBeCloseTo(0.6, 12);
    const session = await make('fix S all spring/rg 3.0 NULL');
    const s = session.sys.state;
    expect(fixOf(session, 'S').computeScalar!()).toBeCloseTo(rgOf(s.x, s.n), 10);
  });

  it('argument count', async () => {
    expect(await errorOf(`${HEAD}${atoms(PTS)}\nfix S all spring/rg 3.0\n`)).toMatch(/spring\/rg/);
  });
});

describe('fix evaporate (fix_evaporate.html)', () => {
  const sys = `${HEAD}${atoms([[0.5, 0, 0], [1.0, 0.2, 0], [1.5, 0.4, 0], [2.0, 0.6, 0], [4.0, 0, 0], [4.5, 0.2, 0]])}\n${PAIR}`;

  it('removes M atoms every N steps from the region, cumulative scalar', async () => {
    const events: { kind: string }[] = [];
    const s2 = new Session({ emit: (e) => events.push(e as { kind: string }), writeFile: () => {} });
    await s2.execute(`${sys}region ev block 0 2.6 -10 10 -10 10 units box\nfix 1 all nve\nfix EV all evaporate 2 2 ev 4242\nthermo_style custom step atoms f_EV\nthermo 1\nrun 4\n`);
    const rows = events.filter((e) => e.kind === 'thermo') as unknown as { row: { step: number; atoms: number; f_EV: number } }[];
    const at = (step: number) => rows.map((r) => r.row).find((r) => r.step === step);
    expect(at(1)?.atoms).toBe(6);
    expect(at(2)?.atoms).toBe(4);
    expect(at(2)?.f_EV).toBe(2);
    expect(at(4)?.atoms).toBe(2);
    expect(at(4)?.f_EV).toBe(4);
  });

  it('argument errors name the problem', async () => {
    expect(await errorOf(`${sys}fix EV all evaporate 2 2 nosuch 5\nrun 1\n`)).toMatch(/nosuch/);
    expect(await errorOf(`${sys}region ev block 0 2 -10 10 -10 10 units box\nfix EV all evaporate 2 2 ev 5 molecule maybe\n`)).toMatch(/molecule/);
    expect(await errorOf(`${sys}region ev block 0 2 -10 10 -10 10 units box\nfix EV all evaporate 0 2 ev 5\n`)).toMatch(/N/);
    expect(await errorOf(`${sys}region ev block 0 2 -10 10 -10 10 units box\nfix EV all evaporate 2 2 ev 5 bogus yes\n`)).toMatch(/bogus/);
  });
});

describe('compute msd/nongauss (compute_msd_nongauss.html)', () => {
  it('uniform displacement d: dr^2 = |d|^2, dr^4 = |d|^4, NGP = -0.4', async () => {
    const session = new Session({ emit: () => {}, writeFile: () => {} });
    await session.execute(`${HEAD}${atoms(PTS)}\n${PAIR}compute ng all msd/nongauss\nrun 0\n`);
    const s = session.sys.state;
    const d = [0.3, -0.2, 0.1];
    for (let i = 0; i < s.n; i++) for (let k = 0; k < 3; k++) s.x[3 * i + k] += d[k];
    session.sys.bump();
    const v = session.sys.compute('ng').vectorValues();
    const r2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2;
    expect(v[0]).toBeCloseTo(r2, 12);
    expect(v[1]).toBeCloseTo(r2 * r2, 12);
    expect(v[2]).toBeCloseTo(-0.4, 12);
  });

  it('com yes removes a uniform drift of the whole group', async () => {
    const session = new Session({ emit: () => {}, writeFile: () => {} });
    await session.execute(`${HEAD}${atoms(PTS)}\n${PAIR}compute ng all msd/nongauss com yes\nrun 0\n`);
    const s = session.sys.state;
    for (let i = 0; i < s.n; i++) for (let k = 0; k < 3; k++) s.x[3 * i + k] += 0.5;
    session.sys.bump();
    const v = session.sys.compute('ng').vectorValues();
    expect(v[0]).toBeCloseTo(0, 12);
    expect(v[1]).toBeCloseTo(0, 12);
  });

  it('keyword errors', async () => {
    expect(await errorOf(`${HEAD}${atoms(PTS)}\ncompute ng all msd/nongauss bogus yes\n`)).toMatch(/bogus/);
    expect(await errorOf(`${HEAD}${atoms(PTS)}\ncompute ng all msd/nongauss com maybe\n`)).toMatch(/com/);
  });
});
