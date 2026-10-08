import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix box/relax (src/engine/fix/box_relax.ts, docs.lammps.org/fix_box_relax.html):
 * argument errors, and a box under hydrostatic tension relaxing to zero pressure.
 */

const SETUP = `units lj
atom_style atomic
lattice fcc 1.0
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
pair_modify shift yes
`;

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  await session.execute(text);
  return events;
};

const fails = async (text: string): Promise<string> => {
  try {
    await run(text);
  } catch (e) {
    return (e as Error).message;
  }
  throw new Error('expected an engine error');
};

describe('fix box/relax argument errors', () => {
  it('rejects an unknown keyword by name', async () => {
    expect(await fails(SETUP + 'fix 1 all box/relax iso 0.0 bogus 3\n')).toMatch(/bogus/);
  });
  it('rejects a keyword without its value', async () => {
    expect(await fails(SETUP + 'fix 1 all box/relax iso\n')).toMatch(/iso.*value/);
  });
  it('rejects a non-positive vmax', async () => {
    expect(await fails(SETUP + 'fix 1 all box/relax iso 0.0 vmax 0.0\n')).toMatch(/vmax must be greater than 0/);
  });
  it('rejects couple with unequal targets', async () => {
    expect(await fails(SETUP + 'fix 1 all box/relax x 1.0 y 2.0 z 1.0 couple xyz\n')).toMatch(/identical Ptarget/);
  });
  it('rejects a box fix with no pressure keyword', async () => {
    expect(await fails(SETUP + 'fix 1 all box/relax couple xyz\n')).toMatch(/no pressure keyword/);
  });
  it('rejects dilate partial (not implemented)', async () => {
    expect(await fails(SETUP + 'fix 1 all box/relax iso 0.0 dilate partial\n')).toMatch(/dilate partial is not supported/);
  });
  it('rejects tilt keywords on an orthogonal box', async () => {
    expect(await fails(SETUP + 'fix 1 all box/relax tri 0.0\nminimize 0 1e-8 10 100\n')).toMatch(/triclinic/);
  });
  it('rejects a fire minimizer with the box fix', async () => {
    expect(await fails(SETUP + 'fix 1 all box/relax iso 0.0\nmin_style fire\nminimize 0 1e-8 10 100\n')).toMatch(/fire.*box\/relax/);
  });
  it('rejects an unknown couple value', async () => {
    expect(await fails(SETUP + 'fix 1 all box/relax x 1.0 couple abc\n')).toMatch(/couple must be/);
  });
});

describe('fix box/relax relaxation', () => {
  const thermoRows = (events: EngineEvent[]): ThermoRow[] =>
    events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);

  it('relaxes a box under hydrostatic tension to zero pressure', async () => {
    // the ideal fcc lattice at density 1.0 is under tension (press about -3.4 at the start)
    const events = await run(SETUP + 'thermo_style custom step pe press vol f_1\nfix 1 all box/relax iso 0.0\nminimize 0 1e-10 20000 200000\n');
    const rows = thermoRows(events);
    const first = rows[0], last = rows[rows.length - 1];
    expect(first.press).toBeLessThan(-3);
    expect(Math.abs(last.press)).toBeLessThan(1e-6);
    // tension (press < 0) pulls the walls in: the box shrinks
    expect(last.vol).toBeLessThan(first.vol);
    expect(Math.abs(last.f_1)).toBeLessThan(1e-9);
  }, 120_000);

  it('reports the box energy as f_ID per atom, zero at the reference box', async () => {
    const events = await run(SETUP + 'thermo_style custom step pe f_1\nfix 1 all box/relax x 0.5 y 0.5 z 0.5\nthermo 0\nrun 0\n');
    expect(thermoRows(events)[0].f_1).toBe(0);
  });

  it('box force equals the finite-difference derivative of the box objective (aniso and tilt DOF)', async () => {
    const displaced = SETUP.replace('pair_modify shift yes', 'pair_modify shift yes\nvariable dx atom 0.07*sin(3.1*x+1.7*y)\nvariable dy atom 0.07*cos(2.3*y+0.9*z)\nvariable dz atom 0.07*sin(1.9*z+2.5*x)\ndisplace_atoms all move v_dx v_dy v_dz units box');
    const session = new Session({ emit: () => {}, writeFile: () => {} });
    await session.execute(displaced + 'change_box all triclinic\nfix 1 all box/relax x 1.0 y 0.5 z -0.3 xy 0.7 xz -0.2 yz 0.4\nminimize 0 0 0 0\n');
    const sys = session.sys as any;
    const st = sys.state;
    const n3 = 3 * st.n;
    const bd = (sys.fixes.find((f: any) => f.style === 'box/relax') as any).boxDof;
    const M: number = bd.dofCount;
    expect(M).toBe(6);
    const x0 = Float64Array.from(st.x.subarray(0, n3));
    // start from a strained box so that the strain and shear terms are nonzero
    bd.trial(Float64Array.from([0.02, -0.01, 0.015, 0.03, -0.02, 0.01]), x0, st.x);
    bd.begin();
    const u0 = new Float64Array(M);
    bd.dof(u0);
    const total = (u: Float64Array) => {
      bd.trial(u, x0, st.x);
      sys.bump();
      const a = sys.forces();
      return a.evdwl + a.ecoul + a.ebond + a.eangle + a.edihed + a.eimp + bd.evaluate(new Float64Array(M));
    };
    const force = new Float64Array(M);
    bd.trial(u0, x0, st.x); sys.bump(); sys.forces(); bd.evaluate(force);
    for (let k = 0; k < M; k++) {
      const h = 1e-6;
      const up = Float64Array.from(u0); up[k] += h;
      const um = Float64Array.from(u0); um[k] -= h;
      const fd = -(total(up) - total(um)) / (2 * h);
      expect(Math.abs(force[k] - fd), `dof ${k}`).toBeLessThan(1e-5 * Math.max(1, Math.abs(fd)));
    }
  });
});
