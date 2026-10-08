import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/** Relative closeness for the checks against manual formulas. */
const close = (got: number, want: number, rel = 1e-12) => Math.abs(got - want) <= rel * Math.max(1, Math.abs(got), Math.abs(want));

/** Runs input text in a fresh session. */
const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  await session.execute(text);
  return session;
};

const TWO_ATOMS = `
units lj
atom_style atomic
boundary p p p
lattice sc 1.0
region box block 0 2 0 1 0 1
create_box 1 box
create_atoms 1 box
mass 1 1.0
group two id 2
displace_atoms two move 0.3 0.0 0.0 units box
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
`;

describe('compute ke/atom', () => {
  it('computes 1/2 m v^2 per atom and zeroes atoms outside the group', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      group half id 2
      compute k half ke/atom
      run 0
    `);
    const k = s.sys.compute('k');
    expect(k.peratomFlag).toBe(true);
    expect(k.sizePeratomCols).toBe(0);
    const vals = k.peratomValues();
    const ke = 0.5 * 1.0 * (0.3 * 0.3 + (-0.2) * (-0.2) + 0.5 * 0.5);
    expect(vals[0]).toBeCloseTo(0, 12); // atom 1 is not in group "half"
    expect(vals[1]).toBeCloseTo(ke, 12);
  });

  it('rejects arguments', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute k all ke/atom extra`)).rejects.toThrow('ke/atom takes no arguments');
  });
});

describe('compute pe/atom', () => {
  it('splits the pair energy in equal halves and matches compute pe', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      compute p all pe/atom
      compute pp all pe/atom pair
      compute pe all pe
      run 0
    `);
    const p = s.sys.compute('p');
    const pp = s.sys.compute('pp');
    const pe = s.sys.compute('pe');
    expect(p.sizePeratomCols).toBe(0);
    const vals = p.peratomValues();
    const total = pe.scalarValue();
    expect(vals[0]).toBeCloseTo(0.5 * total, 12);
    expect(vals[1]).toBeCloseTo(0.5 * total, 12);
    expect(vals[0] + vals[1]).toBeCloseTo(total, 12);
    const pairVals = pp.peratomValues();
    expect(pairVals[0]).toBeCloseTo(vals[0], 12);
    expect(pairVals[1]).toBeCloseTo(vals[1], 12);
    expect(total).not.toBe(0); // r = 0.7 sigma: the LJ energy is nonzero
  });

  it('rejects an unknown keyword', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute p all pe/atom torsion`)).rejects.toThrow("unknown compute pe/atom keyword 'torsion'");
  });
});

describe('compute stress/atom', () => {
  it('applies S_ab = -m v_a v_b - W_ab with the per-atom virial split in halves', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      fix 1 all nve
      compute s all stress/atom NULL
      run 0
    `);
    const sys = s.sys;
    const st = sys.compute('s');
    expect(st.peratomFlag).toBe(true);
    expect(st.sizePeratomCols).toBe(6);
    const vals = st.peratomValues();
    const a = sys.forces();
    const m = 1.0;
    // per-atom pair virial of the single (1,2) pair: half of the global virial
    for (let i = 0; i < 2; i++) {
      expect(close(vals[6 * i], -m * 0.3 * 0.3 - 0.5 * a.virial[0])).toBe(true);
      expect(close(vals[6 * i + 1], -m * (-0.2) * (-0.2) - 0.5 * a.virial[1])).toBe(true);
      expect(close(vals[6 * i + 2], -m * 0.5 * 0.5 - 0.5 * a.virial[2])).toBe(true);
      expect(close(vals[6 * i + 3], -m * 0.3 * (-0.2) - 0.5 * a.virial[3])).toBe(true);
    }
  });

  it('summed diagonal / (d V) equals minus the pressure', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      fix 1 all nve
      compute s all stress/atom NULL
      compute press all pressure thermo_temp
      run 0
    `);
    const sys = s.sys;
    const vals = sys.compute('s').peratomValues();
    const press = sys.compute('press').scalarValue();
    const vol = sys.geom.volume(sys.state.dimension);
    let sum = 0;
    for (let i = 0; i < sys.state.n; i++) sum += vals[6 * i] + vals[6 * i + 1] + vals[6 * i + 2];
    expect(-sum / (3 * vol)).toBeCloseTo(press, 10);
  });

  it('rejects a temp-ID that is not a temperature compute', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute pe all pe\ncompute s all stress/atom pe\nrun 0`)).rejects.toThrow('does not compute a temperature');
  });
});

describe('compute property/atom', () => {
  it('stores unwrapped coordinates, images and mass across a periodic boundary', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      displace_atoms two move 1.8 0.0 0.0 units box
      run 0
      compute q all property/atom xu yu zu ix iy iz fx mass
      run 0
    `);
    const sys = s.sys;
    const q = sys.compute('q');
    expect(q.peratomFlag).toBe(true);
    expect(q.sizePeratomCols).toBe(8);
    const vals = q.peratomValues();
    const st = sys.state;
    const g = sys.geom;
    // atom 2 sits at x = 3.1 in a box of lx = 2: wrapped to 1.1, image ix = 1
    expect(st.image[3 * 1]).toBe(1);
    expect(st.x[3 * 1]).toBeCloseTo(1.1, 12);
    expect(vals[8 * 1]).toBeCloseTo(st.x[3 * 1] + st.image[3 * 1] * g.lx, 12); // xu
    expect(vals[8 * 1 + 2]).toBeCloseTo(0, 12); // zu
    expect(vals[8 * 1 + 3]).toBe(1); // ix
    expect(vals[8 * 1 + 4]).toBe(0); // iy
    expect(vals[8 * 1 + 6]).toBeCloseTo(st.f[3 * 1]); // fx matches the state force
    expect(st.f[3 * 1]).not.toBe(0); // the pair at r = 0.7 exerts a force
    expect(vals[8 * 1 + 7]).toBe(1.0); // mass
  });

  it('produces a per-atom vector for a single input', async () => {
    const s = await runScript(`${TWO_ATOMS}\ncompute q all property/atom type`);
    const q = s.sys.compute('q');
    expect(q.sizePeratomCols).toBe(0);
    expect(q.peratomValues()[0]).toBe(1);
  });

  it('rejects an unsupported attribute instead of storing zeroes', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute q all property/atom end1x`)).rejects.toThrow("attribute 'end1x' is not supported");
    await expect(runScript(`${TWO_ATOMS}\ncompute q all property/atom nbonds`)).rejects.toThrow("attribute 'nbonds' is not supported");
  });
});
