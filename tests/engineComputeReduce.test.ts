import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { ComputeDisplaceAtom } from '../src/engine/compute/reduce';

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
create_box 2 box
create_atoms 1 box
mass 1 1.0
mass 2 1.5
group two id 2
displace_atoms two move 0.3 0.0 0.0 units box
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
pair_coeff 2 2 1.0 1.0
`;

describe('compute reduce', () => {
  it('applies the modes to atom attributes and an atom-style variable', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      variable k atom 0.5*mass*(vx*vx+vy*vy+vz*vz)
      compute r1 all reduce sum vx
      compute r2 all reduce ave vx
      compute r3 all reduce min vx vy
      compute r4 all reduce maxabs vy
      compute r5 all reduce sumsq vx
      compute r6 all reduce aveabs vz
      compute r7 all reduce sum v_k
      run 0
    `);
    const sys = s.sys;
    const r1 = sys.compute('r1');
    expect(r1.scalarFlag).toBe(true);
    expect(r1.vectorFlag).toBe(false);
    expect(r1.extscalar).toBe(1); // sum is extensive
    expect(r1.scalarValue()).toBeCloseTo(0.6, 12);
    expect(sys.compute('r2').extscalar).toBe(0); // ave is intensive
    expect(sys.compute('r2').scalarValue()).toBeCloseTo(0.3, 12);
    const r3 = sys.compute('r3');
    expect(r3.vectorFlag).toBe(true);
    expect(r3.sizeVector).toBe(2);
    expect(r3.vectorValues()[0]).toBeCloseTo(0.3, 12);
    expect(r3.vectorValues()[1]).toBeCloseTo(-0.2, 12);
    expect(sys.compute('r4').scalarValue()).toBeCloseTo(0.2, 12);
    expect(sys.compute('r5').scalarValue()).toBeCloseTo(2 * 0.09, 12);
    expect(sys.compute('r6').scalarValue()).toBeCloseTo(0.5, 12);
    expect(sys.compute('r7').scalarValue()).toBeCloseTo(2 * 0.5 * (0.09 + 0.04 + 0.25), 12);
  });

  it('restricts the reduction to the compute group', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      velocity two set 0.9 0.0 -0.4 units box
      compute rg two reduce sum vx
      run 0
    `);
    expect(s.sys.compute('rg').scalarValue()).toBeCloseTo(0.9, 12);
  });

  it('reduces per-atom values of computes, including wildcard column lists', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      compute k all ke/atom
      compute p all property/atom x y z
      compute f all reduce sum c_k
      compute w all reduce max c_p[*]
      run 0
    `);
    const sys = s.sys;
    expect(sys.compute('f').scalarValue()).toBeCloseTo(2 * 0.5 * (0.09 + 0.04 + 0.25), 12);
    const w = sys.compute('w');
    expect(w.sizeVector).toBe(3); // c_p[*] expanded to the three columns of the per-atom array
    const vals = w.vectorValues();
    const st = sys.state;
    for (let d = 0; d < 3; d++) {
      let mx = -Infinity;
      for (let i = 0; i < st.n; i++) if (st.x[3 * i + d] > mx) mx = st.x[3 * i + d];
      expect(vals[d]).toBeCloseTo(mx, 12);
    }
  });

  it('reduces force components after the run made them current', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      compute fmin all reduce min fx
      compute fmax all reduce max fx
      run 0
    `);
    const sys = s.sys;
    const st = sys.state;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < st.n; i++) {
      if (st.f[3 * i] < mn) mn = st.f[3 * i];
      if (st.f[3 * i] > mx) mx = st.f[3 * i];
    }
    expect(mx).not.toBe(0); // the pair at r = 0.7 exerts a force
    expect(sys.compute('fmin').scalarValue()).toBeCloseTo(mn, 12);
    expect(sys.compute('fmax').scalarValue()).toBeCloseTo(mx, 12);
  });

  it('replace selects from vec1 at the index of the max/min of vec2', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      velocity two set 0.9 0.0 -0.4 units box
      compute r all reduce max vx vz replace 1 2
      run 0
    `);
    const vals = s.sys.compute('r').vectorValues();
    // max vz is atom 1 (0.5); vec1 is selected there, instead of max vx (0.9)
    expect(close(vals[0], 0.3)).toBe(true);
    expect(close(vals[1], 0.5)).toBe(true);
  });

  it('reduce/region only counts atoms currently inside the region', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      velocity two set 0.9 0.0 -0.4 units box
      region right block 0.5 INF INF INF INF INF units box
      compute rr all reduce/region right sum vx
      run 0
    `);
    expect(s.sys.compute('rr').scalarValue()).toBeCloseTo(0.9, 12); // only atom 2 (x = 1.3) is in the region
  });

  it('rejects bad modes, keywords and indices', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute r all reduce median vx`)).rejects.toThrow("unknown reduce mode 'median'");
    await expect(runScript(`${TWO_ATOMS}\ncompute r all reduce sum`)).rejects.toThrow('at least one input value is required');
    await expect(runScript(`${TWO_ATOMS}\ncompute r all reduce sum bogus`)).rejects.toThrow("invalid input value 'bogus'");
    await expect(runScript(`${TWO_ATOMS}\ncompute r all reduce sum vx inputs local`)).rejects.toThrow('inputs local is not supported');
    await expect(runScript(`${TWO_ATOMS}\ncompute r all reduce sum vx inputs sometimes`)).rejects.toThrow('peratom or local');
    await expect(runScript(`${TWO_ATOMS}\ncompute r all reduce sum vx replace 1 1`)).rejects.toThrow('replace keyword can only be used if the mode is min or max');
    await expect(runScript(`${TWO_ATOMS}\ncompute r all reduce min vx replace 1 2`)).rejects.toThrow('replace index out of range 1..1');
    await expect(runScript(`${TWO_ATOMS}\ncompute r all reduce min vx replace x 1`)).rejects.toThrow('positive integer');
  });

  it('rejects inputs that are not per-atom producers', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute t all temp\ncompute r all reduce sum c_t`)).rejects.toThrow('compute t does not calculate per-atom values');
    await expect(runScript(`${TWO_ATOMS}\ncompute p all property/atom x y z\ncompute r all reduce sum c_p`)).rejects.toThrow('give a column');
    await expect(runScript(`${TWO_ATOMS}\ncompute p all property/atom x y z\ncompute r all reduce sum c_p[4]`)).rejects.toThrow('column out of range 1..3');
  });

  it('reduce/region rejects a missing region', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute r all reduce/region nope sum vx`)).rejects.toThrow("region ID 'nope' does not exist");
  });
});

describe('compute displace/atom', () => {
  it('measures the displacement from the definition-time unwrapped position, across PBC', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      compute d two displace/atom
      displace_atoms two move 1.8 0.4 -0.3 units box
      run 0
    `);
    const d = s.sys.compute('d');
    expect(d.peratomFlag).toBe(true);
    expect(d.sizePeratomCols).toBe(4);
    const vals = d.peratomValues();
    // atom 1 is not in group two: 0.0
    expect(vals[0]).toBeCloseTo(0, 12);
    expect(vals[1]).toBeCloseTo(0, 12);
    // atom 2: x = 1.3 + 1.8 = 3.1 wraps to 1.1 (image 1), z = -0.3 wraps to 0.7 (image -1);
    // the unwrapped displacement is still (1.8, 0.4, -0.3)
    expect(vals[4]).toBeCloseTo(1.8, 12);
    expect(vals[5]).toBeCloseTo(0.4, 12);
    expect(vals[6]).toBeCloseTo(-0.3, 12);
    expect(vals[7]).toBeCloseTo(Math.sqrt(1.8 * 1.8 + 0.4 * 0.4 + 0.3 * 0.3), 12);
  });

  it('refresh re-references atoms whose atom-style variable is true', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      variable flag atom "x > 0.5"
      compute d all displace/atom refresh flag
      displace_atoms two move 1.8 0.0 0.0 units box
      run 0
    `);
    const d = s.sys.compute('d') as ComputeDisplaceAtom;
    const before = d.peratomValues();
    expect(before[4]).toBeCloseTo(1.8, 12); // atom 2 moved 1.8 (wrapped to x = 1.1)
    d.refresh(); // variable flag is true for atom 2 (x = 1.1 > 0.5), false for atom 1
    const after = d.peratomValues();
    expect(after[4]).toBeCloseTo(0, 12);
    expect(after[7]).toBeCloseTo(0, 12);
    expect(after[0]).toBeCloseTo(0, 12);
  });

  it('rejects unknown keywords and non-atom-style refresh variables', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute d all displace/atom bogus`)).rejects.toThrow("unknown compute displace/atom keyword 'bogus'");
    await expect(runScript(`${TWO_ATOMS}\nvariable f equal 1\ncompute d all displace/atom refresh f\nrun 0`)).rejects.toThrow('must be atom-style');
  });
});

describe('compute coord/atom', () => {
  it('counts neighbors by type within the cutoff (pair across the periodic boundary)', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      set group two type 2
      compute c all coord/atom cutoff 0.8 1 2
      run 0
    `);
    const c = s.sys.compute('c');
    expect(c.peratomFlag).toBe(true);
    expect(c.sizePeratomCols).toBe(2);
    const vals = c.peratomValues();
    // the atoms sit 0.7 apart through the periodic boundary (x = 0 and x = 1.3 in a box of lx = 2)
    expect(vals[0]).toBe(0); // atom 1: no type-1 neighbors
    expect(vals[1]).toBe(1); // atom 1: one type-2 neighbor
    expect(vals[2]).toBe(1); // atom 2: one type-1 neighbor
    expect(vals[3]).toBe(0); // atom 2: no type-2 neighbors
  });

  it('produces zeros below the cutoff and a per-atom vector without typeN', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      set group two type 2
      compute small all coord/atom cutoff 0.5
      compute wide all coord/atom cutoff 0.8
      run 0
    `);
    const sys = s.sys;
    const small = sys.compute('small');
    expect(small.sizePeratomCols).toBe(0);
    expect(small.peratomValues()[0]).toBe(0);
    expect(small.peratomValues()[1]).toBe(0);
    const wide = sys.compute('wide');
    expect(wide.sizePeratomCols).toBe(0); // single (implicit all-types) count -> per-atom vector
    expect(wide.peratomValues()[0]).toBe(1);
    expect(wide.peratomValues()[1]).toBe(1);
  });

  it('restricts neighbors with the group keyword and type ranges', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      set group two type 2
      compute g all coord/atom cutoff 0.8 group two
      compute t all coord/atom cutoff 0.8 2*2
      run 0
    `);
    const sys = s.sys;
    const g = sys.compute('g').peratomValues();
    // neighbors must be in group "two" (atom 2): atom 1 counts it, atom 2 has no other group members
    expect(g[0]).toBe(1);
    expect(g[1]).toBe(0);
    const t = sys.compute('t').peratomValues();
    expect(t[0]).toBe(1); // atom 2 has type 2
    expect(t[1]).toBe(0);
  });

  it('rejects unsupported cstyles, bad cutoffs and bad types', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute c all coord/atom orientorder 2 0.5`)).rejects.toThrow("'orientorder' is not supported");
    await expect(runScript(`${TWO_ATOMS}\ncompute c all coord/atom weird 1.0`)).rejects.toThrow("unknown compute coord/atom cstyle 'weird'");
    await expect(runScript(`${TWO_ATOMS}\ncompute c all coord/atom cutoff`)).rejects.toThrow('needs a cutoff distance');
    await expect(runScript(`${TWO_ATOMS}\ncompute c all coord/atom cutoff -1.0`)).rejects.toThrow('must be a positive number');
    await expect(runScript(`${TWO_ATOMS}\ncompute c all coord/atom cutoff 0.8 3`)).rejects.toThrow("type '3' is outside 1..2");
    await expect(runScript(`${TWO_ATOMS}\ncompute c all coord/atom cutoff 0.8 2*1`)).rejects.toThrow("type range '2*1' is outside 1..2");
    await expect(runScript(`${TWO_ATOMS}\ncompute c all coord/atom cutoff 0.8 1 group two`)).rejects.toThrow('group keyword must come before the typeN values');
  });
});
