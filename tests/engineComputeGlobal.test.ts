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

/** Two atoms at x = 0 and 1, two types so masses can differ. */
const TWO_ATOMS = `
units lj
atom_style atomic
boundary p p p
lattice sc 1.0
region box block 0 2 0 1 0 1
create_box 2 box
create_atoms 1 box
mass 1 1.0
mass 2 3.0
group two id 2
`;

describe('compute com', () => {
  it('mass-weighted mean of unwrapped positions', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      set atom 2 type 2
      displace_atoms all move 0.25 0.0 0.0 units box
      compute c all com
      run 0
    `);
    const c = s.sys.compute('c');
    expect(c.vectorFlag).toBe(true);
    expect(c.sizeVector).toBe(3);
    const v = c.vectorValues();
    // atoms at x = 0.25 and 1.25, masses 1 and 3: com x = (0.25 + 3*1.25)/4
    expect(close(v[0], (0.25 + 3 * 1.25) / 4)).toBe(true);
    expect(v[1]).toBe(0);
    expect(v[2]).toBe(0);
  });

  it('unwraps atoms that crossed a periodic boundary', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      displace_atoms all move 1.5 0.0 0.0 units box
      compute c all com
      run 0
    `);
    const v = s.sys.compute('c').vectorValues();
    // equal masses at x = 1.5 and 2.5 (the second wrapped to 0.5 with image 1)
    expect(close(v[0], 2.0)).toBe(true);
  });

  it('rejects arguments', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute c all com extra`)).rejects.toThrow('compute com takes no arguments');
  });
});

describe('compute gyration', () => {
  it('Rg and tensor of two atoms about their com', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      set atom 2 type 2
      displace_atoms all move 0.25 0.0 0.0 units box
      compute g all gyration
      run 0
    `);
    const g = s.sys.compute('g');
    expect(g.scalarFlag).toBe(true);
    expect(g.vectorFlag).toBe(true);
    expect(g.sizeVector).toBe(6);
    const t = g.vectorValues();
    const cm = (0.25 + 3 * 1.25) / 4;
    const d1 = 0.25 - cm, d2 = 1.25 - cm;
    const m1 = 1.0, m2 = 3.0, msum = m1 + m2;
    const rgyr2 = (m1 * d1 * d1 + m2 * d2 * d2) / msum;
    expect(close(t[0], rgyr2)).toBe(true);
    expect(t[1]).toBe(0);
    expect(t[2]).toBe(0);
    for (let c = 3; c < 6; c++) expect(t[c]).toBe(0);
    expect(close(g.scalarValue(), Math.sqrt(rgyr2))).toBe(true);
  });

  it('rejects arguments', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute g all gyration 3`)).rejects.toThrow('compute gyration takes no arguments');
  });
});

describe('compute msd', () => {
  it('mean squared displacement from the reference positions', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      compute m all msd
      thermo_style custom step c_m[4]
      displace_atoms two move 0.3 -0.2 0.5 units box
      run 0
    `);
    const m = s.sys.compute('m');
    expect(m.vectorFlag).toBe(true);
    expect(m.sizeVector).toBe(4);
    const v = m.vectorValues();
    // the compute is defined before the displace_atoms, so the reference is the lattice position
    expect(close(v[0], 0.3 * 0.3 / 2)).toBe(true);
    expect(close(v[1], 0.2 * 0.2 / 2)).toBe(true);
    expect(close(v[2], 0.5 * 0.5 / 2)).toBe(true);
    expect(close(v[3], (0.3 * 0.3 + 0.2 * 0.2 + 0.5 * 0.5) / 2)).toBe(true);
  });

  it('com yes subtracts a common drift of the group', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      compute m all msd com yes
      thermo_style custom step c_m[4]
      displace_atoms all move 0.4 0.3 -0.6 units box
      run 0
    `);
    const v = s.sys.compute('m').vectorValues();
    // both atoms moved by the same vector: pure center-of-mass drift, msd = 0
    for (let c = 0; c < 4; c++) expect(Math.abs(v[c])).toBeLessThan(1e-12);
  });

  it('average yes uses the running mean of the called positions', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      compute m all msd average yes
      thermo_style custom step c_m[4]
      run 0
      displace_atoms two move 0.2 0.0 0.0 units box
      run 0
    `);
    const v = s.sys.compute('m').vectorValues();
    // first call: reference = current position, msd = 0; second call: reference =
    // mean of (1.0, 1.2) for atom 2 -> displacement 0.1, atom 1 unmoved
    expect(close(v[3], (0.1 * 0.1) / 2)).toBe(true);
  });

  it('rejects unknown keywords and bad values', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute m all msd drift yes`)).rejects.toThrow("unknown compute msd keyword 'drift'");
    await expect(runScript(`${TWO_ATOMS}\ncompute m all msd com maybe`)).rejects.toThrow('compute msd com must be yes or no');
  });
});

describe('compute vacf', () => {
  it('current velocity dotted into the velocity at definition time', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      compute v all vacf
      velocity all set -0.1 0.4 0.25 units box
      run 0
    `);
    const v = s.sys.compute('v');
    expect(v.vectorFlag).toBe(true);
    expect(v.sizeVector).toBe(4);
    const vals = v.vectorValues();
    expect(close(vals[0], -0.3 * 0.1)).toBe(true);
    expect(close(vals[1], -0.2 * 0.4)).toBe(true);
    expect(close(vals[2], 0.5 * 0.25)).toBe(true);
    expect(close(vals[3], -(0.3 * 0.1) - (0.2 * 0.4) + 0.5 * 0.25)).toBe(true);
  });

  it('averages over the group only', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      compute v two vacf
      velocity all set 0.6 0.8 0.0 units box
      run 0
    `);
    const vals = s.sys.compute('v').vectorValues();
    // group "two" is atom 2 alone: v0 = (0.3, -0.2, 0.5), v = (0.6, 0.8, 0)
    expect(close(vals[3], 0.6 * 0.3 - 0.8 * 0.2)).toBe(true);
  });

  it('rejects arguments', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute v all vacf 3`)).rejects.toThrow('compute vacf takes no arguments');
  });
});

describe('compute rdf', () => {
  // atoms 1, 2 of type 1 at x = 2 and 3 (pair distance 1), atom 3 of type 2 at
  // (4, 2, 2): distances from the type-1 atoms 2.0 and 1.0
  const RDF_SYSTEM = `
    units lj
    atom_style atomic
    boundary p p p
    region box block 0 10 0 10 0 10
    create_box 2 box
    create_atoms 1 single 2.0 2.0 2.0
    create_atoms 1 single 3.0 2.0 2.0
    create_atoms 2 single 4.0 2.0 2.0
    mass 1 1.0
    mass 2 1.0
    pair_style lj/cut 5.0
    pair_coeff 1 1 1.0 1.0
    pair_coeff 2 2 1.0 1.0
    pair_coeff 1 2 1.0 1.0
  `;

  it('normalizes g(r) with the ideal count of ordered pairs', async () => {
    const s = await runScript(`
      ${RDF_SYSTEM}
      compute r all rdf 1 1 1 1 2 cutoff 5.0
      run 0
    `);
    const r = s.sys.compute('r');
    expect(r.arrayFlag).toBe(true);
    expect(r.sizeArrayRows).toBe(1);
    expect(r.sizeArrayCols).toBe(5);
    const a = r.arrayValues();
    // one bin [0, 5); shell volume (4/3) pi 5^3, box volume 1000
    const shell = (4 / 3) * Math.PI * 5 ** 3;
    const nideal = (f: number) => (f * shell) / 10 ** 3;
    // pair (1,1): hist = 2 (the pair counted for both orderings), ideal = 2*2 - 2 self pairs
    expect(close(a[1], 2 / nideal(2 * 2 - 2))).toBe(true);
    expect(close(a[2], 2 / 2)).toBe(true);
    // pair (1,2): hist = 2 (each type-1 atom pairs with the type-2 atom), ideal = 2*1
    expect(close(a[3], 2 / nideal(2 * 1))).toBe(true);
    expect(close(a[4], 2 / 2)).toBe(true);
    expect(close(a[0], 2.5)).toBe(true);
  });

  it('wildcard ranges and the default all-types histogram', async () => {
    const s = await runScript(`
      ${RDF_SYSTEM}
      compute w all rdf 2 1* 2 cutoff 5.0
      compute d all rdf 2 cutoff 5.0
      run 0
    `);
    const shell1 = (4 / 3) * Math.PI * 2.5 ** 3;
    const nideal = (f: number) => (f * shell1) / 10 ** 3;
    const w = s.sys.compute('w');
    expect(w.sizeArrayCols).toBe(3);
    const aw = w.arrayValues();
    // pair (1*,2) = itype {1,2}, jtype {2}: the two type-1 atoms each pair with the
    // type-2 atom (ordered), hist = 2; countI = 3, countJ = 1, shared = 1
    expect(close(aw[1], 2 / nideal(3 * 1 - 1))).toBe(true);
    expect(close(aw[2], 2 / 3)).toBe(true);
    expect(aw[4]).toBe(0);
    expect(close(aw[5], 2 / 3)).toBe(true); // coord is cumulative
    const d = s.sys.compute('d');
    const ad = d.arrayValues();
    // default histogram: itype = jtype = all types; all three ordered pairs are
    // within 5.0, each counted twice: hist = 6, ideal = 3*3 - 3
    expect(close(ad[1], 6 / nideal(3 * 3 - 3))).toBe(true);
    expect(close(ad[2], 6 / 3)).toBe(true);
    expect(close(ad[3], 3.75)).toBe(true); // second bin center
    expect(ad[4]).toBe(0);
    expect(close(ad[5], 6 / 3)).toBe(true);
  });

  it('bins only distances below the cutoff', async () => {
    const s = await runScript(`
      ${RDF_SYSTEM}
      compute r all rdf 5 1 1 cutoff 1.5
      run 0
    `);
    const a = s.sys.compute('r').arrayValues();
    // delr = 0.3; the type-1 pair at r = 1.0 lands in bin index 3 ([0.9, 1.2))
    const shell = (4 / 3) * Math.PI * 0.3 ** 3 * (4 ** 3 - 3 ** 3);
    for (let b = 0; b < 5; b++) {
      if (b === 3) continue;
      expect(a[b * 3 + 1]).toBe(0);
      // coord is cumulative: bins before the pair's bin are 0, later bins keep 1
      expect(a[b * 3 + 2]).toBe(b < 3 ? 0 : 1);
    }
    expect(close(a[3 * 3 + 1], 2 / ((2 * 2 - 2) * shell / 10 ** 3))).toBe(true);
    expect(close(a[3 * 3 + 2], 1)).toBe(true);
  });

  it('rejects bad arguments', async () => {
    await expect(runScript(`${RDF_SYSTEM}\ncompute r all rdf 0 1 1 cutoff 5.0`)).rejects.toThrow('Nbin must be >= 1');
    await expect(runScript(`${RDF_SYSTEM}\ncompute r all rdf 1 1 3 cutoff 5.0`)).rejects.toThrow('out of range');
    await expect(runScript(`${RDF_SYSTEM}\ncompute r all rdf 1 1 cutoff 5.0`)).rejects.toThrow('in pairs');
    await expect(runScript(`${RDF_SYSTEM}\ncompute r all rdf 1 1 1 cutoff 0`)).rejects.toThrow('cutoff must be > 0');
    await expect(runScript(`${RDF_SYSTEM}\ncompute r all rdf 1 1 1 frobnicate 5.0`)).rejects.toThrow('frobnicate');
    await expect(runScript(`${RDF_SYSTEM}\ncompute r all rdf 1 C1 1 cutoff 5.0`)).rejects.toThrow('type labels are not supported');
  });
});
