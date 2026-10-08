import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { ComputeTempRamp } from '../src/engine/compute/temp_ramp';
import { ComputeTempCs } from '../src/engine/compute/temp_cs';
import { ComputeTempChunk } from '../src/engine/compute/temp_chunk';

/** Relative closeness for checks against hand-written formulas. */
const close = (got: number, want: number, rel = 1e-10) => Math.abs(got - want) <= rel * Math.max(1, Math.abs(got), Math.abs(want));

/** Runs input text in a fresh session. */
const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  await session.execute(text);
  return session;
};

/** Two atoms (ids 1, 2) at x = 0 and 1, unit mass, lj units (mvv2e = boltz = 1), lattice spacing 1. */
const TWO_ATOMS = `
units lj
atom_style atomic
boundary p p p
lattice sc 1.0
region box block 0 2 0 1 0 1
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
`;

/** Four atoms at x = 0, 1, 2, 3, bonded 1-2 and 3-4 (core/shell pairs), lj units, lattice spacing 1. */
const FOUR_ATOMS = `
units lj
atom_style bond
boundary p p p
lattice sc 1.0
region box block 0 4 0 1 0 1
create_box 1 box bond/types 1 extra/bond/per/atom 1
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
bond_style harmonic
bond_coeff 1 10.0 1.0
create_bonds single/bond 1 1 2
create_bonds single/bond 1 3 4
`;

describe('compute temp/ramp', () => {
  it('subtracts the ramp with clamping outside the bounds (box units)', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 1.5 -0.5 0.25 units box
      compute r all temp/ramp vx 0.0 2.0 x 0.0 1.0 units box
      run 0
    `);
    const r = s.sys.compute('r') as ComputeTempRamp;
    expect(r.tempFlag).toBe(true);
    expect(r.hasBias()).toBe(true);
    // atom at x = 0 (f = 0, ramp 0) keeps vx = 1.5; atom at x = 1 (f = 1, ramp 2) gives vx - 2 = -0.5
    const ux = [1.5, -0.5];
    const uy = [-0.5, -0.5];
    const uz = [0.25, 0.25];
    const kin = ux.reduce((a, u, i) => a + u * u + uy[i] * uy[i] + uz[i] * uz[i], 0);
    expect(r.dof).toBe(3); // 3 * 2 - 3
    expect(close(r.scalarValue(), kin / 3)).toBe(true);
    const t = r.vectorValues();
    expect(close(t[0], ux[0] * ux[0] + ux[1] * ux[1])).toBe(true);
    expect(close(t[3], ux[0] * uy[0] + ux[1] * uy[1])).toBe(true);
  });

  it('scales the coordinate bounds and velocities with the lattice spacing', async () => {
    const sp = (4 / 0.8442) ** (1 / 3); // lattice fcc 0.8442 spacing (4 atoms per cell)
    const s = await runScript(`
      ${FOUR_ATOMS}
      lattice fcc 0.8442
      velocity all set 1.0 0.0 0.0 units box
      compute r all temp/ramp vx 0.0 1.0 x 0.0 1.0 units lattice
      compute b all temp/ramp vx 0.0 ${sp} x 0.0 ${sp} units box
      run 0
    `);
    expect(s.sys.lattice!.spacing[0]).toBeCloseTo(sp, 6);
    expect(close(s.sys.compute('r').scalarValue(), s.sys.compute('b').scalarValue(), 1e-9)).toBe(true);
  });

  it('rejects an unknown keyword, a bad units value and a bad vdim', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute r all temp/ramp vx 0 1 x 0 1 foo bar`)).rejects.toThrow('unknown keyword');
    await expect(runScript(`${TWO_ATOMS}\ncompute r all temp/ramp vx 0 1 x 0 1 units real`)).rejects.toThrow('units must be lattice or box');
    await expect(runScript(`${TWO_ATOMS}\ncompute r all temp/ramp vw 0 1 x 0 1`)).rejects.toThrow('vdim must be vx, vy or vz');
    await expect(runScript(`${TWO_ATOMS}\ncompute r all temp/ramp vx 0 1 x 1 1`)).rejects.toThrow('chi must be greater than clo');
  });

  it('removes and restores the ramp bias exactly (thermostat contract)', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 1.5 -0.5 0.25 units box
      compute r all temp/ramp vx 0.0 2.0 x 0.0 1.0 units box
      run 0
    `);
    const before = Float64Array.from(s.sys.state.v);
    const r = s.sys.compute('r') as ComputeTempRamp;
    r.computeBias();
    r.removeBiasAll();
    s.sys.refreshComputes();
    // after removal the stored velocities are thermal: vx = (1.5, -0.5); the scalar must not subtract the ramp again
    expect(s.sys.state.v[0]).toBeCloseTo(1.5, 12);
    expect(s.sys.state.v[3]).toBeCloseTo(-0.5, 12);
    expect(close(r.scalarValue(), (2.25 + 0.25 + 0.0625 + 0.25 + 0.25 + 0.0625) / 3)).toBe(true);
    r.restoreBiasAll();
    for (let k = 0; k < before.length; k++) expect(s.sys.state.v[k]).toBe(before[k]);
  });
});

describe('compute temp/cs', () => {
  it('counts each core/shell pair as one particle and uses the pair COM velocity for the KE', async () => {
    const s = await runScript(`
      ${FOUR_ATOMS}
      group cores id 1 3
      group shells id 2 4
      velocity all set 0.6 0.2 -0.4 units box
      compute cs all temp/cs cores shells
      run 0
    `);
    const cs = s.sys.compute('cs') as ComputeTempCs;
    // all velocities equal: the pair COM equals the atom velocity, so the KE is 4 * |v|^2 / 2
    // dof = 3 (4 - 2) - 3 = 3
    expect(cs.dof).toBe(3);
    const v2 = 0.36 + 0.04 + 0.16;
    expect(close(cs.scalarValue(), 4 * v2 / 3)).toBe(true);
    // the tensor is computed on the plain velocities (measured with native LAMMPS)
    expect(close(cs.vectorValues()[0], 4 * 0.36)).toBe(true);
  });

  it('uses the mass-weighted pair COM in the scalar', async () => {
    const TWO_TYPES = FOUR_ATOMS.replace('create_box 1 box', 'create_box 2 box').replace('pair_coeff 1 1 1.0 1.0', 'pair_coeff * * 1.0 1.0').replace('mass 1 1.0', 'mass * 1.0\nmass 2 4.0');
    const s = await runScript(`
      ${TWO_TYPES}
      group cores id 1 3
      group shells id 2 4
      set atom 1 type 2
      set atom 3 type 2
      velocity all set 0.0 0.0 0.0 units box
      set atom 1 vx 2.0
      set atom 2 vx -1.0
      compute cs all temp/cs cores shells
      run 0
    `);
    const cs = s.sys.compute('cs');
    // pair 1-2: masses 4 and 1, COM vx = (4*2 + 1*(-1)) / 5 = 1.4 for both atoms; pair 3-4 at rest
    // KE sum = 4 * 1.4^2 + 1 * 1.4^2 = 5 * 1.96 ; dof = 3 * 2 - 3 = 3
    expect(close(cs.scalarValue(), 5 * 1.96 / 3)).toBe(true);
  });

  it('applies the documented restrictions as StyleErrors', async () => {
    await expect(runScript(`${FOUR_ATOMS}\ngroup cores id 1 3\ngroup shells id 2\ncompute cs all temp/cs cores shells\nrun 0`))
      .rejects.toThrow('must be equal');
    await expect(runScript(`${FOUR_ATOMS}\ngroup cores id 1 3\ngroup shells id 2 4\ncompute cs all temp/cs cores cores\nrun 0`))
      .rejects.toThrow('no bond to the other group');
    await expect(runScript(`${FOUR_ATOMS}\ncompute cs all temp/cs cores`)).rejects.toThrow('usage');
  });

  it('removes the pair-relative velocity as a bias and restores it', async () => {
    const s = await runScript(`
      ${FOUR_ATOMS}
      group cores id 1 3
      group shells id 2 4
      velocity all set 0.6 0.2 -0.4 units box
      set atom 2 vx 1.0
      compute cs all temp/cs cores shells
      run 0
    `);
    const before = Float64Array.from(s.sys.state.v);
    const cs = s.sys.compute('cs') as ComputeTempCs;
    expect(cs.hasBias()).toBe(true);
    cs.computeBias();
    cs.removeBiasAll();
    // the pair members (atoms 1 and 2, equal masses) now share the COM velocity 0.8
    const v = s.sys.state.v;
    expect(v[0]).toBeCloseTo(0.8, 12);
    expect(v[3]).toBeCloseTo(0.8, 12);
    cs.restoreBiasAll();
    for (let k = 0; k < before.length; k++) expect(s.sys.state.v[k]).toBe(before[k]);
  });
});

describe('compute temp/chunk', () => {
  // one chunk (type chunk with a single type) holding both atoms of TWO_ATOMS
  const ONE_CHUNK = `
    ${TWO_ATOMS}
    velocity all set 1.5 -0.5 0.25 units box
    set atom 2 vx -0.5
    compute cc all chunk/atom type
  `;

  it('gives the global temperature, the tensor and the per-chunk array', async () => {
    const s = await runScript(`${ONE_CHUNK}
      compute tc all temp/chunk cc temp kecom internal
      run 0`);
    const tc = s.sys.compute('tc') as ComputeTempChunk;
    const ux = [1.5, -0.5], uy = [-0.5, -0.5], uz = [0.25, 0.25];
    const kin = ux.reduce((a, u, i) => a + u * u + uy[i] * uy[i] + uz[i] * uz[i], 0);
    // dof = 3 * 2 + 0 * 1 = 6 (adof default is the dimension, cdof 0)
    expect(tc.dof).toBe(6);
    expect(close(tc.scalarValue(), kin / 6)).toBe(true);
    expect(tc.arrayFlag).toBe(true);
    const arr = tc.arrayValues();
    expect(tc.sizeArrayRows).toBe(1);
    expect(tc.sizeArrayCols).toBe(3);
    // per-chunk temp row: same as the global value (single chunk)
    expect(close(arr[0], kin / 6)).toBe(true);
    // tensor xx without com: 1.5^2 + (-0.5)^2
    expect(close(tc.vectorValues()[0], 2.5)).toBe(true);
    // kecom: the chunk COM velocity is (0.5, -0.5, 0.25): 0.5 * M * |vcm|^2 with M = 2
    const vcm2 = 0.25 + 0.25 + 0.0625;
    expect(close(arr[1], 0.5 * 2 * vcm2)).toBe(true);
    // internal: 0.5 sum m |v - vcm|^2 with v - vcm = (+1, 0, 0) and (-1, 0, 0)
    const internal = 0.5 * (1 + 1);
    expect(close(arr[2], internal)).toBe(true);
  });

  it('subtracts the per-chunk COM with com yes and removes/restores it as a bias', async () => {
    const s = await runScript(`${ONE_CHUNK}
      compute tcom all temp/chunk cc temp com yes
      run 0`);
    const tcom = s.sys.compute('tcom') as ComputeTempChunk;
    expect(tcom.hasBias()).toBe(true);
    // thermal velocities are v - vcm = (+-1, 0, 0): KE sum 2, dof 6
    expect(close(tcom.scalarValue(), 2 / 6)).toBe(true);
    const before = Float64Array.from(s.sys.state.v);
    tcom.computeBias();
    tcom.removeBiasAll();
    const mid = s.sys.state.v;
    expect(mid[0] + mid[3]).toBeCloseTo(0, 12);
    tcom.restoreBiasAll();
    for (let k = 0; k < before.length; k++) expect(s.sys.state.v[k]).toBe(before[k]);
  });

  it('rejects com yes together with bias, and unknown keywords', async () => {
    await expect(runScript(`${ONE_CHUNK}\ncompute t all temp/chunk cc com yes bias tp\ncompute tp all temp/partial 1 1 1`))
      .rejects.toThrow('cannot be used together');
    await expect(runScript(`${ONE_CHUNK}\ncompute t all temp/chunk cc foo bar`)).rejects.toThrow('unknown value or keyword');
    await expect(runScript(`${ONE_CHUNK}\ncompute t all temp/chunk cc com maybe`)).rejects.toThrow('com must be yes or no');
  });

  it('uses adof and cdof in the DOF formula N adof + Nchunk cdof', async () => {
    const s = await runScript(`${ONE_CHUNK}
      compute t all temp/chunk cc temp adof 2.0 cdof 1.5
      run 0`);
    const t = s.sys.compute('t') as ComputeTempChunk;
    // dof = 2.0 * 2 + 1.5 * 1 = 5.5
    expect(t.dof).toBeCloseTo(5.5, 12);
    expect(close(t.scalarValue(), 3.125 / 5.5)).toBe(true);
  });
});
