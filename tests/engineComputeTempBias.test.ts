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

/** Two atoms (ids 1, 2) at x = 0 and 1, unit mass, lj units (mvv2e = boltz = 1). */
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

describe('compute temp/partial', () => {
  it('computes the temperature of the included components with dof = ninclude*N - scaled extra/dof', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      compute tp all temp/partial 1 1 0
      run 0
    `);
    const tp = s.sys.compute('tp');
    expect(tp.tempFlag).toBe(true);
    expect(tp.hasBias()).toBe(true);
    // sum m (vx^2 + vy^2) = 2 * (0.09 + 0.04) = 0.26; dof = 2*2 - 2 = 2
    expect(tp.dof).toBe(2);
    expect(close(tp.scalarValue(), 0.26 / 2)).toBe(true);
    // tensor: excluded z components are zero
    const t = tp.vectorValues();
    expect(close(t[0], 2 * 0.09)).toBe(true);
    expect(close(t[1], 2 * 0.04)).toBe(true);
    expect(t[2]).toBe(0);
    expect(close(t[3], 2 * 0.3 * -0.2)).toBe(true);
    expect(t[4]).toBe(0);
    expect(t[5]).toBe(0);
  });

  it('scales the extra/dof value with the fraction of included components', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      group half id 2
      compute tp half temp/partial 1 0 0
      run 0
    `);
    const tp = s.sys.compute('tp');
    // dof = 1*1 - 3*(1/3) = 0: the single x component carries no thermal DOF
    expect(tp.dof).toBe(0);
    expect(tp.scalarValue()).toBe(0);
  });

  it('honours compute_modify extra/dof 0 for a one-component group temperature', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      group half id 2
      compute tp half temp/partial 1 0 0
      compute_modify tp extra/dof 0
      run 0
    `);
    const tp = s.sys.compute('tp');
    expect(tp.dof).toBe(1);
    expect(close(tp.scalarValue(), 0.09)).toBe(true);
  });

  it('removes and restores the excluded velocity components of group atoms only', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      group half id 2
      compute tp half temp/partial 1 1 0
      run 0
    `);
    const c = s.sys.compute('tp');
    const v = s.sys.state.v;
    const v0 = v.slice();
    c.computeBias();
    c.removeBiasAll();
    // atom 0 is outside the compute group: untouched
    expect(close(v[0], v0[0]) && close(v[2], v0[2])).toBe(true);
    // atom 1: z zeroed, x/y kept
    expect(close(v[3], 0.3) && close(v[4], -0.2)).toBe(true);
    expect(v[5]).toBe(0);
    c.restoreBiasAll();
    expect(close(v[3], v0[3]) && close(v[4], v0[4]) && close(v[5], v0[5])).toBe(true);
  });

  it('rejects bad flags and argument counts', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute k all temp/partial 1 1`)).rejects.toThrow('temp/partial xflag yflag zflag');
    await expect(runScript(`${TWO_ATOMS}\ncompute k all temp/partial 1 1 2`)).rejects.toThrow('must be 0 or 1');
    await expect(runScript(`${TWO_ATOMS}\ncompute k all temp/partial 0 0 0`)).rejects.toThrow('all three flags are 0');
  });
});

describe('compute temp/com', () => {
  it('subtracts the mass-weighted center-of-mass velocity of the group', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.5 0 0 units box
      group two id 2
      velocity two set -0.1 0.2 0 units box
      compute tc all temp/com
      run 0
    `);
    const tc = s.sys.compute('tc');
    expect(tc.hasBias()).toBe(true);
    // vcm = ((0.5 - 0.1)/2, 0.1, 0); relative KE = 0.5 * 2 * (0.09 + 0.01) = 0.1; dof = 3
    expect(tc.dof).toBe(3);
    expect(close(tc.scalarValue(), 2 * 0.1 / 3)).toBe(true);
    const t = tc.vectorValues();
    expect(close(t[0], 0.18)).toBe(true);
    expect(close(t[1], 0.02)).toBe(true);
    expect(close(t[3], 2 * 0.3 * -0.1)).toBe(true);
  });

  it('removes the center-of-mass velocity and adds it back unchanged', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.5 0 0 units box
      group two id 2
      velocity two set -0.1 0.2 0 units box
      compute tc all temp/com
      run 0
    `);
    const c = s.sys.compute('tc');
    const v = s.sys.state.v;
    const v0 = v.slice();
    c.computeBias();
    c.removeBiasAll();
    expect(close(v[0], 0.3) && close(v[1], -0.1) && close(v[2], 0)).toBe(true);
    expect(close(v[3], -0.3) && close(v[4], 0.1)).toBe(true);
    c.restoreBiasAll();
    for (let k = 0; k < 6; k++) expect(close(v[k], v0[k])).toBe(true);
  });

  it('rejects arguments', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute k all temp/com 1`)).rejects.toThrow('temp/com takes no arguments');
  });
});

describe('compute temp/region', () => {
  it('counts only group atoms inside the region, boundary inclusive, and recomputes dof', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      region right block 0.5 INF INF INF INF INF units box
      compute tr all temp/region right
      compute_modify tr extra/dof 0
      run 0
    `);
    const tr = s.sys.compute('tr');
    // atom 2 (x = 1.0) is inside (0.5 <= x), atom 1 (x = 0) is not; dof = 3*1 - 0 = 3
    expect(tr.dof).toBe(3);
    expect(close(tr.scalarValue(), (0.09 + 0.04 + 0.25) / 3)).toBe(true);
  });

  it('treats atoms exactly on the region boundary as interior', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      region edge block 1.0 INF INF INF INF INF units box
      compute te all temp/region edge
      compute_modify te extra/dof 0
      run 0
    `);
    const te = s.sys.compute('te');
    expect(te.dof).toBe(3);
    expect(close(te.scalarValue(), (0.09 + 0.04 + 0.25) / 3)).toBe(true);
  });

  it('does not subtract fix-removed degrees of freedom', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      region right block 0.5 INF INF INF INF INF units box
      fix 1 all nve
      compute tr all temp/region right
      compute_modify tr extra/dof 0
      run 0
    `);
    expect(s.sys.compute('tr').dof).toBe(3);
  });

  it('removes the velocity of atoms outside the region and restores it', async () => {
    const s = await runScript(`
      ${TWO_ATOMS}
      velocity all set 0.3 -0.2 0.5 units box
      region right block 0.5 INF INF INF INF INF units box
      compute tr all temp/region right
      run 0
    `);
    const c = s.sys.compute('tr');
    const v = s.sys.state.v;
    const v0 = v.slice();
    c.computeBias();
    c.removeBiasAll();
    // atom 0 is outside the region: zeroed; atom 1 inside: untouched
    expect(v[0]).toBe(0); expect(v[1]).toBe(0); expect(v[2]).toBe(0);
    expect(close(v[3], 0.3) && close(v[4], -0.2) && close(v[5], 0.5)).toBe(true);
    c.restoreBiasAll();
    for (let k = 0; k < 6; k++) expect(close(v[k], v0[k])).toBe(true);
  });

  it('rejects a missing region-ID and an unknown region', async () => {
    await expect(runScript(`${TWO_ATOMS}\ncompute k all temp/region`)).rejects.toThrow('temp/region region-ID');
    await expect(runScript(`${TWO_ATOMS}\ncompute k all temp/region nowhere\nrun 0`)).rejects.toThrow("region ID 'nowhere' does not exist");
  });
});
