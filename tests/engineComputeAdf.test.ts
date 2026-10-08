import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * compute adf — docs.lammps.org/compute_adf.html. Native parity is in
 * tests/oracle/w21adf_lj.in (rel 1e-9). Here the doc rules and the rules the
 * page leaves open — both measured with native LAMMPS (black box), probes in
 * plans/scratch/adf — are checked on hand-computable geometries:
 *  - the shell test is Rinner <= R <= Router (both ends included);
 *  - bins are half-open, so an ordinate at the top of the range (180 degrees,
 *    cosine +1) is dropped while cosine -1 is kept;
 *  - first ADF = hist_b / (Nangles * width), second = cumulative / Ncentral;
 *  - the J and K roles are symmetric (order independent).
 */

/** Relative closeness for the checks against manual formulas. */
const close = (got: number, want: number, rel = 1e-12) => Math.abs(got - want) <= rel * Math.max(1, Math.abs(got), Math.abs(want));

/** Runs input text in a fresh session. */
const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  await session.execute(text);
  return session;
};

/** A 5-atom cluster: type-1 centre at (10,10,10) and four type-2 neighbours. */
const C4 = (coords: string) => `
units lj
atom_style atomic
boundary p p p
region box block 0 20 0 20 0 20
create_box 2 box
create_atoms 1 single 10.0 10.0 10.0
${coords}
mass 1 1.0
mass 2 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
`;

/** Neighbours at 0, 45, 90 and 180 degrees, all at distance 1. */
const FOUR_DIRECTIONS = `create_atoms 2 single 11.0 10.0 10.0
create_atoms 2 single 10.7071067811865475 10.7071067811865475 10.0
create_atoms 2 single 10.0 11.0 10.0
create_atoms 2 single 9.0 10.0 10.0
`;

/** A type-1 centre with a type-2 neighbour at +x and a type-1 neighbour at +y. */
const MIXED = `
units lj
atom_style atomic
boundary p p p
region box block 0 20 0 20 0 20
create_box 2 box
create_atoms 1 single 10.0 10.0 10.0
create_atoms 2 single 11.0 10.0 10.0
create_atoms 1 single 10.0 11.0 10.0
mass 1 1.0
mass 2 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
`;

describe('compute adf', () => {
  it('degree histogram: binning, normalization and the dropped 180-degree angle', async () => {
    const s = await runScript(`${C4(FOUR_DIRECTIONS)}
      compute a all adf 36 1 2 2 0.0 2.0 0.0 2.0
      run 0
    `);
    const c = s.sys.compute('a');
    expect(c.arrayFlag).toBe(true);
    expect(c.sizeArrayRows).toBe(36);
    expect(c.sizeArrayCols).toBe(3);
    const a = c.arrayValues();
    // ordinate at the midpoint of each 5-degree bin
    for (let b = 0; b < 36; b++) expect(close(a[b * 3], (b + 0.5) * 5)).toBe(true);
    // pairs: (0,45)=45, (0,90)=90, (0,180)=180 dropped, (45,90)=45, (45,180)=135, (90,180)=90
    // Nangles = 5, width = 5, centre count Ncentral = 1
    const first = (bin: number) => a[bin * 3 + 1];
    const second = (bin: number) => a[bin * 3 + 2];
    expect(close(first(9), 2 / (5 * 5))).toBe(true); // 45 degrees
    expect(close(first(18), 2 / (5 * 5))).toBe(true); // 90 degrees
    expect(close(first(27), 1 / (5 * 5))).toBe(true); // 135 degrees
    expect(first(35)).toBe(0); // 180 degrees was dropped, not clamped
    expect(second(9)).toBe(2);
    expect(second(18)).toBe(4);
    expect(second(27)).toBe(5);
    // the first column is a probability density: integral over the range is 1
    let integral = 0;
    for (let b = 0; b < 36; b++) integral += first(b) * 5;
    expect(close(integral, 1)).toBe(true);
  });

  it('ordinate cosine keeps cosine -1 and bins uniformly in the cosine', async () => {
    const s = await runScript(`${C4(FOUR_DIRECTIONS)}
      compute a all adf 4 1 2 2 0.0 2.0 0.0 2.0 ordinate cosine
      run 0
    `);
    const a = s.sys.compute('a').arrayValues();
    expect(close(a[0], -0.75)).toBe(true);
    expect(close(a[3 * 3], 0.75)).toBe(true);
    // cos(45)=0.707 -> bin 4, cos(90)=0 -> bin 3, cos(135)=-0.707 and cos(180)=-1 -> bin 1
    // six angles, width 0.5, so first = hist / 3
    expect(close(a[3 * 3 + 1], 2 / 3)).toBe(true);
    expect(close(a[2 * 3 + 1], 2 / 3)).toBe(true);
    expect(close(a[0 * 3 + 1], 2 / 3)).toBe(true);
    expect(a[1 * 3 + 1]).toBe(0);
    expect(a[0 * 3 + 2]).toBe(2); // cumulative includes the 135 and 180 degree angles
    let integral = 0;
    for (let b = 0; b < 4; b++) integral += a[b * 3 + 1] * 0.5;
    expect(close(integral, 1)).toBe(true);
  });

  it('ordinate radian bins uniformly from 0 to Pi', async () => {
    const s = await runScript(`${C4(FOUR_DIRECTIONS)}
      compute a all adf 4 1 2 2 0.0 2.0 0.0 2.0 ordinate radian
      run 0
    `);
    const a = s.sys.compute('a').arrayValues();
    const w = Math.PI / 4;
    for (let b = 0; b < 4; b++) expect(close(a[b * 3], (b + 0.5) * w)).toBe(true);
    let integral = 0;
    for (let b = 0; b < 4; b++) integral += a[b * 3 + 1] * w;
    expect(close(integral, 1)).toBe(true);
    // the 180-degree (Pi) angle is dropped: five angles in total, cumulative max 5
    expect(close(a[3 * 3 + 2], 5)).toBe(true);
  });

  it('the J and K roles are symmetric, independent of neighbour order', async () => {
    // the type-3 neighbour is created after the type-2 one, so a loop that only
    // tests the role of the earlier list entry would miss (jtype=3, ktype=2)
    const sym = `units lj
      atom_style atomic
      boundary p p p
      region box block 0 20 0 20 0 20
      create_box 3 box
      create_atoms 1 single 10.0 10.0 10.0
      create_atoms 2 single 11.0 10.0 10.0
      create_atoms 3 single 10.0 11.0 10.0
      mass 1 1.0
      mass 2 1.0
      mass 3 1.0
      pair_style lj/cut 2.5
      pair_coeff * * 1.0 1.0
    `;
    const s = await runScript(`${sym}
      compute jk all adf 4 1 3 2 0.0 2.0 0.0 2.0
      compute kj all adf 4 1 2 3 0.0 2.0 0.0 2.0
      run 0
    `);
    const jk = s.sys.compute('jk').arrayValues();
    const kj = s.sys.compute('kj').arrayValues();
    // the single 90-degree angle between the type-2 (+x) and type-3 (+y) neighbour
    // is found by both role assignments
    expect(close(jk[2 * 3 + 1], 1 / 45)).toBe(true);
    expect(close(kj[2 * 3 + 1], 1 / 45)).toBe(true);
    expect(jk.every((v, i) => close(v, kj[i]))).toBe(true);
  });

  it('includes neighbours exactly at Rinner and Router', async () => {
    const outer = await runScript(`${MIXED}
      compute a all adf 4 1 2 1 0.0 2.0 0.0 1.0
      run 0
    `);
    const inner = await runScript(`${MIXED}
      compute a all adf 4 1 2 1 0.0 2.0 1.0 2.0
      run 0
    `);
    const excluded = await runScript(`${MIXED}
      compute a all adf 4 1 2 1 0.0 2.0 0.0 0.999
      run 0
    `);
    expect(outer.sys.compute('a').arrayValues()[2 * 3 + 2]).toBe(1);
    expect(inner.sys.compute('a').arrayValues()[2 * 3 + 2]).toBe(1);
    expect(excluded.sys.compute('a').arrayValues()[2 * 3 + 2]).toBe(0);
  });

  it('default (no type triple) uses all types and the force cutoff', async () => {
    const s = await runScript(`units lj
      atom_style atomic
      boundary p p p
      region box block 0 20 0 20 0 20
      create_box 2 box
      create_atoms 1 single 10.0 10.0 10.0
      create_atoms 2 single 11.0 10.0 10.0
      create_atoms 2 single 10.0 11.0 10.0
      mass 1 1.0
      mass 2 1.0
      pair_style lj/cut 2.5
      pair_coeff * * 1.0 1.0
      compute a all adf 4
      run 0
    `);
    const c = s.sys.compute('a');
    expect(c.sizeArrayCols).toBe(3);
    const a = c.arrayValues();
    // centres are all three atoms; angles are 45 (atoms 2 and 3), 90 (atom 1):
    // 45 -> two angles, 90 -> one; Nangles = 3, Ncentral = 3
    expect(close(a[1 * 3 + 1], 2 / (3 * 45))).toBe(true);
    expect(close(a[2 * 3 + 1], 1 / (3 * 45))).toBe(true);
    expect(a[3 * 3 + 1]).toBe(0);
    expect(close(a[1 * 3 + 2], 2 / 3)).toBe(true);
    expect(close(a[2 * 3 + 2], 1)).toBe(true);
  });

  it('counts only neighbours whose type lies in the jtype and ktype ranges', async () => {
    const s = await runScript(`${MIXED}
      compute a all adf 4 1 1 1 0.0 2.0 0.0 2.0
      compute b all adf 4 1 2 1* 0.0 2.0 0.0 2.0
      run 0
    `);
    // no pair of two type-1 neighbours: every ADF value is zero
    const a = s.sys.compute('a').arrayValues();
    for (let b = 0; b < 4; b++) {
      expect(a[b * 3 + 1]).toBe(0);
      expect(a[b * 3 + 2]).toBe(0);
    }
    // (1,2,1*) keeps the type-2/type-1 angle
    expect(s.sys.compute('b').arrayValues()[2 * 3 + 2]).toBe(1);
  });

  it('restricts every angle to atoms of the compute group', async () => {
    const s = await runScript(`${MIXED}
      group only id 1
      compute a only adf 4 1 2 1 0.0 2.0 0.0 2.0
      run 0
    `);
    const a = s.sys.compute('a').arrayValues();
    // group "only" holds the centre, but J and K must be in the group too
    for (let b = 0; b < 4; b++) {
      expect(a[b * 3 + 1]).toBe(0);
      expect(a[b * 3 + 2]).toBe(0);
    }
  });

  it('rejects bad arguments with a StyleError', async () => {
    const base = `${MIXED}`;
    await expect(runScript(`${base}\ncompute a all adf 0 1 2 2 0.0 2.0 0.0 2.0`)).rejects.toThrow('Nbin must be >= 1');
    await expect(runScript(`${base}\ncompute a all adf 4 1 2 2 0.0 2.0 0.0 2.0 bogus 3`)).rejects.toThrow('bogus');
    await expect(runScript(`${base}\ncompute a all adf 4 1 2 2 1.0 0.5 0.0 2.0`)).rejects.toThrow('illegal j-cutoff');
    await expect(runScript(`${base}\ncompute a all adf 4 1 2 2 0.0 2.0 0.5 0.5`)).rejects.toThrow('illegal k-cutoff');
    await expect(runScript(`${base}\ncompute a all adf 4 1 2 2 0.0 2.0 0.0 2.0 ordinate bogus`)).rejects.toThrow('unknown ordinate');
    await expect(runScript(`${base}\ncompute a all adf 4 1 2 C1 0.0 2.0 0.0 2.0`)).rejects.toThrow('type labels are not supported');
    await expect(runScript(`${base}\ncompute a all adf 4 1 2 3 0.0 2.0 0.0 2.0`)).rejects.toThrow('out of range');
  });
});
