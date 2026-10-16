/**
 * Special lists are built only as deep as the special_bonds weights need, as native LAMMPS does (src/special.cpp
 * Special::build at b332f28): no 1-3/1-4 lists when their lj and coul weights are all 1.0, no 1-4 list when only its
 * weights are 1.0, and the full lists with one/five. A weight of 1.0 leaves a pair as if it were not special, so the
 * forces do not change; densely bonded systems (bpm, ~120 bonds per atom) no longer build lists to depth 3.
 */
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { buildSpecial } from '../src/engine/atoms';

const chain = (special: string) => [
  'units lj', 'atom_style bond', 'region b block 0 10 0 10 0 10', 'create_box 1 b bond/types 1 extra/bond/per/atom 4 extra/special/per/atom 10',
  ...[1, 2, 3, 4, 5].map((k) => `create_atoms 1 single ${k} 5 5`),
  'mass 1 1.0', 'pair_style lj/cut 2.5', 'pair_coeff 1 1 1.0 1.0', 'bond_style harmonic', 'bond_coeff 1 100.0 1.0',
  'create_bonds single/bond 1 1 2', 'create_bonds single/bond 1 2 3', 'create_bonds single/bond 1 3 4', 'create_bonds single/bond 1 4 5',
  special, 'thermo_style custom step pe', 'run 0', '',
].join('\n');

const orders = async (special: string) => {
  const s = new Session({ emit: () => {}, writeFile: () => {} }, undefined, 0, 1);
  await s.execute(chain(special));
  const list = (s.sys.ff as unknown as { specialList: { order: Int8Array } | null }).specialList!;
  return { counts: [1, 2, 3].map((o) => list.order.filter((x) => x === o).length), pe: s.sys.ff.acc.evdwl + s.sys.ff.acc.ebond };
};

describe('special list depth follows the weights', () => {
  it('builds 1-2 only when the 1-3 and 1-4 weights are 1.0', async () => {
    expect((await orders('special_bonds lj 0.0 1.0 1.0 coul 0.0 1.0 1.0')).counts).toEqual([8, 0, 0]);
  });
  it('builds 1-2 and 1-3 when only the 1-4 weights are 1.0', async () => {
    expect((await orders('special_bonds lj 0.0 0.5 1.0 coul 0.0 0.5 1.0')).counts).toEqual([8, 6, 0]);
  });
  it('builds every list for the default weights', async () => {
    expect((await orders('special_bonds lj 0.0 0.0 0.0')).counts).toEqual([8, 6, 4]);
  });
  it('leaves the energy unchanged when weights of 1.0 drop a list', async () => {
    // lj 0 1 1 skips the lists; lj 0 1 0.999999999 keeps them; the pair energies agree to the weight difference
    const a = await orders('special_bonds lj 0.0 1.0 1.0');
    const b = await orders('special_bonds lj 0.0 1.0 0.9999999999');
    expect(Math.abs(a.pe - b.pe)).toBeLessThan(1e-8);
  });
  it('buildSpecial honours maxOrder', () => {
    const s = { n: 3, id: Int32Array.from([1, 2, 3]), topo: { bonds: { n: 2, type: Int32Array.from([1, 1]), atoms: Int32Array.from([1, 2, 2, 3]) }, angles: { n: 0, atoms: new Int32Array(0) }, dihedrals: { n: 0, atoms: new Int32Array(0) } } };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(Array.from(buildSpecial(s as any, { maxOrder: 1 }).order)).toEqual([1, 1, 1, 1]);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(Array.from(buildSpecial(s as any).order).sort()).toEqual([1, 1, 1, 1, 2, 2]);
  });
});
