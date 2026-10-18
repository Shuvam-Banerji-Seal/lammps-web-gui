import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * compute centroid/stress/atom (src/engine/compute/peratom.ts ComputeCentroidStressAtom): the refusals with native's
 * messages (checked against native LAMMPS b332f28 on the same inputs) and the pairwise identity (a pair style is
 * CENTROID_SAME: the centroid tensor is the stress/atom tensor with yx zx zy = xy xz yz). Native parity of the
 * many-body terms is tests/oracle/w247cs_*.
 */

async function run(script: string) {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  try {
    await session.execute(script);
  } catch {
    // the failure is reported as an error event below
  }
  const err = events.find((e) => e.kind === 'error');
  return { error: err && 'message' in err ? String(err.message) : undefined, session };
}

const LJ = `units lj
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 4928 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
`;

const CHARGED = `units lj
atom_style charge
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 2 box
create_atoms 1 box
set type 1 type/fraction 2 0.5 123
set type 1 charge 1
set type 2 charge -1
mass * 1.0
pair_style lj/cut/coul/long 2.5
pair_coeff * * 1.0 1.0
kspace_style ewald 1e-4
`;

describe('compute centroid/stress/atom', () => {
  it('refuses what native refuses, with native messages', async () => {
    const cases: [string, RegExp][] = [
      [`${LJ}compute c all centroid/stress/atom NULL bogus\nrun 0\n`, /Illegal compute centroid\/stress\/atom command/],
      [`${LJ}compute c all centroid/stress/atom nope\nrun 0\n`, /Could not find compute centroid\/stress\/atom temperature ID nope/],
      [`${LJ}fix r all rigid single\ncompute c all centroid/stress/atom NULL\nrun 0\n`, /Fix rigid does not support compute centroid\/stress\/atom/],
      [`${CHARGED}compute c all centroid/stress/atom NULL\nrun 0\n`, /KSpace style does not support compute centroid\/stress\/atom/],
      [`units lj\natom_style sphere\nlattice fcc 0.8442\nregion box block 0 2 0 2 0 2\ncreate_box 1 box\ncreate_atoms 1 box\npair_style gran/hooke 2000.0 NULL 50.0 NULL 0.5 0\npair_coeff * *\ncomm_modify vel yes\ncompute c all centroid/stress/atom NULL\nrun 0\n`,
        /Pair style does not support compute centroid\/stress\/atom/],
    ];
    for (const [script, msg] of cases) expect((await run(script)).error, String(msg)).toMatch(msg);
  });

  it('runs with a kspace style when the kspace term is not selected', async () => {
    const r = await run(`${CHARGED}compute c all centroid/stress/atom NULL pair\nthermo_style custom step pe\nrun 0\n`);
    expect(r.error).toBeUndefined();
  });

  it('equals stress/atom for a pair-only system, with yx zx zy copied from xy xz yz', async () => {
    const r = await run(`${LJ}compute st all stress/atom NULL\ncompute cs all centroid/stress/atom NULL\nthermo_style custom step pe\nrun 0\n`);
    expect(r.error).toBeUndefined();
    const sys = (r.session as unknown as { sys: { compute(id: string): { peratomValues(): Float64Array }; state: { n: number } } }).sys;
    const st = sys.compute('st').peratomValues(), cs = sys.compute('cs').peratomValues();
    for (let i = 0; i < sys.state.n; i++) {
      for (let c = 0; c < 6; c++) expect(cs[9 * i + c]).toBeCloseTo(st[6 * i + c], 10);
      for (let c = 0; c < 3; c++) expect(cs[9 * i + 6 + c]).toBeCloseTo(st[6 * i + 3 + c], 10);
    }
  });
});
