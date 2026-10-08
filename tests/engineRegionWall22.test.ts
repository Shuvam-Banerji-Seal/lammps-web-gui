import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix wall/region and fix wall/gran/region on compound (union/intersect) and
 * dynamic (move/rotate) regions — docs.lammps.org/fix_wall_region.html,
 * docs.lammps.org/fix_wall_gran_region.html and docs.lammps.org/region.html.
 *
 * "Regions can either be primitive shapes (block, sphere, cylinder, etc) or
 * combinations of primitive shapes specified via the union or intersect region
 * styles."  "Regions can also move dynamically via the region command keywords
 * (move) and rotate."  "LAMMPS discards points that are part of multiple
 * sub-regions when calculating wall/particle interactions, to avoid
 * double-counting the interaction."
 *
 * Exact parity with native LAMMPS is in tests/oracle/w22rwall_*.in; the values
 * asserted here were measured with native LAMMPS (black box) for the same
 * inputs.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { thermo, error, files };
};

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Atom rows of a dump file written with write_dump (sorted by id). */
const parseDump = (text: string) => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const cols = lines[k].split(/\s+/).slice(2);
  return lines.slice(k + 1).map((l) => {
    const w = l.trim().split(/\s+/).map(Number);
    const a: Record<string, number> = {};
    cols.forEach((c, i) => { a[c] = w[i]; });
    return a;
  });
};

const HEAD = `units lj
atom_style atomic
boundary f f f
region box block -10 10 -10 10 -10 10 units box
create_box 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
`;

describe('fix wall/region on compound regions', () => {
  it('intersect: keeps a face only when its contact point is in every sub-region', async () => {
    // A = [0,2]^3, B = [1,3]x[0,2]x[0,2]; particle (1.9,1.9,1.9), harmonic rc 20.
    // A xlo (0,..) and B xhi (3,..) lie outside the intersection and are dropped;
    // the shared ylo/yhi/zlo/zhi faces are counted once per sub-region.
    const r = await runScript(`${HEAD}create_atoms 1 single 1.9 1.9 1.9 units box
region A block 0 2 0 2 0 2 side in units box
region B block 1 3 0 2 0 2 side in units box
region U intersect 2 A B
fix w all wall/region U harmonic 1.0 1.0 20.0
fix_modify w energy yes
thermo_style custom step pe f_w f_w[1] f_w[2] f_w[3]
thermo_modify format float %.15g
run 0`);
    expect(r.error).toBeNull();
    const row = r.thermo[0];
    expect(row['pe']).toBeCloseTo(3655.3, 9);
    expect(row['f_w[1]']).toBeCloseTo(1.6, 9);
    expect(row['f_w[2]']).toBeCloseTo(7.2, 9);
    expect(row['f_w[3]']).toBeCloseTo(7.2, 9);
  });

  it('union: drops a face whose contact point lies inside another sub-region', async () => {
    // A = [0,2]^3, B = [1,3]x[1.5,3.5]x[1.5,3.5] (no coincident faces);
    // particle (1.5,1.8,1.8) keeps A xlo/ylo/zlo and B xhi/yhi/zhi only.
    const r = await runScript(`${HEAD}create_atoms 1 single 1.5 1.8 1.8 units box
region A block 0 2 0 2 0 2 side in units box
region B block 1 3 1.5 3.5 1.5 3.5 side in units box
region U union 2 A B
fix w all wall/region U harmonic 1.0 1.0 20.0
fix_modify w energy yes
thermo_style custom step pe f_w f_w[1] f_w[2] f_w[3]
thermo_modify format float %.15g
run 0`);
    expect(r.error).toBeNull();
    const row = r.thermo[0];
    expect(row['pe']).toBeCloseTo(2016.76, 9);
    expect(row['f_w[1]']).toBeCloseTo(0.0, 9);
    expect(row['f_w[2]']).toBeCloseTo(0.2, 9);
    expect(row['f_w[3]']).toBeCloseTo(0.2, 9);
  });

  it('a side-out union/intersect region is a StyleError naming the style', async () => {
    const r = await runScript(`${HEAD}create_atoms 1 single 5 5 5 units box
region A block 0 2 0 2 0 2 side in units box
region B block 1 3 0 2 0 2 side in units box
region U union 2 A B side out
fix w all wall/region U harmonic 1.0 1.0 20.0
run 0`);
    expect(message(r.error)).toMatch(/side-out union region is not supported/);
  });
});

describe('fix wall/region on dynamic regions', () => {
  it('move: the region is displaced before the particle test', async () => {
    // region [0,2]^3 moved +0.5 in x; particle (0.6,1,1) is 0.1 from the moved xlo face.
    const r = await runScript(`${HEAD}create_atoms 1 single 0.6 1.0 1.0 units box
variable dx equal 0.5
region A block 0 2 0 2 0 2 side in units box move v_dx NULL NULL
fix w all wall/region A harmonic 1.0 1.0 0.5
fix_modify w energy yes
thermo_style custom step pe f_w f_w[1] f_w[2] f_w[3]
thermo_modify format float %.15g
run 0`);
    expect(r.error).toBeNull();
    const row = r.thermo[0];
    expect(row['pe']).toBeCloseTo(0.16, 9);
    expect(row['f_w[1]']).toBeCloseTo(-0.8, 9);
    expect(row['f_w[2]']).toBeCloseTo(0.0, 9);
    expect(row['f_w[3]']).toBeCloseTo(0.0, 9);
  });

  it('rotate: the sphere surface is rotated about the axis through P', async () => {
    // sphere at (5,0,0) r=1 rotated +90 deg about z -> centre (0,5,0);
    // particle (0,4.3,0) is 0.3 from the inner surface.
    const r = await runScript(`${HEAD}create_atoms 1 single 0.0 4.3 0.0 units box
variable th equal 1.5707963267948966
region S sphere 5.0 0.0 0.0 1.0 side in units box rotate v_th 0.0 0.0 0.0 0.0 0.0 1.0
fix w all wall/region S harmonic 1.0 1.0 0.5
fix_modify w energy yes
thermo_style custom step pe f_w f_w[1] f_w[2] f_w[3]
thermo_modify format float %.15g
run 0`);
    expect(r.error).toBeNull();
    const row = r.thermo[0];
    expect(row['pe']).toBeCloseTo(0.04, 9);
    expect(row['f_w[1]']).toBeCloseTo(0.0, 9);
    expect(row['f_w[2]']).toBeCloseTo(-0.4, 9);
    expect(row['f_w[3]']).toBeCloseTo(0.0, 9);
  });
});

describe('fix wall/gran/region on compound and dynamic regions', () => {
  it('union: the per-atom force sums the kept sub-region faces', async () => {
    const r = await runScript(`units lj
atom_style sphere
boundary f f f
region box block -10 10 -10 10 -10 10 units box
create_box 1 box
create_atoms 1 single 0.1 1.8 1.8 units box
set atom 1 diameter 1.0
set atom 1 density 1.0
region A block 0 2 0 2 0 2 side in units box
region B block 1 3 1.5 3.5 1.5 3.5 side in units box
region U union 2 A B
fix w all wall/gran/region granular hooke 1000.0 0.0 damping velocity tangential linear_nohistory 0.0 0.0 region U
run 0
write_dump all custom w.dump id fx fy fz modify format float %.15g`);
    expect(r.error).toBeNull();
    const a = parseDump(r.files.get('w.dump')!)[0];
    // xlo face d=0.1 (+x), yhi d=0.2 (-y), zhi d=0.2 (-z): F = kn (R - d)
    expect(a.fx).toBeCloseTo(400, 9);
    expect(a.fy).toBeCloseTo(-300, 9);
    expect(a.fz).toBeCloseTo(-300, 9);
  });

  it('accepts a dynamic region (move) and applies the face force', async () => {
    const r = await runScript(`units lj
atom_style sphere
boundary f f f
region box block -10 10 -10 10 -10 10 units box
create_box 1 box
create_atoms 1 single 0.1 1.0 1.0 units box
set atom 1 diameter 1.0
set atom 1 density 1.0
variable dx equal 0.05*time
region A block 0 2 0 2 0 2 side in units box move v_dx NULL NULL
fix w all wall/gran/region granular hooke 1000.0 0.0 damping velocity tangential linear_nohistory 0.0 0.0 region A
run 0
write_dump all custom w.dump id fx fy fz modify format float %.15g`);
    expect(r.error).toBeNull();
    const a = parseDump(r.files.get('w.dump')!)[0];
    expect(a.fx).toBeCloseTo(400, 9);
    expect(a.fy).toBeCloseTo(0, 9);
    expect(a.fz).toBeCloseTo(0, 9);
  });
});
