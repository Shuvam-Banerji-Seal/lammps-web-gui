import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { PlaneRegion } from '../src/engine/region';

/*
 * The plane region as a wall — docs.lammps.org/region.html,
 * docs.lammps.org/fix_wall_region.html and docs.lammps.org/fix_wall_gran_region.html.
 *
 * "For style *plane*, a plane is defined which contain the point (px,py,pz) and
 * has a normal vector (nx,ny,nz).  The normal vector does not have to be of
 * unit length.  The "inside" of the plane is the half-space in the direction of
 * the normal vector" (region.html).
 *
 * "The distance between a particle and the region boundary is the distance to
 * the nearest point on the region surface.  The force the wall exerts on the
 * particle is along the direction between that point and the particle center,
 * which is the direction normal to the surface at that point."
 * (fix_wall_gran_region.html)
 *
 * "For a flat wall, delta = radius - r = overlap of particle with wall, m_eff =
 * mass of particle, and the effective radius of contact is just the radius of
 * the particle." (fix_wall_gran_region.html) — the plane's curvature is 0.
 *
 * Exact parity with native LAMMPS is in tests/oracle/w31plane_*.in; the values
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

const HEAD = (extra: string, atom: string) => `units lj
atom_style atomic
boundary f f f
region box block -5 5 -5 5 -5 5 units box
create_box 1 box
create_atoms 1 single ${atom} units box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
${extra}`;

describe('plane region primitiveContacts (docs.lammps.org/region.html)', () => {
  it('distance is the signed projection onto the unit normal; normal points into the interior', () => {
    const reg = new PlaneRegion('p', { variable: () => 0, region: () => undefined }, [1, 1, 1], [1, 1, 1]);
    const out: import('../src/engine/region').SurfaceContact[] = [];
    // (1.6,1.5,1.7): (0.6+0.5+0.7)/sqrt(3) = 1.8/sqrt(3)
    reg.contacts(1.6, 1.5, 1.7, out);
    expect(out).toHaveLength(1);
    expect(out[0].dist).toBeCloseTo(1.8 / Math.sqrt(3), 15);
    expect(Math.hypot(out[0].nx, out[0].ny, out[0].nz)).toBeCloseTo(1, 15);
    expect(out[0].nx).toBeCloseTo(1 / Math.sqrt(3), 15);
    // flat wall: radius of curvature 0
    expect(out[0].curvature).toBe(0);
  });

  it('a point on the wrong side of the normal gives no contact', () => {
    const reg = new PlaneRegion('p', { variable: () => 0, region: () => undefined }, [1, 1, 1], [1, 1, 1]);
    const out: import('../src/engine/region').SurfaceContact[] = [];
    reg.contacts(0.0, 0.0, 0.0, out);
    expect(out).toHaveLength(0);
  });
});

describe('fix wall/region on a plane (region.html + fix_wall_region.html)', () => {
  it('oblique plane, harmonic: energy and wall force measured with native LAMMPS', async () => {
    // Measured with native LAMMPS (black box): region plane 1 1 1 1 1 1 side in
    // units box, harmonic 1.0 0.0 2.5, one atom at (1.6,1.5,1.7):
    // r = 1.8/sqrt(3) = 1.0392304845413265, pe = 2.13384757729337 and the wall
    // force f_w = (-1.68675134594813, -1.68675134594813, -1.68675134594813)
    // (the reaction of the force 2(2.5-r) n_hat on the atom).
    const r = await runScript(`${HEAD('region pl plane 1.0 1.0 1.0 1.0 1.0 1.0 side in units box\nfix w all wall/region pl harmonic 1.0 0.0 2.5\nfix_modify w energy yes', '1.6 1.5 1.7')}
thermo_style custom step pe f_w f_w[1] f_w[2] f_w[3]
thermo_modify format float %.15g
run 0`);
    expect(r.error).toBeNull();
    const row = r.thermo[0];
    expect(row['pe']).toBeCloseTo(2.13384757729337, 9);
    expect(row['f_w[1]']).toBeCloseTo(-1.68675134594813, 9);
    expect(row['f_w[2]']).toBeCloseTo(-1.68675134594813, 9);
    expect(row['f_w[3]']).toBeCloseTo(-1.68675134594813, 9);
  });

  it('a non-unit normal is normalised: scaling (nx,ny,nz) leaves the energy unchanged', async () => {
    const one = await runScript(`${HEAD('region pl plane 1.0 1.0 1.0 1.0 1.0 1.0 side in units box\nfix w all wall/region pl harmonic 1.0 0.0 2.5\nfix_modify w energy yes', '1.6 1.5 1.7')}
thermo_style custom step pe
thermo_modify format float %.15g
run 0`);
    const scaled = await runScript(`${HEAD('region pl plane 1.0 1.0 1.0 3.0 3.0 3.0 side in units box\nfix w all wall/region pl harmonic 1.0 0.0 2.5\nfix_modify w energy yes', '1.6 1.5 1.7')}
thermo_style custom step pe
thermo_modify format float %.15g
run 0`);
    expect(scaled.error).toBeNull();
    expect(scaled.thermo[0]['pe']).toBeCloseTo(one.thermo[0]['pe'], 12);
  });

  it('a side-out plane region is a StyleError naming it', async () => {
    const r = await runScript(`${HEAD('region pl plane 1.0 1.0 1.0 1.0 1.0 1.0 side out units box\nfix w all wall/region pl harmonic 1.0 0.0 2.5', '1.6 1.5 1.7')}
run 0`);
    expect(message(r.error)).toMatch(/side-out plane region is not supported/);
  });
});

describe('fix wall/gran/region on a plane (fix_wall_gran_region.html)', () => {
  const GRAN = (atom: string, plane: string) => `units lj
atom_style sphere
boundary f f f
region box block -5 5 -5 5 -5 5 units box
create_box 1 box
create_atoms 1 single ${atom} units box
set atom 1 diameter 1.0
set atom 1 density 1.0
pair_style none
region pl plane ${plane} side in units box
fix w all wall/gran/region hooke 1000.0 200.0 0.0 0.0 0.5 0 region pl
run 0
write_dump all custom g.dump id x y z fx fy fz modify format float %.15g`;

  it('flat wall overlap: F = Kn (R - r) along the normal', async () => {
    // Measured with native LAMMPS (black box): plane 0 0 1 0 0 1 side in, sphere
    // radius 0.5 at (0,0,1.4) -> r = 0.4, delta = 0.1, f = (0,0,100).
    const r = await runScript(GRAN('0.0 0.0 1.4', '0.0 0.0 1.0 0.0 0.0 1.0'));
    expect(r.error).toBeNull();
    const a = parseDump(r.files.get('g.dump')!)[0];
    expect(a.fx).toBeCloseTo(0, 12);
    expect(a.fy).toBeCloseTo(0, 12);
    expect(a.fz).toBeCloseTo(100, 12);
  });

  it('oblique plane: the force is along n/|n|, effective radius is the particle radius', async () => {
    // Measured with native LAMMPS (black box): plane 0 0 1 1 1 1 side in, sphere
    // radius 0.5 at (0.2,0.1,1.5): r = 0.8/sqrt(3) = 0.46188021535170065,
    // delta = 0.038119784648299, F = 1000 delta = 38.119784648299 along n_hat,
    // so each component is 38.119784648299/sqrt(3) = 22.0084679281462.
    const r = await runScript(GRAN('0.2 0.1 1.5', '0.0 0.0 1.0 1.0 1.0 1.0'));
    expect(r.error).toBeNull();
    const a = parseDump(r.files.get('g.dump')!)[0];
    expect(a.fx).toBeCloseTo(22.0084679281462, 9);
    expect(a.fy).toBeCloseTo(22.0084679281462, 9);
    expect(a.fz).toBeCloseTo(22.0084679281462, 9);
  });
});
