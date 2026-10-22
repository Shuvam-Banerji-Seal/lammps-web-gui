import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix adapt on hybrid sub-styles (pstyle "style:N"), the snap "scale" parameter, and the
 * newly adaptable pair parameters (docs.lammps.org/fix_adapt.html). Parity with native LAMMPS is
 * in tests/oracle/w43ad_*.in; these tests check the energy relations and the error messages.
 */

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const POT_FILES = ['Ni_Zuo_JPCA2020.snapcoeff', 'Ni_Zuo_JPCA2020.snapparam', 'Ni_Zuo_JPCA2020.quadratic.snapcoeff', 'Ni_Zuo_JPCA2020.quadratic.snapparam'];

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  for (const f of POT_FILES) session.addFile(f, readFileSync(join(POT, f), 'utf8'));
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { thermo, error, message: error instanceof Error ? error.message : String(error ?? '') };
};

const LJ = `
units           lj
atom_style      atomic
lattice         sc 0.8
region          b block 0 3 0 3 0 3
create_box      2 b
create_atoms    1 box
mass            * 1.0
set             group all type 1
group           odd id 1:27:2
set             group odd type 2
displace_atoms  all random 0.05 0.05 0.05 4711
thermo_style    custom step pe evdwl
thermo_modify   format float %.15g
thermo          1
`;

const SNAP = `
units           metal
atom_style      atomic
lattice         fcc 3.52
region          b block 0 2 0 2 0 2
create_box      2 b
create_atoms    1 box
mass            * 58.7
set             group all type 1
group           odd id 1:32:2
set             group odd type 2
displace_atoms  all random 0.05 0.05 0.05 12345
thermo_style    custom step pe evdwl press
thermo_modify   format float %.15g
thermo          1
`;

const evdwlOf = async (body: string) => {
  const r = await runScript(body + '\nrun 0\n');
  expect(r.error).toBeNull();
  return r.thermo[r.thermo.length - 1].evdwl;
};

describe('fix adapt on hybrid sub-styles', () => {
  it('snap:1 scale v: energy is v * E1 + E2 (sub-styles alone are the references)', async () => {
    const e1 = await evdwlOf(`${SNAP}
pair_style snap
pair_coeff * * Ni_Zuo_JPCA2020.snapcoeff Ni_Zuo_JPCA2020.snapparam Ni Ni
`);
    const e2 = await evdwlOf(`${SNAP}
pair_style snap
pair_coeff * * Ni_Zuo_JPCA2020.quadratic.snapcoeff Ni_Zuo_JPCA2020.quadratic.snapparam Ni Ni
`);
    const both = await evdwlOf(`${SNAP}
pair_style hybrid/overlay snap snap
pair_coeff * * snap 1 Ni_Zuo_JPCA2020.snapcoeff Ni_Zuo_JPCA2020.snapparam Ni Ni
pair_coeff * * snap 2 Ni_Zuo_JPCA2020.quadratic.snapcoeff Ni_Zuo_JPCA2020.quadratic.snapparam Ni Ni
`);
    expect(both).toBeCloseTo(e1 + e2, 9);
    const scaled = await evdwlOf(`${SNAP}
pair_style hybrid/overlay snap snap
pair_coeff * * snap 1 Ni_Zuo_JPCA2020.snapcoeff Ni_Zuo_JPCA2020.snapparam Ni Ni
pair_coeff * * snap 2 Ni_Zuo_JPCA2020.quadratic.snapcoeff Ni_Zuo_JPCA2020.quadratic.snapparam Ni Ni
variable v equal 0.25
fix 1 all adapt 0 pair snap:1 scale * * v_v
`);
    expect(scaled).toBeCloseTo(0.25 * e1 + e2, 9);
  });

  it('scale 1 2 has no effect (only the diagonal type entry of the central atom is used)', async () => {
    const base = `${SNAP}
pair_style snap
pair_coeff * * Ni_Zuo_JPCA2020.snapcoeff Ni_Zuo_JPCA2020.snapparam Ni Ni
`;
    const e0 = await evdwlOf(base);
    const e12 = await evdwlOf(`${base}variable v equal 0.5\nfix 1 all adapt 0 pair snap scale 1 2 v_v\n`);
    expect(e12).toBe(e0);
    const e11 = await evdwlOf(`${base}variable v equal 0.5\nfix 1 all adapt 0 pair snap scale 1 1 v_v\n`);
    expect(e11).not.toBe(e0);
  });

  it('pstyle:N selects the N-th instance of a repeated sub-style', async () => {
    const two = `${LJ}
pair_style hybrid/overlay lj/cut 2.5 lj/cut 2.0
pair_coeff * * lj/cut 1 1.0 1.0
pair_coeff * * lj/cut 2 0.5 1.2
variable e equal 2.0
`;
    const e0 = await evdwlOf(two);
    const s2 = await evdwlOf(`${two}fix 1 all adapt 0 pair lj/cut:2 epsilon * * v_e\n`);
    const s1 = await evdwlOf(`${two}fix 1 all adapt 0 pair lj/cut:1 epsilon * * v_e\n`);
    expect(s2).not.toBe(e0);
    expect(s1).not.toBe(e0);
    expect(s1).not.toBe(s2);
  });

  it('a repeated sub-style without N is an error that names style:N', async () => {
    const r = await runScript(`${LJ}
pair_style hybrid/overlay lj/cut 2.5 lj/cut 2.0
pair_coeff * * lj/cut 1 1.0 1.0
pair_coeff * * lj/cut 2 0.5 1.2
variable e equal 2.0
fix 1 all adapt 0 pair lj/cut epsilon * * v_e
run 0
`);
    expect(r.message).toContain('lj/cut:N');
  });

  it('an instance that does not exist is an error', async () => {
    const r = await runScript(`${LJ}
pair_style hybrid/overlay lj/cut 2.5 lj/cut 2.0
pair_coeff * * lj/cut 1 1.0 1.0
pair_coeff * * lj/cut 2 0.5 1.2
variable e equal 2.0
fix 1 all adapt 0 pair lj/cut:3 epsilon * * v_e
run 0
`);
    expect(r.message).toContain('lj/cut:3');
  });

  it('a sub-style N on a plain (non-hybrid) pair style is an error', async () => {
    const r = await runScript(`${LJ}
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
variable e equal 2.0
fix 1 all adapt 0 pair lj/cut:2 epsilon * * v_e
run 0
`);
    expect(r.message).toContain('lj/cut:2');
  });

  it('lj/expand delta is refused (its force cutoff is frozen natively)', async () => {
    const r = await runScript(`${LJ}
pair_style lj/expand 2.5
pair_coeff * * 1.0 1.0 0.1
variable d equal 0.2
fix 1 all adapt 0 pair lj/expand delta * * v_d
run 0
`);
    expect(r.message).toContain('delta');
    expect(r.message).toContain('not supported');
  });

  it('a pair style that is not adaptable is still refused', async () => {
    const r = await runScript(`${LJ}
pair_style coul/long 2.5
pair_coeff * *
variable s equal 0.5
fix 1 all adapt 0 pair coul/long scale * * v_s
run 0
`);
    expect(r.message).toContain('coul/long');
  });

  it('buck a and c adapt a type pair (energy changes, mixed pairs are not targets)', async () => {
    const body = `${LJ}
pair_style buck 2.5
pair_coeff 1 1 20.0 0.4 0.5
pair_coeff 2 2 10.0 0.35 0.3
pair_coeff 1 2 15.0 0.38 0.4
`;
    const e0 = await evdwlOf(body);
    const e1 = await evdwlOf(`${body}variable A equal 30.0\nfix 1 all adapt 0 pair buck a 1 1 v_A\n`);
    expect(e1).not.toBe(e0);
    const r = await runScript(`${LJ}
pair_style buck 2.5
pair_coeff 1 1 20.0 0.4 0.5
variable A equal 30.0
fix 1 all adapt 0 pair buck a 1 2 v_A
run 0
`);
    // Measured with native LAMMPS: the pair style's init (force->init, before the fixes' init) stops first, with
    // All pair coeffs are not set. Status:
    expect(r.message).toMatch(/all pair coeffs are not set/i);
  });
});
