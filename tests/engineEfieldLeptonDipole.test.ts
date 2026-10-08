import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix efield/lepton point dipoles (docs.lammps.org/fix_efield_lepton.html):
 * F = (p . grad) E by the central difference |p|/(2h)[E(x + h p) - E(x - h p)],
 * torque T = p x E (analytic E) and energy -p . E, all with fix efield's qe2f.
 * The expected numbers are native LAMMPS 2 Sep 2026 as a black box; the two
 * oracle cases w23eflep_dipole / w23eflep_uniform compare full trajectories.
 */

/** Runs an input in a fresh session; returns the error message or '' and the thermo rows. */
const run = async (text: string): Promise<{ error: string; rows: Array<Record<string, number>> }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let error = '';
  try {
    await session.execute(text);
  } catch (e) {
    error = String(e);
  }
  const err = events.find((x) => x.kind === 'error');
  if (err && 'message' in err) error = String(err.message);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as unknown as Record<string, number>);
  return { error, rows };
};

/** A one-atom hybrid sphere dipole input; `expr` is the field, `step` an optional 'step h'. */
const dipoleInput = (expr: string, dipole: string, pos: string, step = ''): string => `
units lj
atom_style hybrid sphere dipole
region b block -10 10 -10 10 -10 10
create_box 1 b
create_atoms 1 single ${pos}
mass 1 1.0
set type 1 diameter 1.0 density 1.0
set atom 1 dipole ${dipole}
pair_style zero 5.0
pair_coeff * *
fix ex all efield/lepton "${expr}" ${step}
fix_modify ex energy yes
fix 1 all nve/sphere update dipole
compute c all property/atom fx fy fz tqx tqy tqz
compute s all reduce sum c_c[1] c_c[2] c_c[3] c_c[4] c_c[5] c_c[6]
thermo_style custom step pe f_ex c_s[1] c_s[2] c_s[3] c_s[4] c_s[5] c_s[6]
thermo_modify format float %.15g norm no
run 0
`;

describe('fix efield/lepton point dipoles', () => {
  it('uniform field, no step: torque p x E and energy -p . E (measured: tqz 0.5, pe -0.5, f_ex -0.5)', async () => {
    // E = (1,1,0), p = (0.5,0,0): T = (0,0,0.5), U = -0.5
    const r = await run(dipoleInput('-(x+y)', '0.5 0.0 0.0', '1.5 0.0 0.0'));
    expect(r.error).toBe('');
    expect(r.rows[0]['c_s[6]']).toBeCloseTo(0.5, 12);
    expect(r.rows[0]['c_s[1]']).toBeCloseTo(0, 12);
    expect(r.rows[0].pe).toBeCloseTo(-0.5, 12);
    expect(r.rows[0].f_ex).toBeCloseTo(-0.5, 12);
  });

  it('central-difference force on a cubic field (measured: fx 6.76 with step 0.1, not the analytic 6.75)', async () => {
    // 3*x^2 + h^2 at x = 1.5, h = 0.1
    const r = await run(dipoleInput('-0.25*x^4', '1.0 0.0 0.0', '1.5 0.0 0.0', 'step 0.1'));
    expect(r.error).toBe('');
    expect(r.rows[0]['c_s[1]']).toBeCloseTo(6.76, 9);
    expect(r.rows[0].pe).toBeCloseTo(-3.375, 12);
  });

  it('torque uses the analytic E, not a central difference (measured: tqz -3.375 with step 0.1)', async () => {
    // p x E with p = (0,1,0), E = (x^3,0,0): tqz = -x^3 = -3.375
    const r = await run(dipoleInput('-0.25*x^4', '0.0 1.0 0.0', '1.5 0.0 0.0', 'step 0.1'));
    expect(r.error).toBe('');
    expect(r.rows[0]['c_s[6]']).toBeCloseTo(-3.375, 12);
    expect(r.rows[0]['c_s[1]']).toBeCloseTo(0, 12);
  });

  it('step is required for dipoles in a non-uniform field, but not for a uniform one', async () => {
    const nonUniform = await run(dipoleInput('-x^2', '1.0 0.0 0.0', '1.5 0.0 0.0'));
    expect(nonUniform.error).toMatch(/step/);
    const uniform = await run(dipoleInput('-(x+y)', '1.0 0.0 0.0', '1.5 0.0 0.0'));
    expect(uniform.error).toBe('');
  });

  it('atom_style dipole without finite-size particles is rejected (native: Dipoles must be finite-sized to rotate)', async () => {
    const r = await run(`
units lj
atom_style dipole
region b block -10 10 -10 10 -10 10
create_box 1 b
create_atoms 1 single 1.5 0.0 0.0
mass 1 1.0
set atom 1 dipole 0.5 0.0 0.0
pair_style zero 5.0
pair_coeff * *
fix ex all efield/lepton "-x"
fix_modify ex energy yes
run 0
`);
    expect(r.error).toMatch(/dipole/);
  });

  it('virial of the added forces is f (x) r in unwrapped coordinates (measured: pxx, pxy for a charge)', async () => {
    const r = await run(`
units lj
atom_style charge
region b block -10 10 -10 10 -10 10
create_box 1 b
create_atoms 1 single 1.5 2.0 3.0
mass 1 1.0
set atom 1 charge 1.0
pair_style zero 5.0
pair_coeff * *
fix ex all efield/lepton "-(x+y)"
fix_modify ex energy yes virial yes
fix 1 all nve
thermo_style custom step pe press pxx pyy pzz pxy pxz pyz
thermo_modify format float %.15g
run 0
`);
    expect(r.error).toBe('');
    const V = 20 * 20 * 20;
    // F = (1,1,0) at (1.5,2,3): pxx = f_x x / V, pxy = f_x y / V, pxz = f_x z / V, pyz = f_y z / V
    expect(r.rows[0].pxx).toBeCloseTo(1.5 / V, 15);
    expect(r.rows[0].pyy).toBeCloseTo(2 / V, 15);
    expect(r.rows[0].pxy).toBeCloseTo(2 / V, 15);
    expect(r.rows[0].pxz).toBeCloseTo(3 / V, 15);
    expect(r.rows[0].pyz).toBeCloseTo(3 / V, 15);
  });

  it('virial uses the unwrapped position (measured: a charge with image -1 at xu = -0.5 gives pxx = -0.5/V)', async () => {
    const r = await run(`
units lj
atom_style charge
region b block 0 10 0 10 0 10
create_box 1 b
create_atoms 1 single 0.5 5.0 5.0
mass 1 1.0
set atom 1 charge 1.0
displace_atoms all move -1.0 0.0 0.0 units box
pair_style zero 5.0
pair_coeff * *
fix ex all efield/lepton "-x"
fix_modify ex energy yes virial yes
fix 1 all nve
thermo_style custom step pe press pxx pyy pzz pxy pxz pyz
thermo_modify format float %.15g
run 0
`);
    expect(r.error).toBe('');
    const V = 10 * 10 * 10;
    expect(r.rows[0].pxx).toBeCloseTo(-0.5 / V, 15);
    expect(r.rows[0].pxy).toBeCloseTo(5 / V, 15);
    expect(r.rows[0].pxz).toBeCloseTo(5 / V, 15);
    expect(r.rows[0].pyz).toBeCloseTo(0, 15);
  });

  it('the oracle inputs are present', () => {
    for (const f of ['w23eflep_dipole.in', 'w23eflep_uniform.in']) {
      expect(readFileSync(join(__dirname, 'oracle', f), 'utf8').length).toBeGreaterThan(0);
    }
  });
});
