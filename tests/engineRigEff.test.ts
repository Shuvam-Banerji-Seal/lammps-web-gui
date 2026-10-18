import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { quatToMatNative, transposeMatvecNative, extractKe, extractErotational, type RigidBodyView } from '../src/engine-gpl/rigeff/rigid_bodies';
import { multiplyShapeShape } from '../src/engine-gpl/rigeff/compute_temp_deform_eff';
import { COMPUTE_STYLES, FIX_STYLES, ensureStyles } from '../src/engine/styles';
import { LAZY_GROUPS } from '../src/engine/lazyStyles';

/* Unit tests of the rigid-body, EFF, DPD-REACT and MACHDYN ports (src/engine-gpl/rigeff/): the pure helpers against
 * their closed forms, and the refusals and error messages of the native commands (the native text is quoted where the
 * source has one). Parity with native LAMMPS is in the w230re_* oracle cases. */

const run = async (script: string): Promise<string[]> => {
  const errors: string[] = [];
  const s = new Session({
    emit: (e: { kind: string; message?: string }) => { if (e.kind === 'error') errors.push(e.message ?? ''); },
    writeFile: () => {},
  } as never);
  try {
    await s.execute(script);
  } catch (e) {
    errors.push(e instanceof Error ? e.message : String(e));
  }
  return errors;
};

const BOX_LJ = `units lj
atom_style atomic
lattice sc 0.5
region b block 0 3 0 3 0 3
create_box 1 b
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
`;

const BOX_ELECTRON = `units electron
atom_style electron
lattice sc 1.0
region b block -10 10 -10 10 -10 10 units box
create_box 3 b
create_atoms 1 box
mass 1 28.0
mass 2 1.0
mass 3 1.0
pair_style eff/cut 1000.0
pair_coeff * *
`;

describe('rigid-body helpers (rigid_bodies.ts)', () => {
  it('quat_to_mat of the identity quaternion is the identity matrix', () => {
    expect(quatToMatNative([1, 0, 0, 0])).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  });

  it('a quaternion rotation by 90 degrees about z maps x to y (rot^T applied to the space-frame vector)', () => {
    const c = Math.SQRT1_2;
    const rot = quatToMatNative([c, 0, 0, c]);
    const b = transposeMatvecNative(rot, [1, 0, 0]);
    expect(b[0]).toBeCloseTo(0, 12);
    expect(b[1]).toBeCloseTo(-1, 12);
    expect(b[2]).toBeCloseTo(0, 12);
  });

  const body = (over: Partial<RigidBodyView>): RigidBodyView => ({
    atoms: [1], mass: 2, xcm: [0, 0, 0], vcm: [1, 2, 2], fcm: [0, 0, 0], torque: [0, 0, 0], image: [0, 0, 0],
    inertia: [1, 2, 4], quat: [1, 0, 0, 0], angmom: [0, 0, 0], omega: [0, 0, 0], ...over,
  });

  it('extract_ke is 0.5 M |vcm|^2 summed over bodies', () => {
    expect(extractKe([body({})])).toBeCloseTo(0.5 * 2 * 9, 12);
  });

  it('extract_erotational is 0.5 sum I_k w_k^2 with w = L / I in the body frame', () => {
    // L = (2, 0, 0) with the identity orientation: w_x = 2 / I_x = 2, so 0.5 * I_x * 4 = 2
    expect(extractErotational([body({ angmom: [2, 0, 0] })])).toBeCloseTo(2, 12);
    expect(extractErotational([body({ inertia: [0, 2, 4], angmom: [2, 0, 0] })])).toBeCloseTo(0, 12);
  });
});

describe('shape multiplication of fix deform rates (compute_temp_deform_eff.ts)', () => {
  it('multiplies two upper-triangular Voigt shapes as math_extra.h multiply_shape_shape', () => {
    const one = [2, 3, 5, 7, 11, 13];
    const two = [1, 0.5, 0.25, 2, 3, 4];
    const ans = multiplyShapeShape(one, two);
    expect(ans[0]).toBe(2);
    expect(ans[1]).toBe(1.5);
    expect(ans[2]).toBe(1.25);
    expect(ans[3]).toBe(3 * 2 + 7 * 0.25);
    expect(ans[4]).toBe(2 * 3 + 13 * 2 + 11 * 0.25);
    expect(ans[5]).toBe(2 * 4 + 13 * 0.5);
  });
});

describe('style names of the rigeff group (src/engine-gpl/registry/rigeff.ts)', () => {
  it('is listed in the lazy group with every style the registry module exports', async () => {
    const g = LAZY_GROUPS.find((x) => x.id === 'gpl_rigeff');
    expect(g).toBeDefined();
    expect([...(g!.names.COMPUTES ?? [])].sort()).toEqual(['erotate/rigid', 'ke/rigid', 'rigid/local', 'smd/hourglass/error', 'smd/tlsph/defgrad', 'smd/triangle/vertices', 'temp/deform/eff'].sort());
    expect([...(g!.names.FIXES ?? [])].sort()).toEqual(['eos/table', 'langevin/eff', 'nvt/sllod/eff', 'smd/move_tri_surf'].sort());
    await ensureStyles(['compute', 'erotate/rigid', 'fix', 'langevin/eff']);
    expect('erotate/rigid' in COMPUTE_STYLES).toBe(true);
    expect('langevin/eff' in FIX_STYLES).toBe(true);
  });
});

describe('native error messages and refusals', () => {
  it('compute erotate/rigid takes exactly one fix ID', async () => {
    expect((await run(`${BOX_LJ}compute c all erotate/rigid\nthermo_style custom step c_c\nrun 0\n`)).join('\n')).toContain('Illegal compute erotate/rigid command');
  });

  it('compute erotate/rigid and ke/rigid refuse a non-rigid fix (compute_erotate_rigid.cpp init)', async () => {
    const errs = await run(`${BOX_LJ}fix f all nve\ncompute c all erotate/rigid f\nthermo_style custom step c_c\nrun 0\n`);
    expect(errs.join('\n')).toContain('Compute erotate/rigid with non-rigid fix f');
    const errs2 = await run(`${BOX_LJ}fix f all nve\ncompute c all ke/rigid f\nthermo_style custom step c_c\nrun 0\n`);
    expect(errs2.join('\n')).toContain('Compute ke/rigid with non-rigid fix f');
  });

  it('compute rigid/local needs a fix rigid/small (compute_rigid_local.cpp init)', async () => {
    const errs = await run(`${BOX_LJ}fix f all rigid molecule\ncompute c all rigid/local f x\nthermo_style custom step c_c\nrun 0\n`);
    expect(errs.join('\n')).toContain('does not point to fix rigid/small');
  });

  it('compute rigid/local rejects unknown keywords', async () => {
    const errs = await run(`${BOX_LJ}fix f all rigid/small molecule\ncompute c all rigid/local f bogus\nthermo_style custom step c_c\nrun 0\n`);
    expect(errs.join('\n')).toContain('Invalid keyword in compute rigid/local command');
  });

  it('fix langevin/eff refuses the zero keyword with its native reason', async () => {
    const errs = await run(`${BOX_ELECTRON}fix f all langevin/eff 300.0 300.0 20.0 4321 zero yes\n`);
    expect(errs.join('\n')).toContain('zero yes is not supported');
  });

  it('fix langevin/eff refuses the omega and angmom keywords', async () => {
    const e1 = await run(`${BOX_ELECTRON}fix f all langevin/eff 300.0 300.0 20.0 4321 omega yes\n`);
    expect(e1.join('\n')).toContain('omega is not supported');
    const e2 = await run(`${BOX_ELECTRON}fix f all langevin/eff 300.0 300.0 20.0 4321 angmom 1.0\n`);
    expect(e2.join('\n')).toContain('angmom is not supported');
  });

  it('fix langevin/eff requires atom_style electron', async () => {
    const errs = await run(`${BOX_LJ}fix f all langevin/eff 300.0 300.0 20.0 4321\n`);
    expect(errs.join('\n')).toContain('Fix langevin/eff requires atom style electron');
  });

  it('fix nvt/sllod/eff needs a temperature and no pressure control', async () => {
    const errs = await run(`${BOX_ELECTRON}fix f all nvt/sllod/eff iso 1.0 1.0 1.0\n`);
    expect(errs.join('\n')).toContain('Temperature control must be used with fix nvt/sllod/eff');
  });

  it('fix nvt/sllod/eff rejects the legacy integrator with peculiar yes', async () => {
    const errs = await run(`${BOX_ELECTRON}fix f all nvt/sllod/eff temp 300.0 300.0 10.0 integrator legacy peculiar yes\n`);
    expect(errs.join('\n')).toContain('legacy integrator is not compatible with peculiar=yes');
  });

  it('fix eos/table takes exactly the linear style, table, N and keyword', async () => {
    const errs = await run(`${BOX_LJ}fix f all eos/table linear t.txt\n`);
    expect(errs.join('\n')).toContain('Illegal fix eos/table command');
  });

  it('fix smd/move_tri_surf needs atom_style smd', async () => {
    const errs = await run(`${BOX_LJ}fix f all smd/move_tri_surf *LINEAR 1 0 0\n`);
    expect(errs.join('\n')).toContain('fix fix smd/move_tri_surf command requires atom_style smd');
  });

  it('compute smd/hourglass/error needs atom_style smd', async () => {
    const errs = await run(`${BOX_LJ}compute h all smd/hourglass/error\nthermo_style custom step c_h\nrun 0\n`);
    expect(errs.join('\n')).toContain('compute smd/hourglass/error command requires atom_style with hourglass_error');
  });
});
