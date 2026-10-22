import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { pinvSym3 } from '../src/engine/fix/addtorque_group';

/*
 * Unit tests for the batch-48 fixes (addtorque/group, ti/spring, tmd, rhok): parsing and refusals, and a few
 * values measured with native LAMMPS (black box). The full parity checks are the tests/oracle/w48fb_*.in cases
 * (tests/engineOracle.test.ts -t w48fb). Literal numbers below are native outputs of the same inputs.
 */

const run = async (script: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const written = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => written.set(n, (ap ? written.get(n) ?? '' : '') + t),
  }, undefined, 0, 1);
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  await session.execute(script);
  const rows: ThermoRow[] = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { rows, written };
};

const LJ_HEAD = `units lj
atom_style atomic
atom_modify map array
region box block -6 6 -6 6 -6 6
create_box 1 box
`;

describe('pinvSym3', () => {
  it('inverts a symmetric positive matrix and drops the null direction of a singular one', () => {
    const a = [2, 1, 0, 1, 3, 0.5, 0, 0.5, 1];
    const p = pinvSym3(a);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[3 * i + k] * p[3 * k + j];
      expect(s).toBeCloseTo(i === j ? 1 : 0, 12);
    }
    // a rank-2 inertia tensor of two atoms on the x axis: the x direction is dropped
    const sing = [0, 0, 0, 0, 0.72, 0, 0, 0, 0.72];
    const q = pinvSym3(sing);
    expect(q[0]).toBe(0);
    expect(q[4]).toBeCloseTo(1 / 0.72, 12);
    expect(q[8]).toBeCloseTo(1 / 0.72, 12);
  });
});

describe('fix addtorque/group', () => {
  it('refuses the wrong argument count', async () => {
    await expect(run(`${LJ_HEAD}create_atoms 1 single 0 0 0 units box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5\nfix 4 all addtorque/group 1.0 2.0\nrun 0\n`)).rejects.toThrow(/addtorque\/group/);
  });

  it('gives the native scalar (omega . I . omega) for a five-atom group, printed per atom by thermo norm', async () => {
    const script = `${LJ_HEAD}create_atoms 1 single 0.0 0.0 0.0 units box
create_atoms 1 single 1.2 0.0 0.0 units box
create_atoms 1 single 1.5 1.1 0.2 units box
create_atoms 1 single -0.4 1.0 0.3 units box
create_atoms 1 single 0.3 -1.1 -0.5 units box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 0.2 1.0 2.5
set atom 1 vx 0.3 vy -0.2 vz 0.5
set atom 2 vx -0.1 vy 0.4 vz 0.2
set atom 3 vx 0.2 vy 0.1 vz -0.3
set atom 4 vx 0.5 vy 0.0 vz 0.1
set atom 5 vx -0.4 vy -0.3 vz 0.2
fix 1 all nve
fix 4 all addtorque/group 0.0 0.0 2.0
thermo_style custom step pe ke f_4 f_4[1] f_4[2] f_4[3]
thermo_modify format float %.15g norm yes
thermo 1
run 0
`;
    const { rows } = await run(script);
    // thermo_modify norm yes divides the extensive fix scalar by the 5 atoms: native printed 0.084273582969688
    expect(rows[0].f_4 as number).toBeCloseTo(0.084273582969688, 12);
    expect(rows[0]['f_4[3]'] as number).toBeCloseTo(0, 12);
  });
});

describe('fix ti/spring', () => {
  it('refuses a zero switching length, a bad function and an unknown keyword', async () => {
    const head = `${LJ_HEAD}create_atoms 1 single 0 0 0 units box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5\nfix 1 all nve\n`;
    await expect(run(`${head}fix f all ti/spring 3.0 0 3\nrun 0\n`)).rejects.toThrow(/t_s must be a positive integer/);
    await expect(run(`${head}fix f all ti/spring 3.0 4 3 function 3\nrun 0\n`)).rejects.toThrow(/function must be 1 or 2/);
    await expect(run(`${head}fix f all ti/spring 3.0 4 3 bogus 1\nrun 0\n`)).rejects.toThrow(/unknown keyword 'bogus'/);
  });

  it('follows the native lambda schedule (function 1: t_s 4, t_eq 3, defined at step 0)', async () => {
    const script = `units lj
atom_style atomic
lattice fcc 0.9
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
velocity all create 0.5 4321 loop all
fix 1 all nve
fix f2 all ti/spring 3.0 4 3 function 1
thermo_style custom step f_f2 f_f2[1] f_f2[2] pe
thermo_modify format float %.15g norm no
thermo 1
run 14
`;
    const { rows } = await run(script);
    const at = (s: number) => rows.find((r) => r.step === s)!;
    // lambda: 0 through step 3, 0.25 at step 4, 1 at step 7, 0.75 at step 11, 0 from step 14 (native)
    expect(at(3)['f_f2[1]'] as number).toBeCloseTo(0, 12);
    expect(at(4)['f_f2[1]'] as number).toBeCloseTo(0.25, 12);
    expect(at(7)['f_f2[1]'] as number).toBeCloseTo(1, 12);
    expect(at(11)['f_f2[1]'] as number).toBeCloseTo(0.75, 12);
    expect(at(14)['f_f2[1]'] as number).toBeCloseTo(0, 12);
    // dlambda/ds = +1/t_s before the switch down (step 10), -1/t_s from it on
    expect(at(9)['f_f2[2]'] as number).toBeCloseTo(0.25, 12);
    expect(at(10)['f_f2[2]'] as number).toBeCloseTo(-0.25, 12);
  });
});

describe('fix tmd', () => {
  const eight = (extra: string) => `${LJ_HEAD}create_atoms 1 single 0.964426 0.961779 0.980848 units box
create_atoms 1 single 1.031613 0.968073 3.008160 units box
create_atoms 1 single 1.013891 2.987240 1.004774 units box
create_atoms 1 single 0.956279 2.955960 2.970596 units box
create_atoms 1 single 3.018040 0.992759 0.981415 units box
create_atoms 1 single 3.008556 0.995318 2.979977 units box
create_atoms 1 single 3.029438 3.019899 0.974410 units box
create_atoms 1 single 3.007442 3.002520 3.037514 units box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
velocity all create 0.1 4928 loop all
timestep 0.005
fix 1 all nve
${extra}
run 10
`;
  const TARGET = `# target
1 1.347150 0.895255 0.730187
2 1.271731 1.010765 2.673138
3 1.267400 3.002231 0.607499
4 1.380094 2.870957 2.618143
5 3.377356 1.098056 0.624760
6 3.316972 1.038230 2.789542
7 3.423131 2.969004 0.795251
8 3.263975 3.107541 2.657922
`;

  it('refuses a target file that misses a group atom, image flags without a box, and N > 0 without file2', async () => {
    const short = TARGET.split('\n').filter((l) => !l.startsWith('8 ')).join('\n');
    await expect(run(eight('fix 2 all tmd 0.5 t.txt 1 f.log'), { 't.txt': short })).rejects.toThrow(/atom 8 of the group is not listed/);
    const withImages = TARGET.split('\n').map((l) => (l.startsWith('#') || !l ? l : `${l} 0 0 0`)).join('\n');
    await expect(run(eight('fix 2 all tmd 0.5 t.txt 1 f.log'), { 't.txt': withImages })).rejects.toThrow(/image flags but no box lines/);
    await expect(run(eight('fix 2 all tmd 0.5 t.txt 1'), { 't.txt': TARGET })).rejects.toThrow(/file2 is needed/);
  });

  it('writes the native first row of the statistics file (rho_target, rho_old, gammas, lambda, work_lambda)', async () => {
    const { written } = await run(eight('fix 2 all tmd 0.5 t.txt 1 f.log'), { 't.txt': TARGET });
    const lines = (written.get('f.log') ?? '').trim().split('\n');
    expect(lines[0]).toBe('# Step rho_target rho_old gamma_back gamma_forward lambda work_lambda work_analytical');
    const w = lines[1].split(' ').map(Number);
    // native (black box), 8 atoms, zero pair forces, run 10 towards rho_final 0.5: first row of the statistics file
    const want = [0.46853344477317316, 0.46503716085908126, -1.1778821277662881e-5, 0.007506623710580587, -1.7528286736738823, -0.0061283866959250385];
    want.forEach((v, k) => expect(Math.abs(w[k + 1] - v) / Math.abs(v)).toBeLessThan(1e-8));
  });
});

describe('fix rhok', () => {
  it('refuses the wrong argument count', async () => {
    await expect(run(`${LJ_HEAD}create_atoms 1 single 0 0 0 units box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5\nfix B all rhok 1 0 0 2.0\nrun 0\n`)).rejects.toThrow(/Illegal fix rhoKUmbrella command/);
  });

  it('matches the native U, Re, Im and |rho| for eight atoms (k = 2 pi / L_x, K = 2, a = 1)', async () => {
    const atoms = [[0.3, 0.2, 0.1], [1.4, 0.1, 0.6], [0.2, 1.7, 0.9], [1.1, 1.2, 1.4], [2.3, 0.4, 1.9], [2.9, 1.5, 0.2], [2.1, 2.6, 1.1], [0.7, 2.9, 2.4]];
    const script = `units lj
atom_style atomic
atom_modify map array
boundary p p p
region box block 0 3 0 3 0 3
create_box 1 box
${atoms.map((p) => `create_atoms 1 single ${p.join(' ')} units box`).join('\n')}
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
fix 1 all nve
fix B all rhok 1 0 0 2.0 1.0
thermo_style custom step pe f_B f_B[1] f_B[2] f_B[3]
thermo_modify format float %.15g
thermo 1
run 0
`;
    const { rows } = await run(script);
    expect(rows[0].f_B as number).toBeCloseTo(0.316950956727411, 12);
    expect(rows[0]['f_B[1]'] as number).toBeCloseTo(0.337103179882921, 12);
    expect(rows[0]['f_B[2]'] as number).toBeCloseTo(-0.278108704894103, 12);
    expect(rows[0]['f_B[3]'] as number).toBeCloseTo(0.437016024448821, 12);
  });

  it('uses the group count N in the density wave (native: 4 atoms give U = 0.000701337888711706)', async () => {
    const atoms = [[0.3, 0.2, 0.1], [1.4, 0.1, 0.6], [0.2, 1.7, 0.9], [1.1, 1.2, 1.4], [2.3, 0.4, 1.9], [2.9, 1.5, 0.2], [2.1, 2.6, 1.1], [0.7, 2.9, 2.4]];
    const script = `units lj
atom_style atomic
atom_modify map array
boundary p p p
region box block 0 3 0 3 0 3
create_box 1 box
${atoms.map((p) => `create_atoms 1 single ${p.join(' ')} units box`).join('\n')}
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
group gg id 1:4
fix 1 all nve
fix B gg rhok 1 0 0 2.0 1.0
thermo_style custom step pe f_B
thermo_modify format float %.15g
thermo 1
run 0
`;
    const { rows } = await run(script);
    expect(rows[0].f_B as number).toBeCloseTo(0.000701337888711706, 12);
  });
});

