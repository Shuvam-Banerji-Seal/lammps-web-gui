import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { parseGranularSpec } from '../src/engine/force/pair/granular';

/*
 * pair_style granular (docs.lammps.org/pair_granular.html): argument errors and an
 * analytic two-sphere normal force for each normal model (particles at rest, so
 * no damping; zero tangential coefficient, so no friction).
 */

const pairInput = (coeff: string, x1 = -0.45, x2 = 0.45) => `units lj
atom_style sphere
atom_modify map array
comm_modify vel yes
boundary f f f
region b block -10 10 -10 10 -10 10
create_box 1 b
create_atoms 1 single ${x1} 0 0
create_atoms 1 single ${x2} 0 0
set atom 1 diameter 1.0 density 1.0
set atom 2 diameter 1.0 density 1.0
pair_style granular
pair_coeff ${coeff}
timestep 0.0005
run 0
`;

/** Runs the input and returns the x-force on atom 1 (atom ids 1 and 2). */
const fxOnFirst = async (text: string): Promise<number> => {
  const files = new Map<string, string>();
  const session = new Session({
    emit: () => {},
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  await session.execute(`${text}\nwrite_dump all custom pg.dump id fx fy fz modify format float %.17g sort id\n`);
  const lines = files.get('pg.dump')!.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  return Number(lines[k + 1].trim().split(/\s+/)[1]);
};

describe('pair_style granular argument errors', () => {
  it('rejects a missing tangential keyword', () => {
    expect(() => parseGranularSpec(['hooke', '1000', '50'])).toThrow(/tangential/);
  });
  it('rejects an unknown normal model', () => {
    expect(() => parseGranularSpec(['bogus', '1', '2', 'tangential', 'linear_nohistory', '1', '0.4'])).toThrow(/unknown normal model 'bogus'/);
  });
  it('names the unsupported mdr model', () => {
    expect(() => parseGranularSpec(['mdr', '1', '2', '3', '4', '5', '6', 'tangential', 'linear_nohistory', '1', '0.4'])).toThrow(/'mdr' is not implemented/);
  });
  it('accepts dmt and jkr (no longer StyleError) and rejects limit_damping with them', () => {
    expect(() => parseGranularSpec(['dmt', '1000', '0.5', '0.3', '0', 'tangential', 'linear_nohistory', '1', '0.4'])).not.toThrow();
    expect(() => parseGranularSpec(['jkr', '1000', '0.5', '0.3', '0.1', 'tangential', 'linear_nohistory', '1', '0.4', 'limit_damping'])).toThrow(/limit_damping/);
  });
  it('rejects tsuji damping with a cohesive model', () => {
    expect(() => parseGranularSpec(['dmt', '1000', '0.5', '0.3', '0', 'tangential', 'linear_nohistory', '1', '0.4', 'damping', 'tsuji'])).toThrow(/not compatible/);
  });
  it('names the unimplemented damping and tangential options', () => {
    expect(() => parseGranularSpec(['hooke', '1000', '50', 'tangential', 'linear_nohistory', '1', '0.4', 'damping', 'coeff_restitution'])).toThrow(/coeff_restitution/);
    expect(() => parseGranularSpec(['hooke', '1000', '50', 'tangential', 'linear_nohistory', '1', '0.4', 'heat', 'area', '0.1'])).toThrow(/heat/);
  });
  it('rejects NULL k_t for linear_history', () => {
    expect(() => parseGranularSpec(['hooke', '1000', '50', 'tangential', 'linear_history', 'NULL', '1', '0.4'])).toThrow(/NULL/);
  });
  it('rejects an unset pair when the types cannot be mixed', async () => {
    const text = `units lj
atom_style sphere
comm_modify vel yes
region b block -10 10 -10 10 -10 10
create_box 2 b
create_atoms 1 single -0.45 0 0
create_atoms 2 single 0.45 0 0
set atom 1 diameter 1.0 density 1.0
set atom 2 diameter 1.0 density 1.0
pair_style granular
pair_coeff 1 1 hooke 1000 50 tangential linear_nohistory 1 0.4
pair_coeff 2 2 hertz 1000 50 tangential linear_nohistory 1 0.4
run 0
`;
    await expect(fxOnFirst(text)).rejects.toThrow(/not set|differ/);
  });
});

describe('pair_style granular analytic two-sphere normal force', () => {
  // at rest: separation 0.9, overlap delta = 0.1, n = (x_1 - x_2)/r = -x; force on atom 1 is -|F_n|
  const delta = 0.1, R = 0.25;
  it('hooke: F = k_n delta', async () => {
    expect(await fxOnFirst(pairInput('* * hooke 1000.0 0.0 tangential linear_nohistory 0.0 0.0'))).toBeCloseTo(-1000 * delta, 9);
  });
  it('hertz: F = k_n sqrt(R) delta^(3/2)', async () => {
    const f = await fxOnFirst(pairInput('* * hertz 1000.0 0.0 tangential linear_nohistory 0.0 0.0'));
    expect(f).toBeCloseTo(-1000 * Math.sqrt(R) * delta ** 1.5, 9);
  });
  it('hertz/material: F = (4/3) E_eff sqrt(R) delta^(3/2), E_eff = E / (2 (1 - nu^2))', async () => {
    const E = 1e6, nu = 0.3;
    const eff = E / (2 * (1 - nu * nu));
    const f = await fxOnFirst(pairInput(`* * hertz/material ${E} 0.0 ${nu} tangential linear_nohistory 0.0 0.0`));
    expect(f).toBeCloseTo(-(4 / 3) * eff * Math.sqrt(R) * delta ** 1.5, 6);
  });
  it('dmt: F = (4/3) E_eff sqrt(R) delta^(3/2) - 4 pi gamma R for delta > 0 (no force beyond contact)', async () => {
    const E = 1e6, nu = 0.3, gamma = 0.1;
    const eff = E / (2 * (1 - nu * nu));
    const f = await fxOnFirst(pairInput(`* * dmt ${E} 0.0 ${nu} ${gamma} tangential linear_nohistory 0.0 0.0`));
    expect(f).toBeCloseTo(-((4 / 3) * eff * Math.sqrt(R) * delta ** 1.5 - 4 * Math.PI * gamma * R), 6);
    const zero = await fxOnFirst(pairInput(`* * dmt ${E} 0.0 ${nu} ${gamma} tangential linear_nohistory 0.0 0.0`, -0.51, 0.51));
    expect(zero).toBe(0);
  });

  /*
   * Bouncing collision (contact breaks and re-forms, rolling and twisting history, unequal radii).
   * Without the neighbour bin-width fix (src/engine/neighbor.ts buildList: bx, by, bz must be at
   * least binsize) the second neighbour build of this case allocates ~2e10 bins and the run never
   * returns; the test is therefore skipped until that fix lands (see the report).
   */
  it('bouncing collision runs 60 steps with rolling and twisting history', async () => {
    const text = `units lj
atom_style sphere
atom_modify map array
comm_modify vel yes
boundary f f f
region b block -10 10 -10 10 -10 10
create_box 1 b
create_atoms 1 single -0.6 0 0
create_atoms 1 single 0.6 0 0
set atom 1 diameter 1.0 density 1.0
set atom 2 diameter 1.3 density 1.0
set atom 1 vx 0.6 vy 0.1 vz 0.01
set atom 2 vx -0.6 vy -0.2 vz 0.1
set atom 1 omega 3.0 -1.0 0.5
set atom 2 omega -1.5 0.8 1.0
pair_style granular
pair_coeff * * hertz 2000.0 5.0 tangential linear_history 700.0 0.6 0.4 rolling sds 300.0 20.0 0.3 twisting marshall
timestep 0.0005
fix 1 all nve/sphere
run 60
`;
    const rows: { press: number; etotal: number }[] = [];
    const session = new Session({
      emit: (e: any) => { if (e.kind === "thermo" && e.row) rows.push(e.row); },
      writeFile: () => {},
    });
    await session.execute(text);
    expect(rows.length).toBeGreaterThan(0);
    for (const r of rows) {
      expect(Number.isFinite(r.press)).toBe(true);
      expect(Number.isFinite(r.etotal)).toBe(true);
    }
  }, 60000);
});
