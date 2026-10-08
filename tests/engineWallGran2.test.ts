import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * fix wall/gran granular and wall/gran/region granular (docs.lammps.org/fix_wall_gran.html,
 * fix_wall_gran_region.html): argument errors and analytic single-sphere wall forces.
 */

const base = (fix: string, region = '') => `units lj
atom_style sphere
atom_modify map array
comm_modify vel yes
boundary f f f
region box block -6 6 -6 6 -6 6 units box
create_box 1 box
create_atoms 1 single 0.1 0.05 0.4
set atom 1 diameter 1.0 density 1.0
${region}
${fix}
run 0
`;

/** Runs the input; returns a force component of atom 1 after run 0 (dump columns: 1 = fx, 2 = fy, 3 = fz). */
const fzOn = async (text: string, comp = 3): Promise<number> => {
  const files = new Map<string, string>();
  const session = new Session({
    emit: () => {},
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  await session.execute(`${text}\nwrite_dump all custom wg.dump id fx fy fz modify format float %.17g sort id\n`);
  const lines = files.get('wg.dump')!.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  return Number(lines[k + 1].trim().split(/\s+/)[comp]);
};

describe('fix wall/gran granular argument errors', () => {
  it('needs a wallstyle', async () => {
    await expect(fzOn(base('fix 2 all wall/gran granular hooke 1000 50 tangential linear_nohistory 1 0.4 damping velocity'))).rejects.toThrow(/missing wallstyle/);
  });
  it('rejects the removed zcylinder wallstyle and region wallstyle on wall/gran', async () => {
    await expect(fzOn(base('fix 2 all wall/gran granular hooke 1000 50 tangential linear_nohistory 1 0.4 damping velocity zcylinder 2.0'))).rejects.toThrow(/zcylinder keyword has been removed/);
    await expect(fzOn(base('fix 2 all wall/gran granular hooke 1000 50 tangential linear_nohistory 1 0.4 damping velocity region reg', 'region reg block -1 1 -1 1 -1 1 units box'))).rejects.toThrow(/wall\/gran\/region/);
  });
  it('needs both lo and hi (NULL allowed, not both)', async () => {
    await expect(fzOn(base('fix 2 all wall/gran granular hooke 1000 50 tangential linear_nohistory 1 0.4 damping velocity zplane NULL NULL'))).rejects.toThrow(/both NULL/);
  });
  it('names an unsupported keyword and an unsupported region style', async () => {
    await expect(fzOn(base('fix 2 all wall/gran granular hooke 1000 50 tangential linear_nohistory 1 0.4 damping velocity zplane 0.0 NULL contacts'))).rejects.toThrow(/contacts/);
    await expect(fzOn(base('fix 2 all wall/gran/region granular hooke 1000 50 tangential linear_nohistory 1 0.4 damping velocity region reg', 'region reg block -1 1 -1 1 -1 1 units box side out'))).rejects.toThrow(/side-out/);
  });
  it('rejects a side-out region', async () => {
    await expect(fzOn(base('fix 2 all wall/gran/region granular hooke 1000 50 tangential linear_nohistory 1 0.4 damping velocity region reg', 'region reg block -1 1 -1 1 -1 1 side out units box'))).rejects.toThrow(/side-out/);
  });
});

describe('fix wall/gran granular analytic single-sphere force', () => {
  // sphere of radius 0.5 at z = 0.4 against a zplane wall at z = 0: overlap 0.1, normal +z
  it('hooke on a plane: F_z = k_n delta', async () => {
    const f = await fzOn(base('fix 2 all wall/gran granular hooke 1000.0 0.0 tangential linear_nohistory 0.0 0.0 damping velocity zplane 0.0 NULL'));
    expect(f).toBeCloseTo(1000 * 0.1, 9);
  });
  it('hertz/material on a plane: F = (4/3) E_eff sqrt(R) delta^(3/2), E_eff = E / (2 (1 - nu^2))', async () => {
    const E = 1e6, nu = 0.3;
    const eff = E / (2 * (1 - nu * nu));
    const f = await fzOn(base(`fix 2 all wall/gran granular hertz/material ${E} 0.0 ${nu} tangential linear_nohistory 0.0 0.0 damping velocity zplane 0.0 NULL`));
    expect(f).toBeCloseTo((4 / 3) * eff * Math.sqrt(0.5) * 0.1 ** 1.5, 6);
  });
  it('hertz on an interior cylinder: R_eff = R Rw / (R + Rw) with Rw = -2 Rc (measured)', async () => {
    // sphere at x = 2.2 inside a cylinder of radius 2.5 (axis z): overlap 0.2, Rw = -5, R_eff = 5/9;
    // the wall pushes the sphere back towards the axis (-x)
    const reg = 'region reg cylinder z 0.0 0.0 2.5 0.3 6.0 side in units box';
    const text = base('fix 2 all wall/gran/region granular hertz 2000.0 0.0 tangential linear_nohistory 0.0 0.0 damping velocity region reg', reg)
      .replace('create_atoms 1 single 0.1 0.05 0.4', 'create_atoms 1 single 2.2 0.0 3.0');
    const fx = await fzOn(text, 1);
    expect(fx).toBeCloseTo(-2000 * Math.sqrt(5 / 9) * 0.2 ** 1.5, 6);
  });
});

describe('fix wall/gran/region classic fstyles and cones', () => {
  it('rejects a classic fstyle with too few parameters and a bad dampflag', async () => {
    await expect(fzOn(base('fix 2 all wall/gran/region hooke 1000 NULL 0.4 region reg', 'region reg block -1 1 -1 1 -1 1 units box'))).rejects.toThrow(/needs Kn Kt/);
    await expect(fzOn(base('fix 2 all wall/gran/region hooke 1000 NULL 0.4 NULL 0.6 2 region reg', 'region reg block -1 1 -1 1 -1 1 units box'))).rejects.toThrow(/dampflag/);
  });
  it('hooke on an interior cone: normal along the generator, overlap = distance to the generator (measured)', async () => {
    // cone radius 1.0 at z = 0 to 2.0 at z = 3; sphere at (1.2, 0, 1.5): distance 0.2846, F = k_n delta n
    const reg = 'region reg cone z 0.0 0.0 1.0 2.0 0.0 3.0 side in units box';
    const text = base('fix 2 all wall/gran/region hooke 2000.0 NULL 0.0 NULL 0.0 0 region reg', reg)
      .replace('create_atoms 1 single 0.1 0.05 0.4', 'create_atoms 1 single 1.2 0.0 1.5');
    const fx = await fzOn(text, 1);
    const fz = await fzOn(text, 3);
    const delta = 0.5 - 0.3 / Math.sqrt(1 + (1 / 3) ** 2);
    const nx = -1 / Math.sqrt(1 + 1 / 9), nz = (1 / 3) / Math.sqrt(1 + 1 / 9);
    expect(fx).toBeCloseTo(2000 * delta * nx, 4);
    expect(fz).toBeCloseTo(2000 * delta * nz, 4);
  });
});
