import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * pair_style granular/superellipsoid (src/engine-gpl/superellipsoid). The static contact forces below were measured with
 * native LAMMPS (black box, run 0 on two particles, hooke 1000 0, tangential linear_history 285 0 0.5, damping
 * mass_velocity, so only the normal force acts at setup); the engine must reproduce them to 1e-12. The dynamic cases are
 * the tests/oracle/w80superell_*.in oracle cases.
 */

const pairScript = (shape1: string, block1: string, shape2: string, block2: string, p1 = '0 0 0', p2 = '1.5 0 0', pair = 'hooke 1000.0 0.0 tangential linear_history 285 0.0 0.5 damping mass_velocity') => `units si
atom_style ellipsoid superellipsoid
dimension 3
boundary p p p
newton off
comm_modify vel yes
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single ${p1}
create_atoms 2 single ${p2}
set type 1 mass 1.0
set type 1 shape ${shape1}
set type 1 block ${block1}
set type 2 mass 1.0
set type 2 shape ${shape2}
set type 2 block ${block2}
pair_style granular/superellipsoid
pair_coeff * * ${pair}
`;

const forcesOf = async (script: string): Promise<{ f1: number[]; f2: number[]; tq1: number[] }> => {
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  await session.execute(`${script}run 0\n`);
  const s = session.sys.state;
  const row = (id: number) => {
    const i = Array.from(s.id).indexOf(id);
    return { f: [0, 1, 2].map((d) => s.f[3 * i + d]), t: [0, 1, 2].map((d) => s.torque![3 * i + d]) };
  };
  return { f1: row(1).f, f2: row(2).f, tq1: row(1).t };
};

const close = (a: number[], b: number[]) => {
  for (let d = 0; d < 3; d++) expect(a[d]).toBeCloseTo(b[d], 9);
};

describe('pair granular/superellipsoid, static contacts against native LAMMPS (black box)', () => {
  it('ball (ellipsoid) and pin (superquadric) overlapping in the xy plane', async () => {
    // native: f on atom 1 = -171.30321416454558 -137.04257133163642 0 (normal force 219.37, the midway solution)
    const r = await forcesOf(pairScript('2 2 1', '2 2', '1 1 4', '8 2', '4.0 4.2 5.0', '5.0 5.0 5.0'));
    close(r.f1, [-171.30321416454558, -137.04257133163642, 0]);
    close(r.f2, [171.30321416454558, 137.04257133163642, 0]);
  });

  it('a deeper ball overlap', async () => {
    // native: f on atom 1 = -671.3032141645458 -537.0425713316356 0
    const r = await forcesOf(pairScript('2 2 1', '2 2', '1 1 4', '8 2', '4.5 4.6 5.0', '5.0 5.0 5.0'));
    close(r.f1, [-671.3032141645458, -537.0425713316356, 0]);
  });

  it('two superquadrics with a three-component force and torque', async () => {
    // native: f on atom 1 = -188.02615014016911 -25.560497817823137 -1.4819828036121472; torque 0.44090955078881761 -3.9019100194155367 11.357916637584367
    const r = await forcesOf(pairScript('2 1 1', '3 4', '2 1 1', '4 3', '0.0 0.0 0.0', '1.8 0.3 0.1'));
    close(r.f1, [-188.02615014016911, -25.560497817823137, -1.4819828036121472]);
    close(r.tq1, [0.44090955078881761, -3.9019100194155367, 11.357916637584367]);
  });

  it('pin pair at flat-face distances (native: -70.710678 -70.710678 0 for an overlap of 0.1 along the diagonal)', async () => {
    const r = await forcesOf(pairScript('1 1 4', '8 2', '1 1 4', '8 2', '7.0 7.0 5.0', `${7 + 0.9 / Math.SQRT2} ${7 + 0.9 / Math.SQRT2} 5.0`));
    close(r.f1, [-70.710678118655125, -70.710678118655125, 0]);
  });
});

describe('pair granular/superellipsoid refusals', () => {
  it('names the rolling, twisting and heat models it does not implement', async () => {
    const base = pairScript('2 2 2', '2 2', '2 2 2', '2 2');
    await expect(forcesOf(base.replace('damping mass_velocity', 'damping mass_velocity rolling sds 1 1 1'))).rejects.toThrow(/Rolling models not yet implemented for superellipsoids/);
    await expect(forcesOf(base.replace('damping mass_velocity', 'damping mass_velocity twisting marshall'))).rejects.toThrow(/Twisting models not yet implemented for superellipsoids/);
    await expect(forcesOf(base.replace('damping mass_velocity', 'damping mass_velocity heat area 1.0'))).rejects.toThrow(/Heat models not yet implemented for superellipsoids/);
  });

  it('refuses blockiness below 2 (the contact solver needs convex particles)', async () => {
    await expect(forcesOf(pairScript('2 2 2', '1.5 2', '2 2 2', '2 2'))).rejects.toThrow(/block1 and block2 >= 2/);
  });

  it('refuses an unknown normal model', async () => {
    await expect(forcesOf(pairScript('2 2 2', '2 2', '2 2 2', '2 2', '0 0 0', '1 0 0', 'dmt 1 1 1'))).rejects.toThrow(/Unknown normal model/);
  });

  it('needs atom style ellipsoid superellipsoid', async () => {
    const session = new Session({ emit: () => {}, writeFile: () => {} });
    await expect(session.execute(`units si
atom_style ellipsoid
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5 5 5
set type 1 shape 2 2 2
pair_style granular/superellipsoid
pair_coeff * * hooke 1000.0 0.0 tangential linear_history 285 0.0 0.5
run 0
`)).rejects.toThrow(/requires atom attributes rmass, angmom and superellipsoid flag/);
  });
});
