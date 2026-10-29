import { beforeAll, describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { PAIR_STYLES, loadAllStyles } from '../src/engine/styles';

/*
 * compute pair/local pN for the pair styles that define single() extras natively (single_extra):
 *   gran/hooke, gran/hooke/history, gran/hertz/history: 10 (pair_gran_hooke_history.cpp: single_extra = 10; hooke and
 *     hertz derive from it), gayberne and resquared: 6 (pair_gayberne.cpp and pair_resquared.cpp: single_extra = 6),
 *   pair_style hybrid and hybrid/molecular: the largest sub-style value; hybrid/overlay and hybrid/scaled: the sum
 *     (pair_hybrid.cpp, pair_hybrid_overlay.cpp and pair_hybrid_scaled.cpp init_svector).
 * The values themselves are checked against native LAMMPS in tests/oracle/w174mx_pn_*.in. A pN past the count gets
 * native's message (src/compute_pair_local.cpp init at b332f28), which names neither the style nor the field.
 */

beforeAll(() => loadAllStyles());

const SPHERES = `units lj
atom_style sphere
comm_modify vel yes
region b block 0 5 0 5 0 5
create_box 2 b
create_atoms 1 single 1 1 1
create_atoms 2 single 1.8 1 1
set atom * diameter 1.0
set atom * density 1.0
`;

const run = async (script: string) => {
  const session = new Session({ emit: () => undefined, writeFile: () => undefined });
  await session.execute(script);
  return session;
};

describe('pN values of the pair styles with native single() extras', () => {
  it('reports single_extra for the gran styles, gayberne and resquared', () => {
    for (const name of ['gran/hooke', 'gran/hooke/history', 'gran/hertz/history']) {
      expect(PAIR_STYLES[name]().singleExtra()).toBe(10);
    }
    expect(PAIR_STYLES.gayberne().singleExtra()).toBe(6);
    expect(PAIR_STYLES.resquared().singleExtra()).toBe(6);
  });

  it('pN goes up to the style count: p10 is accepted for gran/hooke/history and p11 is refused', async () => {
    await expect(run(`${SPHERES}pair_style gran/hooke/history 2000.0 NULL 50.0 NULL 0.5 1\npair_coeff * *\ncompute c all pair/local p10 cutoff radius\ndump d all local 1 pl.txt c_c\nrun 0\n`)).resolves.toBeTruthy();
    await expect(run(`${SPHERES}pair_style gran/hooke/history 2000.0 NULL 50.0 NULL 0.5 1\npair_coeff * *\ncompute c all pair/local p11 cutoff radius\ndump d all local 1 pl.txt c_c\nrun 0\n`))
      .rejects.toThrow(/Pair style does not have extra field requested by compute pair\/local/);
  });

  it('hybrid takes the largest sub-style count, hybrid/overlay the sum (gran 10 plus lj/cut 0)', async () => {
    const hybrid = await run(`${SPHERES}pair_style hybrid gran/hooke/history 2000.0 NULL 50.0 NULL 0.5 1 lj/cut 1.2\npair_coeff 1 1 gran/hooke/history\npair_coeff 1 2 gran/hooke/history\npair_coeff 2 2 lj/cut 0.5 0.6 1.2\n`);
    expect(hybrid.sys.ff.pair!.singleExtra()).toBe(10);
    const overlay = await run(`${SPHERES}pair_style hybrid/overlay gran/hooke/history 2000.0 NULL 50.0 NULL 0.5 1 lj/cut 1.2\npair_coeff * * gran/hooke/history\npair_coeff * * lj/cut 0.5 0.6 1.2\n`);
    expect(overlay.sys.ff.pair!.singleExtra()).toBe(10);
    await expect(run(`${SPHERES}pair_style hybrid/overlay gran/hooke/history 2000.0 NULL 50.0 NULL 0.5 1 lj/cut 1.2\npair_coeff * * gran/hooke/history\npair_coeff * * lj/cut 0.5 0.6 1.2\ncompute c all pair/local p11 cutoff radius\ndump d all local 1 pl.txt c_c\nrun 0\n`))
      .rejects.toThrow(/Pair style does not have extra field requested by compute pair\/local/);
  });

  it('pN of a style without single() extras is refused', async () => {
    await expect(run(`${SPHERES}pair_style lj/cut 1.2\npair_coeff * * 0.5 0.6\ncompute c all pair/local p1\ndump d all local 1 pl.txt c_c\nrun 0\n`)).rejects.toThrow(/Pair style does not have extra field requested by compute pair\/local/);
  });
});
