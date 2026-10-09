import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { UNIT_SYSTEMS } from '../src/engine/units';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix gcmc in the physical unit styles (src/engine/fix/gcmc.ts). The browser
 * engine used to reject every style but lj because the thermal de Broglie
 * length needs Planck's constant; units.ts now carries native's hplanck per
 * style (measured, see its header) and these tests pin the resulting
 * behaviour. The exact draw-by-draw checks are the w33gcmc_* oracle cases.
 */

const run = async (script: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { error, rows };
};

const box = (units: string, mass: number, mu: number, X: number, steps: number, seed = 29494, T = 300.0) => `units ${units}
atom_style atomic
pair_style lj/cut 3.0
pair_modify tail no
region box block 0 5 0 5 0 5
create_box 1 box
pair_coeff * * 0.0 1.0
mass 1 ${mass}
group gcmcgroup type 1
fix mygcmc gcmcgroup gcmc 1 ${X} 0 1 ${seed} ${T} ${mu} 1.0
thermo_style custom step atoms f_mygcmc[3] f_mygcmc[4] f_mygcmc[5] f_mygcmc[6]
thermo 10
run ${steps}
`;

const meanN = async (units: string, mass: number, mu: number, seed = 29494) => {
  const { error, rows } = await run(box(units, mass, mu, 50, 400, seed));
  expect(error).toBeNull();
  const tail = rows.filter((r) => r.step >= 200).map((r) => r.atoms);
  return tail.reduce((a, b) => a + b, 0) / tail.length;
};

describe('fix gcmc in physical units', () => {
  it('stores a Planck constant for every style that gives native Lambda', () => {
    // Measured with native LAMMPS (black box): Lambda(m = 1, T = 300) at the first-insertion
    // threshold is 1.00727590813955 A in units real and 1.00795006047796 A in units metal.
    const lambdaOf = (style: 'real' | 'metal') => {
      const u = UNIT_SYSTEMS[style];
      return Math.sqrt((u.hplanck * u.hplanck) / (2 * Math.PI * 1.0 * u.boltz * 300 * u.mvv2e));
    };
    expect(Math.abs(lambdaOf('real') / 1.00727590813955 - 1)).toBeLessThan(1e-9);
    expect(Math.abs(lambdaOf('metal') / 1.00795006047796 - 1)).toBeLessThan(1e-9);
    expect(UNIT_SYSTEMS.lj.hplanck).toBe(1);
    for (const style of ['real', 'metal', 'si', 'cgs', 'electron', 'micro', 'nano'] as const) {
      expect(UNIT_SYSTEMS[style].hplanck).toBeGreaterThan(0);
    }
  });

  it('accepts every fix gcmc exchange in units real and metal', async () => {
    for (const units of ['real', 'metal']) {
      const { error } = await run(box(units, 1.0, units === 'real' ? 50.0 : 2.0, 3, 1));
      expect(error, units).toBeNull();
    }
  });

  it('reproduces the native exchange stream for a forced-acceptance run (units real)', async () => {
    // Native (black box, seed 29494, X=3): at step 1 three insertions, step 2 two
    // insertions and one rejected deletion attempt, ...
    const { error, rows } = await run(box('real', 1.0, 50.0, 3, 2));
    expect(error).toBeNull();
    expect(rows[rows.length - 1].atoms).toBe(5);
    expect(rows[rows.length - 1]['f_mygcmc[3]']).toBe(5);
    expect(rows[rows.length - 1]['f_mygcmc[4]']).toBe(5);
    expect(rows[rows.length - 1]['f_mygcmc[5]']).toBe(1);
    expect(rows[rows.length - 1]['f_mygcmc[6]']).toBe(0);
  });

  it('still rejects unsupported keywords naming them', async () => {
    for (const kw of ['pressure', 'rigid', 'shake', 'fugacity_coeff']) {
      const { error } = await run(box('real', 1.0, 0.0, 1, 1).replace('fix mygcmc gcmcgroup gcmc 1 1 0 1 29494 300 0 1.0', `fix mygcmc gcmcgroup gcmc 1 1 0 1 29494 300 0 1.0 ${kw} 1.0`));
      expect(error, kw).toContain(kw);
    }
  });

  it('follows the dilute ideal-gas limit N/V = exp(beta mu) / Lambda^3 in units real', async () => {
    // Lambda(m=1, T=300) from the style's h; V = 125, T = 300, boltz = 0.0019872067.
    const h = UNIT_SYSTEMS.real.hplanck, boltz = UNIT_SYSTEMS.real.boltz, mvv2e = UNIT_SYSTEMS.real.mvv2e;
    const lambda = Math.sqrt((h * h) / (2 * Math.PI * 1.0 * boltz * 300 * mvv2e));
    const mu = 300 * boltz * Math.log((8 * lambda ** 3) / 125);
    const n = await meanN('real', 1.0, mu);
    expect(n).toBeGreaterThan(5.5);
    expect(n).toBeLessThan(11.0);
  }, 120_000);

  it('the mean particle number scales as Lambda^-3 with the mass (heavy gas is denser)', async () => {
    // N <~> V exp(beta mu) / Lambda^3 and Lambda^3 prop m^-3/2, so doubling the
    // mass by 4 raises <N> by 8 in the dilute limit. h cancels in the ratio.
    const mu = -1.6;
    const light = await meanN('real', 1.0, mu);
    const heavy = await meanN('real', 4.0, mu);
    expect(heavy / light).toBeGreaterThan(6.0);
    expect(heavy / light).toBeLessThan(10.5);
  }, 120_000);

  it('keeps the native 6.7e-4 gap between Lambda in real and metal units', () => {
    const lambdaOf = (style: 'real' | 'metal') => {
      const u = UNIT_SYSTEMS[style];
      return Math.sqrt((u.hplanck * u.hplanck) / (2 * Math.PI * 1.0 * u.boltz * 300 * u.mvv2e));
    };
    // Measured with native LAMMPS (black box): 1.00727590813955 / 1.00795006047796.
    expect(lambdaOf('real') / lambdaOf('metal')).toBeCloseTo(0.999331165, 8);
  });
});
