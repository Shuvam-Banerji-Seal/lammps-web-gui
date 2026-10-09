import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix gcmc (src/engine/fix/gcmc.ts): grand canonical Monte Carlo. The engine
 * does not reproduce native's random stream (see the header of gcmc.ts), so
 * this file checks the documented behaviour: the vector counts, the argument
 * errors, the deterministic zero-displacement moves, and the statistical
 * limit that the average number of particles follows the reservoir chemical
 * potential (in the dilute limit N/V -> exp(beta mu) for units lj, Lambda = 1).
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

const LJ = (fix: string) => `units lj
atom_style atomic
boundary p p p
region box block 0 5 0 5 0 5
create_box 1 box
mass 1 1.0
pair_style lj/cut 3.0
pair_coeff * * 1.0 1.0
${fix}
thermo_style custom step atoms pe f_g[1] f_g[2] f_g[3] f_g[4] f_g[5] f_g[6] f_g[7] f_g[8]
thermo 10
`;

describe('fix gcmc', () => {
  it('accepts every zero-displacement translation and counts it (f[1]=f[2]=steps)', async () => {
    const script = `units lj
atom_style atomic
boundary p p p
region box block 0 5 0 5 0 5
create_box 1 box
mass 1 1.0
pair_style lj/cut 3.0
pair_coeff * * 1.0 1.0
create_atoms 1 single 1.0 1.0 1.0 units box
create_atoms 1 single 2.2 1.0 1.0 units box
fix g all gcmc 1 0 1 1 29494 2.0 -1.0 0.0
thermo_style custom step atoms pe f_g[1] f_g[2] f_g[3] f_g[4] f_g[5] f_g[6] f_g[7] f_g[8]
thermo 1
run 4
`;
    const { error, rows } = await run(script);
    expect(error).toBeNull();
    const last = rows[rows.length - 1];
    expect(last.step).toBe(4);
    expect(last.atoms).toBe(2);
    expect(last['f_g[1]']).toBe(4);
    expect(last['f_g[2]']).toBe(4);
    for (const k of ['f_g[3]', 'f_g[4]', 'f_g[5]', 'f_g[6]', 'f_g[7]', 'f_g[8]']) expect(last[k]).toBe(0);
    expect(rows.every((r) => Math.abs(r.pe - rows[0].pe) < 1e-12)).toBe(true);
  });

  it('rejects unsupported keywords with a StyleError naming them', async () => {
    for (const [kw, val, name] of [['pressure', '1.0', 'pressure'], ['rigid', 'r', 'rigid'], ['fugacity_coeff', '0.9', 'fugacity_coeff']]) {
      const { error } = await run(LJ(`fix g all gcmc 1 1 1 1 29494 2.0 -1.0 1.0 ${kw} ${val}\nrun 1\n`));
      expect(error, kw).toContain(name);
    }
    const bad = await run(LJ('fix g all gcmc 1 1 1 1 29494 2.0 -1.0 1.0 nonsense 1\nrun 1\n'));
    expect(bad.error).toContain('nonsense');
  });

  it('rejects a non-lj unit style naming the restriction', async () => {
    const { error } = await run('units real\natom_style atomic\nregion box block 0 5 0 5 0 5\ncreate_box 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff * * 0.1 3.0\nfix g all gcmc 1 1 1 1 29494 300.0 -5.0 0.5\nrun 1\n');
    expect(error).toMatch(/lj unit style/);
  });

  it('follows the reservoir chemical potential (N grows with mu)', async () => {
    const meanN = async (mu: number): Promise<number> => {
      const script = `units lj
atom_style atomic
boundary p p p
region box block 0 6 0 6 0 6
create_box 1 box
mass 1 1.0
pair_style lj/cut 3.0
pair_coeff * * 1.0 1.0
fix g all gcmc 1 15 0 1 777 2.0 ${mu} 1.0
thermo_style custom step atoms
thermo 100
run 400
`;
      const { error, rows } = await run(script);
      expect(error).toBeNull();
      const tail = rows.filter((r) => r.step >= 200).map((r) => r.atoms);
      return tail.reduce((a, b) => a + b, 0) / tail.length;
    };
    const low = await meanN(-4.0);
    const high = await meanN(-1.0);
    // dilute ideal-gas limit N/V ~ exp(beta mu): T = 2, V = 216, beta mu = -2 / -0.5
    const idealLow = 216 * Math.exp(-2);
    const idealHigh = 216 * Math.exp(-0.5);
    expect(high).toBeGreaterThan(low + 30);
    // attraction raises the density above the ideal gas; the dense branch stays below it
    expect(low).toBeGreaterThan(idealLow * 0.5);
    expect(high).toBeGreaterThan(idealHigh * 0.5);
    expect(high).toBeLessThan(idealHigh * 1.2);
  }, 120_000);
});
