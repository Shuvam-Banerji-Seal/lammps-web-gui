import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

const PARAM36 = readFileSync(join(__dirname, 'oracle', 'w36qeq_param.qeq2'), 'utf8');
const ROOT = join(__dirname, '..');

/** The w36qeq_* oracle system (six charged atoms, coul/long with pppm), up to its fix line, then the given fix and run 0. */
const W36 = readFileSync(join(__dirname, 'oracle', 'w36qeq_point.in'), 'utf8');
const pppmSystem = (fix: string) => {
  const pre = W36.slice(0, W36.indexOf('fix q all'));
  return `${pre}${fix}\nthermo_style custom step pe\nthermo_modify format float %.15g\nrun 0\n`;
};

const smallSystem = (fix: string) => `units metal
atom_style charge
boundary f f f
region box block 0 30 0 30 0 30
create_box 2 box
create_atoms 1 single 5 15 15
create_atoms 2 single 8 15 15
mass 1 1.0
mass 2 1.0
set atom 1 charge 0.0
set atom 2 charge 0.0
pair_style coul/cut 20.0
pair_coeff * *
${fix}
`;

const session = () => {
  const lines: string[] = [];
  const s = new Session({ emit: (e: unknown) => { lines.push(typeof e === 'string' ? e : JSON.stringify(e)); } });
  s.addFile('params.qeq', PARAM36);
  return { s, lines };
};

type Sys = { sys: { ff: { kspace: { paramsFrozen: boolean; gEwald: number } | null } } };

describe('fix qeq family in src/engine-gpl (lazy group fix_qeq)', () => {
  it('the core qeq implementation is gone; the GPL class and registry exist', () => {
    expect(existsSync(join(ROOT, 'src/engine/fix/qeq.ts'))).toBe(false);
    expect(existsSync(join(ROOT, 'src/engine/registry/fix_qeq.ts'))).toBe(false);
    expect(existsSync(join(ROOT, 'src/engine-gpl/qeq/fix_qeq.ts'))).toBe(true);
    expect(existsSync(join(ROOT, 'src/engine-gpl/registry/fix_qeq.ts'))).toBe(true);
  });

  it('the five styles are the GPL registry entries', async () => {
    const mod = await import('../src/engine-gpl/registry/fix_qeq');
    expect(Object.keys(mod.FIXES).sort()).toEqual(['qeq/dynamic', 'qeq/fire', 'qeq/point', 'qeq/shielded', 'qeq/slater']);
  });

  it('point, dynamic and fire solve before the setup force (native setup_pre_force): step-0 pe matches native', async () => {
    // native LAMMPS (tests/fixtures/oracle/w36qeq_{point,dyn,fire}.json, first thermo row)
    const native: Record<string, number> = {
      'qeq/point': -6.4223085227782, 'qeq/dynamic': -6.42230852295664, 'qeq/fire': -6.42230852275624,
    };
    for (const style of Object.keys(native)) {
      const { s, lines } = session();
      await s.execute(pppmSystem(`fix q all ${style} 1 8 1.0e-10 100000 params.qeq`));
      const thermo = lines.map((l) => JSON.parse(l) as { kind: string; row?: { step: number; pe: number } })
        .filter((e) => e.kind === 'thermo' && e.row?.step === 0);
      expect(thermo.length, 'thermo row at step 0').toBeGreaterThan(0);
      const pe = thermo[0].row!.pe;
      expect(Math.abs(pe - native[style]) / Math.abs(native[style]), style).toBeLessThan(1e-12);
    }
  });

  it('the kspace G-ewald is frozen during a run and released after it (native: init-time charges)', async () => {
    const { s } = session();
    await s.execute(pppmSystem('fix q all qeq/point 1 8 1.0e-12 100000 params.qeq\nrun 0'));
    const kspace = (s as unknown as Sys).sys.ff.kspace!;
    expect(kspace.paramsFrozen).toBe(false);
  });

  it('native CG warning text: "Fix qeq CG convergence failed (...) after N iterations at step S"', async () => {
    const { s, lines } = session();
    await s.execute(smallSystem('fix q all qeq/point 1 20 1.0e-12 1 params.qeq\nrun 0'));
    expect(lines.join('\n')).toMatch(/Fix qeq CG convergence failed \(.+\) after 1 iterations at step 0/);
  });

  it('native dynamic warning text: "Charges did not converge at step S: X"', async () => {
    const { s, lines } = session();
    await s.execute(smallSystem('fix q all qeq/dynamic 1 20 1.0e-12 3 params.qeq\nrun 0'));
    expect(lines.join('\n')).toMatch(/Charges did not converge at step 0: /);
  });
});
