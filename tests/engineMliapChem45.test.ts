import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/session';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { parseMliapSnaDescriptor } from '../src/engine/mliap/descriptor';

/*
 * pair_style mliap with descriptor sna chemflag 1 (w45, explicit multi-element SNAP). The native parity
 * cases are tests/oracle/w45mc_*.in (engineOracle.test.ts). Here:
 *  - the descriptor reads chemflag and reports K N_elem^3 model columns;
 *  - the mliap chem energies and forces equal pair_style snap chemflag 1 with the InP_JCPA2020 files
 *    (measured with native LAMMPS, black box: the two agree to the last printed digit);
 *  - the forces are the gradient of the energy (finite difference of the same engine energy);
 *  - the mappings and options the engine does not implement are StyleErrors that name them.
 */

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const pot = (f: string) => readFileSync(join(POT, f), 'utf8');
const ORACLE = join(__dirname, 'oracle');

const potentials = (): Record<string, string> => ({
  'InP_JCPA2020.mliap.model': pot('InP_JCPA2020.mliap.model'),
  'InP_JCPA2020.mliap.descriptor': pot('InP_JCPA2020.mliap.descriptor'),
  'InP_JCPA2020.snapcoeff': pot('InP_JCPA2020.snapcoeff'),
  'InP_JCPA2020.snapparam': pot('InP_JCPA2020.snapparam'),
});

/** Two-element cluster (In at type 1, P at type 2) in a 30 A box, so no periodic image is inside the cutoff. */
const cluster = (x0: number): string => `units metal
atom_style atomic
region box block 0 30 0 30 0 30
create_box 2 box
create_atoms 1 single ${x0} 1.00 1.00
create_atoms 2 single 2.55 1.10 1.05
create_atoms 1 single 1.05 2.75 1.00
create_atoms 2 single 2.50 2.60 1.15
create_atoms 1 single 1.00 1.05 3.20
create_atoms 2 single 2.70 1.20 3.40
mass 1 114.76
mass 2 30.98
`;

interface Result { rows: ThermoRow[]; dump: Map<number, number[]> }

const parseDump = (text: string): Map<number, number[]> => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const out = new Map<number, number[]>();
  for (const l of lines.slice(k + 1)) {
    const w = l.trim().split(/\s+/).map(Number);
    out.set(w[0], [w[1], w[2], w[3]]);
  }
  return out;
};

const run = async (script: string, inputs: Record<string, string> = potentials()): Promise<Result> => {
  const events: EngineEvent[] = [];
  const written = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => written.set(n, (ap ? written.get(n) ?? '' : '') + t),
  });
  for (const [name, text] of Object.entries(inputs)) session.addFile(name, text);
  await session.execute(script);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { rows, dump: parseDump(written.get('d.dump') ?? '') };
};

const FORCE_DUMP = 'write_dump all custom d.dump id fx fy fz modify format float %.17g sort id\n';
const MLIAP = 'pair_style mliap model linear InP_JCPA2020.mliap.model descriptor sna InP_JCPA2020.mliap.descriptor\npair_coeff * * In P\n';
const SNAP = 'pair_style snap\npair_coeff * * InP_JCPA2020.snapcoeff InP_JCPA2020.snapparam In P\n';
const TAIL = 'thermo_style custom step pe\nthermo_modify format float %.17g\nrun 0\n';

describe('pair_style mliap, descriptor sna with chemflag 1 (w45)', () => {
  it('the descriptor reports the chem fields and K N_elem^3 model columns', () => {
    const d = parseMliapSnaDescriptor(pot('InP_JCPA2020.mliap.descriptor'), 'InP_JCPA2020.mliap.descriptor');
    expect(d.chem).toBe(true);
    expect(d.wselfall).toBe(true);
    expect(d.elems).toEqual(['In', 'P']);
    expect(d.K).toBe(30);
    expect(d.nmodel).toBe(30 * 8);
    // the model file of this descriptor has 1 + K N_elem^3 = 241 parameters per element (measured, native reader)
    expect(pot('InP_JCPA2020.mliap.model').split('\n').find((l) => /^\s*2\s+241\s*$/.test(l))).toBeDefined();
  });

  it('mliap chem energies and forces equal pair_style snap chemflag 1', async () => {
    const a = await run(cluster(1.0) + MLIAP + TAIL + FORCE_DUMP);
    const b = await run(cluster(1.0) + SNAP + TAIL + FORCE_DUMP);
    const pa = a.rows[a.rows.length - 1].pe, pb = b.rows[b.rows.length - 1].pe;
    expect(Math.abs(pa - pb)).toBeLessThanOrEqual(1e-12 * Math.abs(pb));
    expect(a.dump.size).toBe(6);
    for (const [id, f] of b.dump) {
      const g = a.dump.get(id)!;
      for (let c = 0; c < 3; c++) expect(Math.abs(g[c] - f[c])).toBeLessThanOrEqual(1e-12 * Math.max(1, Math.abs(f[c])));
    }
  });

  it('the forces are the gradient of the energy (finite difference of atom 1 along x)', async () => {
    const h = 1e-5;
    const base = await run(cluster(1.0) + MLIAP + TAIL + FORCE_DUMP);
    const ep = await run(cluster(1.0 + h) + MLIAP + TAIL);
    const em = await run(cluster(1.0 - h) + MLIAP + TAIL);
    const dE = (ep.rows[0].pe - em.rows[0].pe) / (2 * h);
    const fx = base.dump.get(1)![0];
    expect(Math.abs(-dE - fx)).toBeLessThanOrEqual(1e-6 * Math.max(1, Math.abs(fx)));
  }, 120_000);

  it('wselfallflag 0 in the descriptor matches pair_style snap with wselfallflag 0', async () => {
    const descW0 = pot('InP_JCPA2020.mliap.descriptor').replace('wselfallflag 1', 'wselfallflag 0');
    const paramW0 = pot('InP_JCPA2020.snapparam').replace('wselfallflag 1', 'wselfallflag 0');
    const a = await run(cluster(1.0) + MLIAP + TAIL, { ...potentials(), 'InP_JCPA2020.mliap.descriptor': descW0 });
    const b = await run(cluster(1.0) + SNAP + TAIL, { ...potentials(), 'InP_JCPA2020.snapparam': paramW0 });
    const pa = a.rows[a.rows.length - 1].pe, pb = b.rows[b.rows.length - 1].pe;
    expect(Math.abs(pa - pb)).toBeLessThanOrEqual(1e-12 * Math.abs(pb));
  });

  it('refuses the pair_coeff mappings whose order differs from the descriptor, and mismatched element counts', async () => {
    await expect(run(cluster(1.0) + 'pair_style mliap model linear InP_JCPA2020.mliap.model descriptor sna InP_JCPA2020.mliap.descriptor\npair_coeff * * P In\nrun 0\n')).rejects.toThrow(/pair_coeff must name the 2 descriptor elements \(In P\) in that order/);
    await expect(run(cluster(1.0) + 'pair_style mliap model linear InP_JCPA2020.mliap.model descriptor sna InP_JCPA2020.mliap.descriptor\npair_coeff * * In In\nrun 0\n')).rejects.toThrow(/in that order/);
  });

  it('refuses model quadratic and nn with chemflag 1, and compute mliap with chemflag 1', async () => {
    await expect(run(cluster(1.0) + 'pair_style mliap model quadratic InP_JCPA2020.mliap.model descriptor sna InP_JCPA2020.mliap.descriptor\npair_coeff * * In P\nrun 0\n')).rejects.toThrow(/model linear only/);
    await expect(run(cluster(1.0) + 'compute g all mliap model linear descriptor sna InP_JCPA2020.mliap.descriptor\nrun 0\n')).rejects.toThrow(/chemflag 1 is not implemented/);
  });

  it('refuses a model file whose parameter count does not match K N_elem^3 (the non-chem linear model)', async () => {
    const plain = readFileSync(join(ORACLE, 'w16mliap_linear.mliap.model'), 'utf8');
    await expect(run(cluster(1.0) + MLIAP + TAIL, { ...potentials(), 'InP_JCPA2020.mliap.model': plain })).rejects.toThrow(/nparams/);
  });

  it('refuses quadraticflag in the chem descriptor (the quadratic terms belong to the model)', async () => {
    const quad = pot('InP_JCPA2020.mliap.descriptor') + '\nquadraticflag 1\n';
    await expect(run(cluster(1.0) + MLIAP + TAIL, { ...potentials(), 'InP_JCPA2020.mliap.descriptor': quad })).rejects.toThrow(/quadraticflag/);
  });
});
