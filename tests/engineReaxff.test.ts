import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { readForceField, readControlFile } from '../src/engine-gpl/reaxff/ffield';

/*
 * pair_style reaxff (src/engine-gpl/reaxff, ported from LAMMPS src/REAXFF at b332f28): potential-file
 * and control-file parsing, element mapping and refused keywords, and the energies of an isolated H2
 * and a water molecule against native LAMMPS (black box) values from the w55reax oracle fixtures.
 */

const POTENTIALS = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const CASES = join(__dirname, 'oracle');
const FIX = join(__dirname, 'fixtures', 'oracle');

const runScript = async (text: string, files: Record<string, string>) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  for (const [name, body] of Object.entries(files)) session.addFile(name, body);
  await session.execute(text);
  return events;
};

describe('ReaxFF force field parsing', () => {
  it('reads the RDX potential: 4 element types C H O N, 39 global parameters', () => {
    const text = readFileSync(join(POTENTIALS, 'ffield.reax.rdx'), 'utf8');
    const { param, control } = readForceField(text, 'ffield.reax.rdx', 0);
    expect(param.gpL.length).toBe(39);
    expect(param.numAtomTypes).toBe(4);
    expect(param.sbp.map((s) => s.name)).toEqual(['C', 'H', 'O', 'N']);
    expect(param.vdwType).toBe(1);
    // bo_cut = 0.01 * gp[29], nonb_low = gp[11], nonb_cut = gp[12] (reaxff_ffield.cpp)
    expect(control.boCut).toBeCloseTo(0.01 * param.gpL[29], 15);
    expect(control.nonbLow).toBe(param.gpL[11]);
    expect(control.nonbCut).toBe(param.gpL[12]);
    // the two-body parameters are symmetric
    const nt = param.numAtomTypes;
    expect(param.tbp[1 * nt + 2]).toEqual(param.tbp[2 * nt + 1]);
  });

  it('rejects a file whose first line is a general-parameter count', () => {
    expect(() => readForceField('39 ! Number of general parameters\n', 'bad.ff', 0)).toThrow();
  });

  it('parses the control file keywords and refuses an unknown one by name', () => {
    const c = readControlFile('tabulate_long_range 10000 ! denotes\nnbrhood_cutoff 4.5\nhbond_cutoff 6.0\n', 'ctl');
    expect(c.tabulate).toBe(10000);
    expect(c.bondCut).toBe(4.5);
    expect(c.hbondCut).toBe(6.0);
    expect(() => readControlFile('frobnicate 3\n', 'ctl')).toThrow(/frobnicate/);
  });
});

describe('pair_style reaxff in the engine', () => {
  const water = readFileSync(join(CASES, 'w55reax_water.data'), 'utf8');
  const pairLines = (ff: string, types: string) =>
    `units real\natom_style charge\nread_data water.data\npair_style reaxff NULL checkqeq no\npair_coeff * * ${ff} ${types}\nrun 0\n`;

  it('refuses an element that is not in the potential file', async () => {
    await expect(runScript(pairLines('ffield.reax.rdx', 'H Xx'), { 'ffield.reax.rdx': readFileSync(join(POTENTIALS, 'ffield.reax.rdx'), 'utf8'), 'water.data': water }))
      .rejects.toThrow(/Non-existent ReaxFF type/);
  });

  it('refuses lgvdw yes and an unported keyword by name', async () => {
    const ff = readFileSync(join(POTENTIALS, 'ffield.reax.rdx'), 'utf8');
    const files = { 'ffield.reax.rdx': ff, 'water.data': water };
    await expect(runScript(`units real\natom_style charge\nread_data water.data\npair_style reaxff NULL lgvdw yes\npair_coeff * * ffield.reax.rdx H O\nrun 0\n`, files)).rejects.toThrow(/lgvdw/);
    await expect(runScript(`units real\natom_style charge\nread_data water.data\npair_style reaxff NULL acks2 yes\npair_coeff * * ffield.reax.rdx H O\nrun 0\n`, files)).rejects.toThrow(/acks2/);
  });

  it('refuses checkqeq yes with no fix qeq/reaxff at the first evaluation', async () => {
    const ff = readFileSync(join(POTENTIALS, 'ffield.reax.rdx'), 'utf8');
    await expect(runScript(`units real\natom_style charge\nread_data water.data\npair_style reaxff NULL\npair_coeff * * ffield.reax.rdx H O\nrun 0\n`, { 'ffield.reax.rdx': ff, 'water.data': water }))
      .rejects.toThrow(/qeq\/reaxff/);
  });

  const stepZero = (fixture: string): ThermoRow => {
    const fx = JSON.parse(readFileSync(join(FIX, fixture), 'utf8')) as { thermo: ThermoRow[] };
    return fx.thermo[0];
  };

  it('isolated H2: the step-0 energies match native LAMMPS', async () => {
    const h2 = readFileSync(join(CASES, 'w55reax_h2.data'), 'utf8');
    const ff = readFileSync(join(POTENTIALS, 'ffield.reax.rdx'), 'utf8');
    const events = await runScript(`units real\natom_style charge\nboundary p p p\nread_data h2.data\npair_style reaxff NULL checkqeq no\npair_coeff * * ffield.reax.rdx H C O N\nthermo_style custom step pe evdwl ecoul\nthermo_modify format float %.15g\nrun 0\n`,
      { 'ffield.reax.rdx': ff, 'h2.data': h2 });
    const row = events.find((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')!.row;
    const want = stepZero('w55reax_h2.json');
    expect(row.pe as number).toBeCloseTo(want.pe as number, 9);
    expect(row.evdwl as number).toBeCloseTo(want.evdwl as number, 9);
    expect(row.ecoul as number).toBeCloseTo(want.ecoul as number, 9);
  });

  it('water molecule: the step-0 energies match native LAMMPS', async () => {
    const ff = readFileSync(join(POTENTIALS, 'ffield.reax.rdx'), 'utf8');
    const events = await runScript(`units real\natom_style charge\nboundary p p p\nread_data water.data\npair_style reaxff NULL checkqeq no\npair_coeff * * ffield.reax.rdx H O\nthermo_style custom step pe evdwl ecoul\nthermo_modify format float %.15g\nrun 0\n`,
      { 'ffield.reax.rdx': ff, 'water.data': water });
    const row = events.find((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')!.row;
    const want = stepZero('w55reax_water.json');
    expect(row.pe as number).toBeCloseTo(want.pe as number, 9);
    expect(row.evdwl as number).toBeCloseTo(want.evdwl as number, 9);
    expect(row.ecoul as number).toBeCloseTo(want.ecoul as number, 9);
  });

  it('refuses fix qeq/reaxff dual and matfree, and names the stage-3 fixes and computes it does not port', async () => {
    const ff = readFileSync(join(POTENTIALS, 'ffield.reax.rdx'), 'utf8');
    const files = { 'ffield.reax.rdx': ff, 'water.data': water };
    const head = `units real\natom_style charge\nread_data water.data\npair_style reaxff NULL\npair_coeff * * ffield.reax.rdx H O\n`;
    await expect(runScript(`${head}fix 2 all qeq/reaxff 1 0.0 10.0 1e-6 reaxff dual\nrun 0\n`, files)).rejects.toThrow(/dual/);
    await expect(runScript(`${head}fix 2 all qeq/reaxff 1 0.0 10.0 1e-6 reaxff matfree\nrun 0\n`, files)).rejects.toThrow(/matfree/);
    await expect(runScript(`${head}fix 2 all reaxff/bonds 1 bonds.reaxff\nrun 0\n`, files)).rejects.toThrow(/reaxff\/bonds/);
    await expect(runScript(`${head}compute c all reaxff/atom bonds yes\nrun 0\n`, files)).rejects.toThrow(/reaxff\/atom/);
  });

  it('refuses fix qeq/reaxff together with fix efield (the field term is not ported)', async () => {
    const ff = readFileSync(join(POTENTIALS, 'ffield.reax.rdx'), 'utf8');
    const head = `units real\natom_style charge\nread_data water.data\npair_style reaxff NULL\npair_coeff * * ffield.reax.rdx H O\n`;
    await expect(runScript(`${head}fix 2 all qeq/reaxff 1 0.0 10.0 1e-6 reaxff\nfix 3 all efield 0.0 0.0 0.1\nrun 0\n`,
      { 'ffield.reax.rdx': ff, 'water.data': water })).rejects.toThrow(/fix qeq\/reaxff with fix efield/);
  });

  it('checkqeq yes (the default) without a fix qeq/reaxff is refused', async () => {
    const ff = readFileSync(join(POTENTIALS, 'ffield.reax.rdx'), 'utf8');
    await expect(runScript(`units real\natom_style charge\nread_data water.data\npair_style reaxff NULL\npair_coeff * * ffield.reax.rdx H O\nrun 0\n`,
      { 'ffield.reax.rdx': ff, 'water.data': water })).rejects.toThrow(/qeq\/reaxff/);
  });
});
