/**
 * atom_style apip and the APIP commands of the core (src/engine/atoms.ts, commands/setup.ts, compute/peratom.ts,
 * output/data.ts and output/restart.ts): the messages and behaviour checked against native LAMMPS b332f28 built with
 * PKG_APIP. Native runs of these inputs gave: "unknown argument foo", "Invalid atom_style apip command", "Cannot set
 * attribute apip/lambda for atom style atomic", "apip/lambda 1.5 not in [0,1]", "Invalid keyword apip_lambda_input for
 * atom style apip in compute property/atom command" (and "... apip_lambda for atom style atomic ..." for an atomic
 * style); atom_style hybrid apip atomic runs; write_restart/read_restart keeps
 * the switching parameters (sum 1 for four atoms at 0.25) and read_data starts them at 0 ("Atoms # apip" lines). The
 * stage-2 messages (fix lambda/apip, fix lambda_thermostat/apip, pair lambda/input/csp/apip and lambda/zone/apip) and
 * the stage-3 ones (fix lambda/la/csp/apip) are also the native ones of these inputs. The native parity of the pair
 * styles and fixes is the oracle cases w278ap_*, w279ap_* and w280ap_*.
 */
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const base = ['units metal', 'lattice fcc 3.6', 'region b block 0 1 0 1 0 1', 'create_box 1 b', 'create_atoms 1 box', 'mass 1 63.5'];

const run = async (lines: string[]): Promise<{ error: string; rows: Record<string, number>[]; files: Record<string, string> }> => {
  const events: EngineEvent[] = [];
  const files: Record<string, string> = {};
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: (n: string, t: string) => { files[n] = t; } } as never);
  try {
    await session.execute(lines.join('\n'));
  } catch {
    // the error event carries the message
  }
  const err = events.find((e) => e.kind === 'error');
  const rows = events.filter((e) => e.kind === 'thermo').map((e) => (e as unknown as { row: Record<string, number> }).row);
  return { error: err && 'message' in err ? String(err.message) : '', rows, files };
};

describe('atom_style apip', () => {
  it('refuses an unknown argument and more than one argument (native messages)', async () => {
    expect((await run(['atom_style apip foo', ...base])).error).toMatch(/unknown argument foo/);
    expect((await run(['atom_style apip thermostat conservative', ...base])).error).toMatch(/Invalid atom_style apip command/);
  });

  it('runs as a hybrid sub-style, and in both modes', async () => {
    for (const style of ['atom_style hybrid apip atomic', 'atom_style apip thermostat', 'atom_style apip conservative']) {
      const r = await run([style, ...base, 'pair_style zero 3.0', 'pair_coeff * *', 'run 0']);
      expect(r.error, style).toBe('');
    }
  });
});

describe('set apip/lambda', () => {
  it('needs an apip atom style and a value in [0,1] (native messages)', async () => {
    expect((await run(['atom_style atomic', ...base, 'set group all apip/lambda 0.5'])).error).toMatch(/Cannot set attribute apip\/lambda for atom style atomic/);
    expect((await run(['atom_style apip', ...base, 'set group all apip/lambda 1.5'])).error).toMatch(/apip\/lambda 1\.5 not in \[0,1\]/);
  });

  it('takes fast (1), precise (0), a number and an atom-style variable', async () => {
    const sum = async (setLine: string): Promise<number> => {
      const r = await run(['atom_style apip', ...base, setLine, 'compute c all property/atom apip_lambda', 'compute s all reduce sum c_c',
        'thermo_style custom step c_s', 'pair_style zero 3.0', 'pair_coeff * *', 'run 0']);
      expect(r.error).toBe('');
      return r.rows[0].c_s;
    };
    expect(await sum('set group all apip/lambda fast')).toBe(4);
    expect(await sum('set group all apip/lambda precise')).toBe(0);
    expect(await sum('set group all apip/lambda 0.25')).toBe(1);
    expect(await sum('variable l atom 0.1*id\nset group all apip/lambda v_l')).toBeCloseTo(1.0, 14);
  });
});

describe('compute property/atom apip fields', () => {
  it('refuses apip_lambda_input in the conservative mode (native message)', async () => {
    const r = await run(['atom_style apip conservative', ...base, 'compute c all property/atom apip_lambda_input']);
    expect(r.error).toMatch(/Invalid keyword apip_lambda_input for atom style apip in compute property\/atom command/);
  });

  it('refuses the apip fields for another atom style', async () => {
    const r = await run(['atom_style atomic', ...base, 'compute c all property/atom apip_lambda']);
    expect(r.error).toMatch(/Invalid keyword apip_lambda for atom style atomic in compute property\/atom command/);
  });
});

describe('data and restart files', () => {
  const sumLines = ['compute c all property/atom apip_lambda', 'compute s all reduce sum c_c', 'thermo_style custom step c_s', 'pair_style zero 3.0', 'pair_coeff * *', 'run 0'];

  it('write_data labels the Atoms section apip, and read_data starts lambda at zero', async () => {
    const w = await run(['atom_style apip conservative', ...base, 'set group all apip/lambda 0.25', 'write_data d.data']);
    expect(w.error).toBe('');
    expect(w.files['d.data']).toMatch(/^Atoms # apip$/m);
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
    session.addFile('d.data', w.files['d.data']);
    await session.execute(['units metal', 'atom_style apip conservative', 'read_data d.data', ...sumLines].join('\n'));
    const row = events.filter((e) => e.kind === 'thermo').map((e) => (e as unknown as { row: Record<string, number> }).row)[0];
    expect(row.c_s).toBe(0);
  });

  it('write_restart keeps the switching parameters', async () => {
    const r = await run(['atom_style apip conservative', ...base, 'set group all apip/lambda 0.25', 'write_restart r.rst', 'clear', 'read_restart r.rst', ...sumLines]);
    expect(r.error).toBe('');
    expect(r.rows[0].c_s).toBe(1);
  });
});

describe('dynamic switching parameters (stage 2): native messages', () => {
  const ex = join(__dirname, '..', 'third_party', 'lammps', 'examples', 'PACKAGES', 'apip');
  const box = ['units metal', 'atom_style apip', 'lattice fcc 3.615', 'region b block 0 3 0 3 0 3', 'create_box 1 b', 'create_atoms 1 box', 'mass 1 63.5'];
  const pairs = ['pair_style hybrid/overlay eam/fs/apip lambda/input/csp/apip fcc cutoff 6.0 lambda/zone/apip 3.0',
    'pair_coeff * * eam/fs/apip Cu_300K_Immel_2023.eam.fs Cu', 'pair_coeff * * lambda/input/csp/apip', 'pair_coeff * * lambda/zone/apip'];
  const errorOf = async (lines: string[]): Promise<string> => {
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
    session.addFile('Cu_300K_Immel_2023.eam.fs', readFileSync(join(ex, 'Cu_300K_Immel_2023.eam.fs'), 'latin1'));
    try {
      await session.execute([...box, ...lines].join('\n'));
    } catch {
      // the error event carries the message
    }
    const err = events.find((e) => e.kind === 'error');
    return err && 'message' in err ? String(err.message) : '';
  };
  const cases: [string, string[], RegExp][] = [
    ['thresholds', ['fix l all lambda/apip 2.0 1.0'], /fix lambda: Illegal or missing threshold values/],
    ['history length', ['fix l all lambda/apip 1.0 2.0 time_averaged_zone 2 3 1 5'], /fix lambda: history_length > 1 required/],
    ['unknown fix lambda argument', ['fix l all lambda/apip 1.0 2.0 bogus'], /fix lambda\/apip: unknown argument bogus/],
    ['lambda_non_group', ['fix l all lambda/apip 1.0 2.0 lambda_non_group 2'], /fix lambda\/apip: Illegal value of lambda_non_group/],
    ['thermostat seed', ['fix t all lambda_thermostat/apip seed 0'], /fix lambda_thermostat\/apip seed <= 0/],
    ['unknown thermostat argument', ['fix t all lambda_thermostat/apip foo'], /fix lambda_thermostat\/apip: unknown argument foo/],
    ['N_rescaling', ['fix t all lambda_thermostat/apip N_rescaling 1'], /fix lambda_thermostat\/apip: rescaling_N_neighbours <= 1/],
    ['odd csp neighbour count', ['pair_style lambda/input/csp/apip 7'], /pair_lambda_input_csp: even number of neighbours > 1 for csp calculation required/],
    ['unknown csp argument', ['pair_style lambda/input/csp/apip fcc foo 1'], /pair_lambda_input_csp: unknown argument foo/],
    ['zone argument count', ['pair_style lambda/zone/apip 1 2'], /pair_lambda_zone: expected 1 instead of 2 arguments/],
    ['zone cutoff', ['pair_style lambda/zone/apip 0'], /pair_lambda_zone: cut_global = 0 <= 0/],
    ['fix_modify before a run', [...pairs, 'fix l all lambda/apip 1.0 2.0', 'fix_modify l 2.0 3.0'], /fix lambda\/apip: cutoff of potential smaller than cutoff of switching region/],
    ['lambda/input without fix lambda/apip', [...pairs, 'run 0'], /Exact one fix lambda\/apip required/],
  ];
  for (const [label, lines, re] of cases) {
    it(label, async () => {
      expect(await errorOf(lines)).toMatch(re);
    });
  }
});

describe('conservative switching parameters (stage 3, fix lambda/la/csp/apip): native messages', () => {
  const box = ['units metal', 'atom_style apip conservative', 'lattice fcc 3.615', 'region b block 0 3 0 3 0 3', 'create_box 1 b', 'create_atoms 1 box', 'mass 1 63.5'];
  const fix = 'fix l all lambda/la/csp/apip 0.7 2.0 3.0 5.0 fcc';
  const cases: [string, string[], RegExp][] = [
    ['too few arguments', ['fix l all lambda/la/csp/apip 0.7 2.0 3.0'], /fix lambda\/la\/csp\/apip requires five arguments/],
    ['csp_mode', [`${fix} csp_mode foo`], /expected dynamic or static instead of foo/],
    ['unknown argument', [`${fix} bogus`], /unknown argument bogus/],
    ['cutoffs', ['fix l all lambda/la/csp/apip 0.7 2.0 5.0 3.0 fcc'], /0 <= cut_lo <= cut_hi required/],
    ['thresholds', ['fix l all lambda/la/csp/apip 2.0 0.7 3.0 5.0 fcc'], /0 <= threshold_lo <= threshold_hi required/],
    ['lambda_non_group', [`${fix} lambda_non_group 3`], /0 <= lambda_non_group <= 1 required/],
    ['store_peratom', [`${fix} store_peratom 0`], /store_peratom frequency needs to be positive\./],
    ['pair cutoff', [fix, 'pair_style zero 4.0', 'pair_coeff * *', 'run 0'], /cutoff of potential \(4\) smaller than cutoff of weighting function \(5\)/],
  ];
  for (const [label, lines, re] of cases) {
    it(label, async () => {
      expect((await run([...box, ...lines])).error).toMatch(re);
    });
  }
});
