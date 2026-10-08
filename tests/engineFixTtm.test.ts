import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { substepCount } from '../src/engine/fix/ttm';
import { parseTtmModParams, ttmModBlockers } from '../src/engine/fix/ttm_mod';

/*
 * fix ttm / ttm/grid / ttm/mod (wave 14). The coupled runs are oracle cases
 * (tests/oracle/w14ttm_*.in, checked by engineOracle.test.ts). This file checks:
 *  - the outfile of fix ttm against a native outfile (tests/fixtures/oracle/w14ttm_outfile.json,
 *    written by native LAMMPS from the w14ttm_diff input with outfile 2; line 1 is the DATE header
 *    and is skipped, the grid numbers are compared to 1e-8 relative);
 *  - the measured sub-step rule of the explicit diffusion (docs.lammps.org/fix_ttm.html: the heat
 *    equation is solved as given there; the sub-step count is the measured native one);
 *  - the parameter file of fix ttm/mod and the StyleErrors of every unsupported case.
 */

const FIX = join(__dirname, 'fixtures', 'oracle');
const CASES = join(__dirname, 'oracle');

const runWithFiles = async (text: string, files: Record<string, string>) => {
  const events: EngineEvent[] = [];
  const written = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => written.set(n, (ap ? written.get(n) ?? '' : '') + t),
  });
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  await session.execute(text);
  return written;
};

const BOX = `
units metal
atom_style atomic
lattice sc 3.0
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 10.0
displace_atoms all move 0.5 0.5 0.5 units box
velocity all set 1.0 0.5 -0.3 units box
pair_style lj/cut 3.0
pair_coeff 1 1 0.0 2.0
fix 1 all nve
timestep 0.001
`;

const INIT = '# UNITS: metal COMMENT: x\n1 1 1 4000\n2 1 1 2000\n1 2 1 3000\n2 2 1 1500\n1 1 2 6000\n2 1 2 2000\n1 2 2 2800\n2 2 2 1200\n';

describe('fix ttm outfile', () => {
  it('matches the native outfile of the diffusion case (steps 2 and 4)', async () => {
    const native = JSON.parse(readFileSync(join(FIX, 'w14ttm_outfile.json'), 'utf8')) as { files: Record<string, string> };
    // the native input itself (w14ttm_diff.in) with the outfile keyword added
    const text = readFileSync(join(CASES, 'w14ttm_diff.in'), 'utf8')
      .split('\n').filter((l) => !l.startsWith('#')).join('\n')
      .replace('infile w14ttm_diff.init', 'infile w14ttm_diff.init outfile 2 T.out');
    const init = readFileSync(join(CASES, 'w14ttm_diff.init'), 'utf8');
    const written = await runWithFiles(text, { 'w14ttm_diff.init': init });
    for (const name of ['T.out.2', 'T.out.4']) {
      const mine = written.get(name);
      expect(mine, name).toBeDefined();
      const a = mine!.trim().split('\n'), b = native.files[name].trim().split('\n');
      expect(a.length).toBe(b.length);
      expect(a[0].startsWith('# DATE:')).toBe(true);
      for (let i = 1; i < a.length; i++) {
        const x = a[i].trim().split(/\s+/), y = b[i].trim().split(/\s+/);
        expect(x.slice(0, 3)).toEqual(y.slice(0, 3));
        const u = Number(x[3]), v = Number(y[3]);
        expect(Math.abs(u - v) <= 1e-8 * Math.max(1, Math.abs(v)), `${name} line ${i + 1}: ${u} vs ${v}`).toBe(true);
      }
    }
  });

  it('writes the header and grid lines in the infile layout', async () => {
    const text = `${BOX}
fix 2 all ttm 1 0.001 1.0 0.0 1.0 0.0 2.0 2 2 2 set 1000.0 outfile 1 Q.out
run 1`;
    const written = await runWithFiles(text, {});
    const q = written.get('Q.out.1');
    expect(q).toBeDefined();
    const lines = q!.trim().split('\n');
    expect(lines[0]).toMatch(/^# DATE: \d{4}-\d{2}-\d{2} UNITS: metal COMMENT: Electron temperature on 2x2x2 grid at step 1 - created by fix ttm$/);
    expect(lines.length).toBe(9);
    // the atoms move on the first step and hand a little energy to the grid: the value stays near the set temperature
    expect(Math.abs(Number(lines[1].split(' ')[3]) - 1000)).toBeLessThan(0.1);
  });
});

describe('explicit sub-step count (measured with native LAMMPS, C = dt kappa/(C_e rho_e) sum 1/L_d^2)', () => {
  it('uses one step up to C = 1/2 and floor(2C)+1 above', () => {
    expect(substepCount(0.2)).toBe(1);
    expect(substepCount(0.4999)).toBe(1);
    expect(substepCount(0.5)).toBe(1);
    expect(substepCount(0.5133)).toBe(2);
    expect(substepCount(0.9167)).toBe(2);
    expect(substepCount(1.0)).toBe(3);
    expect(substepCount(1.1)).toBe(3);
    expect(substepCount(2.0)).toBe(5);
    expect(substepCount(3.0)).toBe(7);
  });
});

describe('fix ttm errors', () => {
  const expectError = (text: string, pattern: RegExp, files: Record<string, string> = {}) =>
    expect(runWithFiles(text, files)).rejects.toThrow(pattern);

  it('needs gamma_p > 0 (measured with native LAMMPS, black box)', () => expectError(`${BOX}
fix 2 all ttm 1 1.0 1.0 0.0 0.0 0.0 2.0 2 2 2
run 1`, /gamma_p must be > 0/));

  it('rejects set with a non-positive temperature', () => expectError(`${BOX}
fix 2 all ttm 1 1.0 1.0 0.0 1.0 0.0 2.0 2 2 2 set 0.0
run 1`, /set Tinit must be > 0/));

  it('rejects an unknown keyword', () => expectError(`${BOX}
fix 2 all ttm 1 1.0 1.0 0.0 1.0 0.0 2.0 2 2 2 bogus 3
run 1`, /unknown keyword 'bogus'/));

  it('ttm/grid has no outfile keyword (fix_ttm.html)', () => expectError(`${BOX}
fix 2 all ttm/grid 1 1.0 1.0 0.0 1.0 0.0 2.0 2 2 2 set 1000.0 outfile 1 Q.out
run 1`, /does not support the outfile keyword/));

  it('rejects an infile whose UNITS tag differs from the run units', () => expectError(`${BOX}
fix 2 all ttm 1 1.0 1.0 0.0 1.0 0.0 2.0 2 2 2 infile g.init
run 1`, /written for units real/, { 'g.init': '# UNITS: real COMMENT: x\n' + INIT.split('\n').slice(1).join('\n') }));

  it('rejects an infile that leaves a grid point unset', () => expectError(`${BOX}
fix 2 all ttm 1 1.0 1.0 0.0 1.0 0.0 2.0 2 2 2 infile g.init
run 1`, /did not set all/, { 'g.init': INIT.split('\n').slice(0, 8).join('\n') + '\n' }));

  it('rejects a negative infile temperature', () => expectError(`${BOX}
fix 2 all ttm 1 1.0 1.0 0.0 1.0 0.0 2.0 2 2 2 infile g.init
run 1`, /must not be negative/, { 'g.init': INIT.replace('4000', '-4000') }));

  it('needs a positive integer seed', () => expectError(`${BOX}
fix 2 all ttm 0 1.0 1.0 0.0 1.0 0.0 2.0 2 2 2
run 1`, /positive integer/));
});

describe('fix ttm/mod parameter file', () => {
  const values: Record<string, string> = {
    a_0: '0.5', a_1: '0', a_2: '0', a_3: '0', a_4: '0', C_0: '0.5', A: '1.0', rho_e: '1.0', D_e: '10.0',
    gamma_p: '2.0', gamma_s: '0.0', v_0: '0.0', I_0: '0.0', lsurface: '1', rsurface: '4', l_skin: '3',
    tau: '1.0', B: '0.0', lambda: '1.0', n_ion: '1.0', surface_movement: '0', T_e_min: '300',
  };
  const file = (over: Record<string, string> = {}) => Object.entries(values)
    .map(([k, v]) => `# ${k}\n${over[k] ?? v} comment text`)
    .join('\n') + '\n';

  it('reads the even lines as values and ignores the comments and trailing text', () => {
    const p = parseTtmModParams(file(), 'f', 'p.txt');
    expect(p.a[0]).toBe(0.5);
    expect(p.Cz).toBe(0.5);
    expect(p.De).toBe(10);
    expect(p.rsurface).toBe(4);
    expect(p.teMin).toBe(300);
  });

  it('reads lsurface, rsurface, l_skin and surface_movement as integers (measured with native LAMMPS: a non-integer value stops the run)', () => {
    expect(() => parseTtmModParams(file({ l_skin: '1.0' }), 'f', 'p.txt')).toThrow(/l_skin in p.txt must be an integer/);
    expect(() => parseTtmModParams(file({ lsurface: '0.5' }), 'f', 'p.txt')).toThrow(/lsurface/);
  });

  it('names the unsupported physics of an input (laser, pressure, diffusion, vacuum)', () => {
    const p = parseTtmModParams(file({ D_e: '0.0', I_0: '2.0', B: '0.5' }), 'f', 'p.txt');
    expect(ttmModBlockers(p, 4)).toEqual(['I_0 != 0 (laser source)', 'B != 0 (electronic pressure force)']);
    const q = parseTtmModParams(file({ rsurface: '3' }), 'f', 'p.txt');
    expect(ttmModBlockers(q, 4).join(' ')).toMatch(/vacuum/);
  });

  it('is an unsupported style in the engine (not registered)', async () => {
    await expect(runWithFiles(`${BOX}
fix 2 all ttm/mod 5 m.txt 2 2 2 infile g.init
run 1`, { 'm.txt': file(), 'g.init': INIT })).rejects.toThrow(/fix style 'ttm\/mod' is not supported/);
  });
});
