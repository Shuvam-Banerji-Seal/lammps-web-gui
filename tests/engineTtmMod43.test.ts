import { describe, expect, it } from 'vitest';
import { FIXES } from '../src/engine/registry/fix_ttm';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { ttmModGridIndex, ttmModSubsteps, parseTtmModPositional } from '../src/engine/fix/ttm_mod';

/*
 * fix ttm/mod (wave 43). The coupled, diffusion, laser, vacuum and Si cases are oracle cases (tests/oracle/w43tm_*.in,
 * checked by engineOracle.test.ts). This file checks the rules that are measured in small inputs and the refusals:
 *  - the sub-step count max(2, floor(4 C_s) + 1) (measured on cubic and non-cubic grids, see fix/ttm_mod.ts);
 *  - the grid index of an atom on a cell face, below the box and at the top (measured with native LAMMPS);
 *  - the electron energy of the doc's C_e integral (measured: f_2[1] = 429689.700992171);
 *  - the laser clock: a step ending exactly at tau is off, and the clock starts at the fix definition (measured);
 *  - the StyleErrors of the parameters this engine does not reproduce or that native LAMMPS refuses.
 */

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

const PAR_KEYS = ['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'C_0', 'A', 'rho_e', 'D_e', 'gamma_p', 'gamma_s', 'v_0', 'I_0',
  'lsurface', 'rsurface', 'l_skin', 'tau', 'B', 'lambda', 'n_ion', 'surface_movement', 'T_e_min'];

const parFile = (over: Record<string, string> = {}): string => {
  const base: Record<string, string> = {
    a_0: '0', a_1: '0', a_2: '0', a_3: '0', a_4: '0', C_0: '1', A: '0.18', rho_e: '1', D_e: '0', gamma_p: '1',
    gamma_s: '0', v_0: '0', I_0: '0', lsurface: '0', rsurface: '2', l_skin: '1', tau: '0', B: '0', lambda: '0',
    n_ion: '1', surface_movement: '0', T_e_min: '0',
  };
  return PAR_KEYS.map((k) => `# ${k}\n${over[k] ?? base[k]}`).join('\n') + '\n';
};

const runWithFiles = async (text: string, files: Record<string, string>) => {
  const events: EngineEvent[] = [];
  const written = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => written.set(n, (ap ? written.get(n) ?? '' : '') + t),
  });
  for (const [name, t] of Object.entries(files)) session.addFile(name, t);
  await session.execute(text);
  return { events, written };
};

const INIT = '# UNITS: metal COMMENT: x\n1 1 1 1000\n2 1 1 1000\n1 2 1 1000\n2 2 1 1000\n1 1 2 1000\n2 1 2 1000\n1 2 2 1000\n2 2 2 1000\n';

describe('fix ttm/mod registration and positional arguments', () => {
  it('is registered as a ttm-family style', () => {
    expect(Object.keys(FIXES)).toContain('ttm/mod');
  });

  it('takes seed, init_file and the grid (usage error otherwise)', () => {
    const p = parseTtmModPositional('f', ['5', 'p.txt', '4', '4', '4', 'set', '1000']);
    expect(p).toMatchObject({ seed: 5, initFile: 'p.txt', nx: 4, ny: 4, nz: 4, consumed: 5 });
    expect(() => parseTtmModPositional('f', ['5', 'p.txt', '4', '4'])).toThrow(/usage: fix ID group-ID ttm\/mod/);
    expect(() => parseTtmModPositional('f', ['0', 'p.txt', '4', '4', '4'])).toThrow(/seed must be a positive integer/);
  });
});

describe('measured rules', () => {
  it('sub-step count: 2 up to C_s = 0.49, then floor(4 C_s) + 1 (cubic and non-cubic grids); 2 with a non-positive heat capacity', () => {
    expect(ttmModSubsteps(0.05, false)).toBe(2);
    expect(ttmModSubsteps(0.49, false)).toBe(2);
    expect(ttmModSubsteps(0.51, false)).toBe(3);
    expect(ttmModSubsteps(0.75, false)).toBe(4);
    expect(ttmModSubsteps(1.0, false)).toBe(5);
    expect(ttmModSubsteps(1.01, false)).toBe(5);
    expect(ttmModSubsteps(3.0, false)).toBe(13);
    expect(ttmModSubsteps(0.8137, true)).toBe(2);
  });

  it('grid index: a face goes to the upper cell, x = hi to the last cell, a rounding error below lo to the first cell, and a larger shortfall wraps', () => {
    expect(ttmModGridIndex(1.5, 0, 6, 4)).toBe(1);
    expect(ttmModGridIndex(0, 0, 6, 4)).toBe(0);
    expect(ttmModGridIndex(6, 0, 6, 4)).toBe(3);
    expect(ttmModGridIndex(-5.6e-21, 0, 10.8618, 4)).toBe(0);
    expect(ttmModGridIndex(-9.0e-6, 0, 10.8618, 4)).toBe(3);
    expect(ttmModGridIndex(8.14635, 0, 10.8618, 4)).toBe(3);
  });

  it('electron energy is the integral of C_e over the temperature (measured f_2[1] = 429689.700992171)', async () => {
    const text = `${BOX}
fix 2 all ttm/mod 4242 p.txt 2 2 2 set 1000
thermo_style custom step temp f_2[1] f_2[2]
run 1`;
    const { events } = await runWithFiles(text, { 'p.txt': parFile({ a_0: '1', A: '0.18', C_0: '1', rho_e: '1', n_ion: '1' }) });
    const rows = events.filter((e) => e.kind === 'thermo') as { row: Record<string, number> }[];
    const e0 = rows[0].row['f_2[1]'];
    expect(Math.abs(e0 - 429689.700992171) / 429689.700992171).toBeLessThan(1e-12);
  });

  it('laser clock: a sub-step ending exactly at tau is off; the clock starts when the fix is defined (measured: a fix defined after three steps is on for steps 4-6 at tau 0.0035)', async () => {
    const text = `${BOX}
fix 2 all ttm/mod 4242 p.txt 4 4 4 set 1000 outfile 1 q.txt
run 3
`;
    const { written } = await runWithFiles(text, { 'p.txt': parFile({ C_0: '1', rho_e: '1', I_0: '1', l_skin: '1', tau: '0.0015', gamma_p: '1e-12', lsurface: '0', rsurface: '4' }) });
    const cell = (step: number) => {
      const line = written.get(`q.txt.${step}`)!.split('\n').find((l) => l.startsWith('1 1 1 '))!;
      return Number(line.split(' ')[3]);
    };
    // Measured with native LAMMPS (black box, tau 0.0015, dt 0.001): step 1 adds (2/3) I_0 dt, later steps add nothing.
    expect(Math.abs(cell(1) - 1000.000666666667)).toBeLessThan(1e-9);
    expect(Math.abs(cell(2) - cell(1))).toBeLessThan(1e-9);
    expect(Math.abs(cell(3) - cell(1))).toBeLessThan(1e-9);
  });
});

describe('refusals (native LAMMPS refuses these, or the engine does not reproduce them)', () => {
  const expectError = (text: string, pattern: RegExp, files: Record<string, string>) =>
    expect(runWithFiles(text, files)).rejects.toThrow(pattern);
  const cmd = `${BOX}
fix 2 all ttm/mod 4242 p.txt 2 2 2 infile g.init
run 1`;

  it('n_ion must be positive (native: ionic_density must be > 0.0)', () =>
    expectError(cmd, /n_ion \(ionic density\) must be > 0/, { 'p.txt': parFile({ n_ion: '0' }), 'g.init': INIT }));

  it('lsurface must be below rsurface (native message)', () =>
    expectError(cmd, /Left surface coordinate must be less than right surface coordinate/, { 'p.txt': parFile({ lsurface: '2', rsurface: '2' }), 'g.init': INIT }));

  it('a negative lsurface is an error (native message)', () =>
    expectError(cmd, /Surface coordinates must be >= 0/, { 'p.txt': parFile({ lsurface: '-1' }), 'g.init': INIT }));

  it('the electronic pressure (B != 0) is refused by name', () =>
    expectError(cmd, /B != 0 \(electronic pressure force\)/, { 'p.txt': parFile({ B: '0.5' }), 'g.init': INIT }));

  it('a moving vacuum surface (surface_movement 1) is refused by name', () =>
    expectError(cmd, /surface_movement 1 \(moving vacuum surface\)/, { 'p.txt': parFile({ surface_movement: '1' }), 'g.init': INIT }));

  it('a laser with l_skin 0 is refused', () =>
    expectError(cmd, /l_skin must be > 0 when I_0 is set/, { 'p.txt': parFile({ I_0: '1', l_skin: '0' }), 'g.init': INIT }));

  it('a non-integer l_skin is a parse error (native: integer read)', () =>
    expectError(cmd, /l_skin in p.txt must be an integer/, { 'p.txt': parFile({ l_skin: '1.0' }), 'g.init': INIT }));
});
