import { describe, expect, it } from 'vitest';
import { Session, SUPPORTED_COMMANDS } from '../src/engine/interpreter';
import { formatNumber, splitCommands, substituteVariables, tokenize } from '../src/engine/script';
import { evaluateFormula } from '../src/engine/expr';
import { Rng } from '../src/engine/rng';
import { EngineError, type EngineEvent, type ThermoRow } from '../src/engine/types';

/** Runs input text in a fresh session and collects everything it emits. */
const runScript = async (text: string, frameEvery = 0) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (name, body, append) => files.set(name, (append ? files.get(name) ?? '' : '') + body),
  }, undefined, frameEvery);
  let error: EngineError | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as EngineError;
  }
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  return { session, events, files, thermo, logs, error };
};

// The documented examples/melt setup, written out command by command.
const MELT = `
# 3d Lennard-Jones melt
units           lj
atom_style      atomic

lattice         fcc 0.8442
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create 3.0 87287 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5

neighbor        0.3 bin
neigh_modify    every 20 delay 0 check no

fix             1 all nve

thermo          50
run             250
`;

describe('input splitting (Commands_parse.html rules)', () => {
  it('joins & continuations, drops comments, keeps quoted # and $', () => {
    const cmds = splitCommands([
      'units lj   # trailing comment',
      'region box block 0 10 &',
      '   0 10 0 10',
      'print "a # not a comment $x"',
      'fix 1 all nve & # comment after & stops continuation',
      'run 10',
      '# region r2 block &',
      '  0 1 0 1 0 1',
      'print """',
      'two lines & kept',
      '"""',
    ].join('\n'));
    expect(cmds.map((c) => c.text)).toEqual([
      'units lj',
      // "The next line is concatenated to the previous line by removing the
      // “&” character and line break" — its leading blanks stay
      'region box block 0 10    0 10 0 10',
      'print "a # not a comment $x"',
      'fix 1 all nve &',
      'run 10',
      'print """\ntwo lines & kept\n"""',
    ]);
    expect(cmds.map((c) => c.line)).toEqual([1, 2, 4, 5, 6, 9]);
  });

  it('substitutes $x, ${name}, $(expr) and $(expr:%fmt), tokenises quotes', () => {
    const vars: Record<string, string> = { x: '2.5', name: 'hello world' };
    const out = substituteVariables('a $x ${name} $(1+2) $(PI:%.3f) $xx', (n) => vars[n],
      (f, fmt) => formatNumber(evaluateFormula(f, { thermo: () => undefined, variable: () => 0, rng: () => new Rng(1) }), fmt ?? '%.20g'));
    expect(out).toBe('a 2.5 hello world 3 3.142 2.5x');
    expect(tokenize(`print "two words" 'single' """triple"""`)).toEqual(['print', 'two words', 'single', 'triple']);
    expect(() => tokenize('print "open')).toThrow(/quote/);
  });

  it('formats numbers like C printf', () => {
    expect(formatNumber(3.14159, '%.2f')).toBe('3.14');
    expect(formatNumber(1234.56, '%10.3e')).toBe(' 1.235e+03');
    expect(formatNumber(0.0001234, '%g')).toBe('0.0001234');
    expect(formatNumber(123456789, '%g')).toBe('1.23457e+08');
    expect(formatNumber(2, '%.15g')).toBe('2');
    expect(formatNumber(-7.9, '%d')).toBe('-7');
    expect(() => formatNumber(1, '%.3f%d')).toThrow();
  });
});

describe('equal-style formulas (variable.html)', () => {
  const ev = (f: string) => evaluateFormula(f, { thermo: () => undefined, variable: () => 0, rng: () => new Rng(1) });
  it('follows the documented precedence and functions', () => {
    expect(ev('-2^2')).toBe(4);                 // "will evaluate to 4, not -4"
    expect(ev('-(2^2)')).toBe(-4);
    expect(ev('2^3^2')).toBe(64);               // left to right
    expect(ev('1+2*3')).toBe(7);
    expect(ev('7%4')).toBe(3);
    expect(ev('1 < 2 && 3 >= 3')).toBe(1);
    expect(ev('0 || 0 |^ 1')).toBe(1);
    expect(ev('!0')).toBe(1);
    expect(ev('ternary(0,1,2)')).toBe(2);
    expect(ev('ln(exp(2))')).toBeCloseTo(2, 14);
    expect(ev('log(1000)')).toBeCloseTo(3, 14);
    expect(ev('sign(0)')).toBe(1);
    expect(ev('round(-2.5)')).toBe(-3);
    expect(ev('atan2(1,1)')).toBeCloseTo(Math.PI / 4, 14);
    expect(() => ev('1/0')).toThrow(/division/);
    expect(() => ev('foo + 1')).toThrow(/unknown name/);
  });
});

describe('Session: examples/melt (LAMMPS log.8Apr21.melt.g++.1)', () => {
  it('reproduces the published step-0 thermo line and lands in range at 250', async () => {
    const { thermo, error, events, logs } = await runScript(MELT);
    expect(error).toBeNull();
    const header = events.find((e) => e.kind === 'thermo-header');
    expect(header).toEqual({ kind: 'thermo-header', keywords: ['step', 'temp', 'epair', 'emol', 'etotal', 'press'] });
    expect(thermo.map((r) => r.step)).toEqual([0, 50, 100, 150, 200, 250]);
    const r0 = thermo[0];
    expect(r0.temp).toBeCloseTo(3, 10);
    expect(r0.epair).toBeCloseTo(-6.7733681, 6);
    expect(r0.emol).toBe(0);
    expect(r0.etotal).toBeCloseTo(-2.2744931, 6);
    expect(r0.press).toBeCloseTo(-3.7033504, 6);
    const last = thermo[thermo.length - 1];
    expect(last.temp).toBeGreaterThan(1.5);
    expect(last.temp).toBeLessThan(1.8);
    expect(last.press).toBeGreaterThan(5.0);
    expect(last.press).toBeLessThan(6.4);
    expect(logs).toContain('Created 4000 atoms');
    expect(logs.some((l) => /^Lattice spacing in x,y,z = 1\.6795962/.test(l))).toBe(true);
    expect(events.some((e) => e.kind === 'done' && e.steps === 250)).toBe(true);
  }, 60_000);
});

describe('Session: errors are explicit and carry the line', () => {
  it('rejects an unsupported command with its line number and the supported list', async () => {
    const { error, events } = await runScript('units lj\n\nneb 0.1 0.0 1000 100 10 final coords.final\n');
    expect(error).toBeInstanceOf(EngineError);
    expect(error!.line).toBe(3);
    expect(error!.command).toBe('neb');
    expect(error!.message).toMatch(/not supported/);
    expect(error!.message).toContain(SUPPORTED_COMMANDS.join(', '));
    expect(events.some((e) => e.kind === 'error' && e.line === 3)).toBe(true);
  });

  it('explains commands a browser cannot run, and does not list them as supported', async () => {
    for (const [cmd, why] of [['python', /no Python interpreter/], ['shell', /no operating-system shell/], ['package', /accelerator packages/]] as const) {
      expect(SUPPORTED_COMMANDS).not.toContain(cmd);
      const { error } = await runScript(`units lj\n${cmd} gpu 1\n`);
      expect(error?.command).toBe(cmd);
      expect(error!.message).toMatch(why);
    }
    expect(SUPPORTED_COMMANDS).toEqual(expect.arrayContaining(['processors', 'run', 'include']));
  });

  it.each([
    ['a file the notebook does not have', 'read_data system.data', /cannot open file system.data/],
    ['units after the box', 'lattice sc 1\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\nunits real', /cannot be used after the simulation box/],
    ['2d with a non-periodic z', 'dimension 2\nboundary p p f', /z dimension must be periodic/],
    ['an unsupported pair style', 'pair_style reaxff NULL', /pair_style 'reaxff' is not supported/],
    ['an undefined group', 'lattice sc 1\nregion b block 0 2 0 2 0 2\ncreate_box 1 b\nfix 1 mobile nve', /unknown group 'mobile'/],
    ['missing mass before run', 'lattice sc 1\nregion b block 0 3 0 3 0 3\ncreate_box 1 b\ncreate_atoms 1 box\npair_style lj/cut 1.1\npair_coeff * * 1 1\nrun 1', /masses are set \(type 1\)/],
    ['missing pair coeffs', 'lattice sc 1\nregion b block 0 3 0 3 0 3\ncreate_box 2 b\nmass * 1\npair_style lj/cut 1.1\npair_coeff 1 1 1 1\nrun 1', /all pair coeffs are not set/],
    ['2d box not bracketing z = 0', 'dimension 2\nlattice sq 1\nregion b block 0 2 0 2 0 1\ncreate_box 1 b', /bracket zero/],
    ['undefined variable', 'print "${nope}"', /illegal variable nope/],
    ['unquoted formula with spaces', 'variable a equal 1 + 2', /quote/],
  ])('%s', async (_name, script, pattern) => {
    const { error } = await runScript(script);
    expect(error).toBeInstanceOf(EngineError);
    expect(error!.message).toMatch(pattern);
  });

  it('two integrators on the same atoms warn, as native LAMMPS does', async () => {
    const { error, logs } = await runScript('lattice sc 1\nregion b block 0 3 0 3 0 3\ncreate_box 1 b\ncreate_atoms 1 box\nmass 1 1\npair_style lj/cut 1.1\npair_coeff 1 1 1 1\nfix 1 all nve\nfix 2 all nvt temp 1 1 0.1\nrun 1');
    expect(error).toBeNull();
    expect(logs).toContain('WARNING: One or more atoms are time integrated more than once');
  });
});

describe('Session: variables, print, thermo keywords', () => {
  it('equal variables are evaluated when used; index is not redefined; string is', async () => {
    const { logs, error } = await runScript([
      'variable n index 4',
      'variable n index 9',
      'variable s string first',
      'variable s string second',
      'variable a equal 2*${n}',
      'variable b equal v_a+1',
      'print "n=$n s=${s} a=${a} b=$(v_b:%.2f)"',
      "print 'single $n'",
    ].join('\n'));
    expect(error).toBeNull();
    expect(logs).toContain('n=4 s=second a=8 b=9.00');
    expect(logs).toContain('single 4');
  });

  it('thermo keywords in formulas use the current state (forces computed on demand)', async () => {
    const { logs, error } = await runScript(MELT.replace('run             250', [
      'variable e equal pe',
      'print "pe=$(pe:%.7f) atoms=$(atoms) vol=$(vol:%.4f) e=$e"',
    ].join('\n')));
    expect(error).toBeNull();
    const line = logs.find((l) => l.startsWith('pe='))!;
    expect(line).toMatch(/^pe=-6\.7733681 atoms=4000 vol=4738\.2\d{3} e=-6\.77336/);
  });
});

describe('Session: atoms, dumps and data files', () => {
  it('create_atoms random honours overlap; single uses lattice units by default', async () => {
    const { session, error } = await runScript([
      'lattice sc 1.0',
      'region b block 0 6 0 6 0 6',
      'create_box 2 b',
      'create_atoms 1 random 50 12345 NULL overlap 0.9',
      'create_atoms 2 single 1 2 3',
    ].join('\n'));
    expect(error).toBeNull();
    const s = session.system!;
    expect(s.n).toBe(51);
    expect([s.x[150], s.x[151], s.x[152]]).toEqual([1, 2, 3]);
    let minR2 = Infinity;
    for (let i = 0; i < 50; i++) for (let j = i + 1; j < 50; j++) {
      let r2 = 0;
      for (let d = 0; d < 3; d++) { let dx = s.x[3 * i + d] - s.x[3 * j + d]; dx -= 6 * Math.round(dx / 6); r2 += dx * dx; }
      minR2 = Math.min(minR2, r2);
    }
    expect(Math.sqrt(minR2)).toBeGreaterThanOrEqual(0.9);
  });

  it('dump atom writes the documented header and scaled coordinates at multiples of N', async () => {
    const { files, error } = await runScript(MELT
      .replace('region          box block 0 10 0 10 0 10', 'region          box block 0 4 0 4 0 4')
      .replace('thermo          50\nrun             250', 'dump d all atom 10 melt.dump\ndump c all custom 20 c.*.txt id type x y z ix iy iz\nrun 20'));
    expect(error).toBeNull();
    const text = files.get('melt.dump')!;
    const snaps = text.split('ITEM: TIMESTEP\n').slice(1);
    expect(snaps.map((s) => Number(s.split('\n')[0]))).toEqual([0, 10, 20]);
    const lines = snaps[0].split('\n');
    expect(lines[1]).toBe('ITEM: NUMBER OF ATOMS');
    expect(lines[2]).toBe('256');
    expect(lines[3]).toBe('ITEM: BOX BOUNDS pp pp pp');
    expect(lines[7]).toBe('ITEM: ATOMS id type xs ys zs');
    expect(lines[8]).toBe('1 1 0 0 0');
    expect([...files.keys()].filter((k) => k.startsWith('c.'))).toEqual(['c.0.txt', 'c.20.txt']);
  });

  it('write_data writes atomic Atoms with image flags and Velocities', async () => {
    const { files, error } = await runScript(MELT
      .replace('region          box block 0 10 0 10 0 10', 'region          box block 0 3 0 3 0 3')
      .replace('thermo          50\nrun             250', 'run 30\nwrite_data out.data'));
    expect(error).toBeNull();
    const data = files.get('out.data')!;
    expect(data).toMatch(/^108 atoms$/m);
    expect(data).toMatch(/^1 atom types$/m);
    expect(data).toMatch(/^Atoms # atomic$/m);
    expect(data).toMatch(/^Pair Coeffs # lj\/cut\n\n1 1 1$/m);
    const atoms = data.split('Atoms # atomic\n\n')[1].split('\n\nVelocities')[0].split('\n');
    expect(atoms).toHaveLength(108);
    expect(atoms[0].split(' ')).toHaveLength(8);
  });

  it('run upto, reset_timestep and clear (variables survive)', async () => {
    const { session, thermo, error } = await runScript(MELT
      .replace('region          box block 0 10 0 10 0 10', 'region          box block 0 3 0 3 0 3')
      .replace('thermo          50\nrun             250', [
        'thermo_style custom step temp',
        'run 10',
        'run 25 upto',
        'reset_timestep 1000',
        'run 0',
        'variable keep string yes',
        'clear',
        'print "kept=${keep}"',
      ].join('\n')));
    expect(error).toBeNull();
    expect(thermo.map((r: ThermoRow) => r.step)).toEqual([0, 10, 10, 25, 1000]);
    expect(session.system).toBeNull();
  });
});

describe('Session: 2d', () => {
  it('runs a 2d LJ liquid with enforce2d in the plane', async () => {
    const { session, thermo, error } = await runScript([
      'dimension 2',
      'lattice sq2 0.7',
      'region b block 0 10 0 10 -0.5 0.5',
      'create_box 1 b',
      'create_atoms 1 box',
      'mass 1 1.0',
      'velocity all create 1.0 4928459',
      'pair_style lj/cut 2.5',
      'pair_coeff 1 1 1.0 1.0 2.5',
      'fix 1 all nve',
      'fix 2 all enforce2d',
      'thermo 100',
      'run 300',
    ].join('\n'));
    expect(error).toBeNull();
    const s = session.system!;
    expect(s.n).toBe(200);
    for (let i = 0; i < s.n; i++) expect(s.x[3 * i + 2]).toBe(0);
    expect(thermo[0].temp).toBeCloseTo(1, 10);
    expect(Math.abs(thermo[thermo.length - 1].etotal! - thermo[0].etotal!)).toBeLessThan(0.02);
  });
});

describe('Session: frames for the viewer', () => {
  it('emits frames after create_atoms, every frameEvery steps and at the end', async () => {
    const { events } = await runScript(MELT
      .replace('region          box block 0 10 0 10 0 10', 'region          box block 0 3 0 3 0 3')
      .replace('run             250', 'run 20'), 5);
    const steps = events.filter((e) => e.kind === 'frame').map((e) => (e as { step: number }).step);
    expect(steps).toEqual([0, 5, 10, 15, 20, 20]);
  });
});
