import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { PAIR_STYLES } from '../src/engine/styles';

/*
 * Pair styles through write_restart / read_restart (output/restart.ts).
 *
 * read_restart.rst: "Here is a list of information included in a restart file" and the
 * per-style doc pages say which pair styles "write their information to binary restart
 * files". For each doc-declared style below, a setup taken from a native-checked oracle
 * script runs 10 steps uninterrupted, and a second run writes a restart after 5 steps,
 * clears, reads it in the same session and runs 5 more steps. Positions and velocities must
 * match (forces are not stored; the next run recomputes them).
 */

const ORACLE_DIR = join(__dirname, 'oracle');

const newSession = () => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  return { session, events, files };
};

/** The setup lines of an oracle script (before its first run), without the lines a restart does not keep. */
const setupFor = (file: string): string[] | null => {
  const lines = readFileSync(join(ORACLE_DIR, file), 'utf8').split('\n');
  const first = lines.findIndex((l) => /^run\b/.test(l.trim()));
  if (first < 0) return null;
  const setup = lines.slice(0, first).filter((l) => !/^\s*(#|$)/.test(l) || false);
  const kspace = setup.filter((l) => /^\s*kspace_/.test(l));
  if (setup.some((l) => /\b(read_data|include|read_restart|molecule|read_dump|create_atoms.*\bfile)\b/.test(l))) return null;
  return setup.filter((l) => !/^\s*(fix|thermo|dump|write_|minimize|compute|restart)/.test(l));
};

/** The pair style name a setup selects, when it selects exactly one. */
const pairNameOf = (setup: string[]): string | null => {
  const ps = setup.filter((l) => /^\s*pair_style\s/.test(l));
  if (ps.length !== 1) return null;
  return ps[0].trim().split(/\s+/)[1];
};

const oracleSetups = (): Map<string, { file: string; setup: string[] }> => {
  const out = new Map<string, { file: string; setup: string[] }>();
  for (const file of readdirSync(ORACLE_DIR).filter((f) => f.endsWith('.in')).sort()) {
    const setup = setupFor(file);
    if (!setup) continue;
    const name = pairNameOf(setup);
    if (name && !out.has(name)) out.set(name, { file, setup });
  }
  return out;
};

/** Runs the setup, then either 10 steps straight through or 5 steps, a restart, and 5 more. */
const runPaths = async (setup: string[]) => {
  // fixes and kspace settings are not stored: the restarted run specifies them again
  const kspace = setup.filter((l) => /^\s*kspace_/.test(l));
  const tail = ['fix 1 all nve'];
  const a = newSession();
  await a.session.execute([...setup, ...tail, 'run 10', ''].join('\n'));
  const boxOf = (s: Session): [number, number, number] => {
    const bx = s.sys.state.box;
    return [bx.hi[0] - bx.lo[0], bx.hi[1] - bx.lo[1], bx.hi[2] - bx.lo[2]];
  };
  const direct = { x: Array.from(a.session.sys.state.x), v: Array.from(a.session.sys.state.v), step: a.session.sys.state.step };

  const b = newSession();
  await b.session.execute([...setup, ...tail, 'run 5', 'write_restart pairs.restart', 'clear', 'read_restart pairs.restart', ...kspace, ...tail, 'run 5', ''].join('\n'));
  const restarted = { x: Array.from(b.session.sys.state.x), v: Array.from(b.session.sys.state.v), step: b.session.sys.state.step };
  return { direct, restarted, lengths: boxOf(b.session), pair: b.session.sys.ff.pair?.name ?? null, text: b.files.get('pairs.restart') ?? '' };
};

/** Compares two flat xyz arrays; positions are wrapped into the box, so periodic differences use the minimum image. */
const close = (a: number[], b: number[], box: [number, number, number] | undefined, tol: number) => {
  expect(a.length).toBe(b.length);
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    let d = a[i] - b[i];
    if (box) { const L = box[i % 3]; d -= L * Math.round(d / L); }
    worst = Math.max(worst, Math.abs(d) / (1 + Math.abs(a[i])));
  }
  expect(worst).toBeLessThan(tol);
};

/*
 * Measured with native LAMMPS (black box): a gran/hooke/history run (the oracle setup, 100 spheres
 * under gravity) restarted after 5 steps reaches a different kinetic energy at step 10 than the
 * uninterrupted run (0.720252 against 0.720631, a relative difference of about 5e-4). pair_granular.rst says the same ("This pair style
 * will not restart exactly"). The browser engine follows that, so the gran family is compared to 1e-2 (the browser engine's restarted gran runs differ by up to about 1e-3 relative).
 */
const GRANULAR_TOLERANCE = 1e-2;

/** Pair styles whose doc page says they write their information to binary restart files (see restart.ts). */
const DOC_STORED = [
  'lj/cut', 'lj/cut/coul/cut', 'lj/cut/coul/long', 'lj/cut/coul/debye', 'lj/cut/coul/dsf', 'lj/cut/coul/wolf',
  'lj/charmm/coul/charmm', 'lj/class2', 'lj/gromacs',
  'coul/cut', 'coul/debye', 'coul/dsf', 'coul/wolf',
  'born', 'buck', 'buck/coul/cut',
  'lj/cubic', 'lj/expand', 'lj/relres', 'lj/smooth', 'lj/smooth/linear', 'lj96/cut', 'mie/cut', 'morse', 'soft',
  'yukawa', 'yukawa/colloid', 'zero', 'colloid', 'gauss', 'atm',
  'gran/hooke', 'gran/hooke/history', 'gran/hertz/history', 'granular',
];

/** Doc-declared styles that the restore cannot rebuild yet (see restart.ts, HOOKS NEEDED). */
const DOC_STORED_PENDING = ['lepton', 'lepton/coul', 'lepton/sphere'];

describe('pair styles through write_restart / read_restart', () => {
  const setups = oracleSetups();
  for (const style of DOC_STORED) {
    const hit = setups.get(style);
    it(`${style}: restart run equals the uninterrupted run`, async () => {
      expect(style in PAIR_STYLES).toBe(true);
      expect(hit, `no oracle setup selects pair_style ${style}`).toBeTruthy();
      const r = await runPaths(hit!.setup);
      expect(r.pair).toBe(style);
      expect(r.text.startsWith('LAMMPS-WEB-RESTART 1\n')).toBe(true);
      expect(r.restarted.step).toBe(r.direct.step);
      // gran/* and granular: see GRANULAR_TOLERANCE
      const tol = style.startsWith('gran') ? GRANULAR_TOLERANCE : 1e-9;
      close(r.restarted.x, r.direct.x, r.lengths, tol);
      close(r.restarted.v, r.direct.v, undefined, tol);
    });
  }

  for (const style of DOC_STORED_PENDING) {
    it(`${style}: write_restart refuses with a StyleError until the restore hook exists`, async () => {
      const hit = setups.get(style);
      expect(hit, `no oracle setup selects pair_style ${style}`).toBeTruthy();
      const { session } = newSession();
      await expect(session.execute([...hit!.setup, 'run 0', 'write_restart x.restart', ''].join('\n'))).rejects.toThrow(/not stored in the browser restart file yet/);
    });
  }
});

/*
 * pair_table.rst (quoted in restart.ts, PAIR_SETTINGS_ONLY): the settings are stored and the
 * coefficients are not, so pair_coeff is specified again after read_restart.
 */
const TABLE_FILE = ['TAB', 'N 12 R 1.0 6.5', '', ...Array.from({ length: 12 }, (_, i) => {
  const r = 1 + i * 0.5;
  return `${i + 1} ${r} ${(4 * (Math.pow(1 / r, 12) - Math.pow(1 / r, 6))).toFixed(10)} ${(-24 * (2 * Math.pow(1 / r, 12) - Math.pow(1 / r, 6)) / r).toFixed(10)}`;
})].join('\n') + '\n';

const TABLE_BOX = `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 3 0 3 0 3
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 1.2 4321 loop geom
pair_style      table linear 1000
pair_coeff      * * tab.txt TAB 6.5
timestep        0.002
`;

describe('pair_style table through write_restart / read_restart', () => {
  it('stores the table settings, not the coefficients; pair_coeff is specified again after the restart', async () => {
    const direct = newSession();
    direct.session.addFile('tab.txt', TABLE_FILE);
    await direct.session.execute(`${TABLE_BOX}fix 1 all nve\nrun 10\n`);
    const want = Array.from(direct.session.sys.state.x);

    const { session, files } = newSession();
    session.addFile('tab.txt', TABLE_FILE);
    await session.execute(`${TABLE_BOX}fix 1 all nve\nrun 5\nwrite_restart tab.restart\n`);
    expect(files.get('tab.restart')).toContain('"name":"table"');
    expect(files.get('tab.restart')).not.toContain('"tE"');
    await session.execute('clear\nread_restart tab.restart\n');
    expect(session.sys.ff.pair?.name).toBe('table');
    expect((session.sys.ff.pair as unknown as { mode: string; ntable: number }).mode).toBe('linear');
    expect((session.sys.ff.pair as unknown as { ntable: number }).ntable).toBe(1000);
    await expect(session.execute('run 0\n')).rejects.toThrow();
    await session.execute('pair_coeff * * tab.txt TAB 6.5\nfix 1 all nve\nrun 5\n');
    const got = Array.from(session.sys.state.x);
    expect(got.length).toBe(want.length);
    let worst = 0;
    const L = session.sys.state.box.hi[0] - session.sys.state.box.lo[0];
    for (let i = 0; i < want.length; i++) {
      let d = got[i] - want[i];
      d -= L * Math.round(d / L);
      worst = Math.max(worst, Math.abs(d));
    }
    expect(worst).toBeLessThan(1e-9);
  });
});

/*
 * Pair styles whose doc page says they do not write their information (restart.ts,
 * PAIR_NOT_IN_RESTART): write_restart stores no pair style, read_restart leaves the pair style
 * unset, and the next run needs a new pair_style and pair_coeff.
 */
describe('pair styles the doc says are not written to restart files', () => {
  it('zbl: no pair style in the file, and the restarted run needs pair_style again', async () => {
    const { session, files } = newSession();
    await session.execute(`${LJ_BOX_ZBL}run 2\nwrite_restart zbl.restart\n`);
    expect(files.get('zbl.restart')).toContain('"pair":null');
    expect(files.get('zbl.restart')).toContain('"pairNotStored":"zbl"');
    await session.execute('clear\nread_restart zbl.restart\n');
    expect(session.sys.ff.pair).toBeNull();
    expect(session.sys.ff.pairNotRestarted).toBe('zbl');
  });
});

const LJ_BOX_ZBL = `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 1.5 1234 loop geom
pair_style      zbl 1.0 1.2
pair_coeff      * * 14 14
timestep        0.004
fix             1 all nve
`;

/*
 * pair_modify settings are stored with the style (read_restart.rst lists "pair_modify settings (mix,
 * shift, tail, table)"). Measured with native LAMMPS (black box): pe and press after the restart equal
 * those before it for "pair_modify shift yes mix arithmetic table 0" and for "pair_modify tail yes";
 * the same script's "tail yes" with "shift yes" is refused ("Cannot have both pair_modify shift and tail
 * set to yes").
 */
describe('pair_modify settings through a restart', () => {
  const PM_BOX = `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 2 0 2 0 2
create_box      2 box
create_atoms    1 box
mass            * 1.0
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0
pair_coeff      2 2 0.8 1.1
pair_coeff      1 2 0.9 1.0 2.0
`;
  const pe = (events: EngineEvent[]): number[] => events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row.pe ?? e.row.PotEng);

  for (const [label, pm] of [['shift yes mix arithmetic table 0', 'pair_modify shift yes mix arithmetic table 0'], ['tail yes', 'pair_modify tail yes']] as const) {
    it(`${label}: the restarted pe equals the pe before the restart`, async () => {
      const { session, events, files } = newSession();
      await session.execute(`${PM_BOX}${pm}\nthermo_style custom step pe press\nrun 0\nwrite_restart pm.restart\n`);
      const before = pe(events);
      events.length = 0;
      await session.execute('clear\nread_restart pm.restart\nthermo_style custom step pe press\nrun 0\n');
      const after = pe(events);
      expect(files.has('pm.restart')).toBe(true);
      expect(before.length).toBe(1);
      expect(after.length).toBe(1);
      expect(after[0]).toBeCloseTo(before[0], 12);
      expect(session.sys.ff.pair?.name).toBe('lj/cut');
    });
  }
});
