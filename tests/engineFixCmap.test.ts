import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { FixCmap, parseCmapGrids, buildCmapTable, derivativeMatrix, evaluateCmap } from '../src/engine/fix/cmap';

/*
 * fix cmap (wave 15): grid reader, bicubic interpolation, forces, read_data / write_data plumbing and the
 * oracle cases w15cmap_* (tests/oracle). The cases are run with the fix pushed into the session before
 * read_data, because the `fix` command itself is refused before the box exists (see the wave-15 report).
 */

const ORACLE = join(__dirname, 'oracle');
/** The same final dump the oracle harness appends (tests/engineOracle.test.ts). */
const FINAL_DUMP = 'write_dump all custom oracle_final.dump id type xu yu zu vx vy vz fx fy fz modify format float %.17g sort id';
const FIX = join(__dirname, 'fixtures', 'oracle');

/** A 24 x 24 grid with node values from a smooth non-separable function (row = phi node, column = psi node). */
const syntheticGrid = (): number[][] => {
  const G: number[][] = [];
  for (let i = 0; i < 24; i++) {
    const row: number[] = [];
    for (let j = 0; j < 24; j++) {
      const phi = ((-180 + 15 * i) * Math.PI) / 180, psi = ((-180 + 15 * j) * Math.PI) / 180;
      row.push(1.3 * Math.cos(phi) - 0.7 * Math.sin(2 * psi) + 0.5 * Math.cos(phi - psi) + 0.2 * Math.sin(3 * phi) * Math.cos(psi));
    }
    G.push(row);
  }
  return G;
};
const gridText = (G: number[][]): string => G.map((r) => r.map((v) => String(v)).join(' ')).join('\n');

const rad = (deg: number) => (deg * Math.PI) / 180;

describe('CMAP grid reader', () => {
  it('reads grids in file order, skipping # comments (inline too)', () => {
    const G = syntheticGrid();
    const text = `# header comment\n${gridText(G).replace('\n', ' # first row\n')}\n# second grid\n${gridText(G)}\n`;
    const grids = parseCmapGrids(text, 'g.txt');
    expect(grids.length).toBe(2);
    expect(grids[0][3 * 24 + 12]).toBeCloseTo(G[3][12], 12);
    expect(grids[1][23 * 24 + 5]).toBeCloseTo(G[23][5], 12);
  });

  it('takes whole lines per grid and drops surplus values on the last line (measured with native)', () => {
    // a leading value, then 24 rows of 24 values: the grid is 577 values over whole lines, so the last value is dropped
    const rows = Array.from({ length: 24 }, (_, r) => Array.from({ length: 24 }, (_, c) => String(r * 24 + c)).join(' '));
    const grids = parseCmapGrids(['7', ...rows].join('\n'), 'g.txt');
    expect(grids.length).toBe(1);
    expect(grids[0][0]).toBe(7);
    expect(grids[0][1]).toBe(0);
    expect(grids[0][575]).toBe(574);
  });

  it('refuses non-numeric tokens and incomplete grids', () => {
    expect(() => parseCmapGrids('1 2 x', 'g.txt')).toThrow(/non-numeric value 'x'/);
    expect(() => parseCmapGrids('1 2 3', 'g.txt')).toThrow(/ends inside a CMAP grid/);
    expect(() => parseCmapGrids('# nothing\n', 'g.txt')).toThrow(/holds no CMAP grid/);
  });
});

describe('CMAP bicubic interpolation', () => {
  const G = syntheticGrid();
  const tab = buildCmapTable(Float64Array.from(G.flat()), derivativeMatrix());

  it('reproduces the grid values at the nodes', () => {
    for (const [i, j] of [[0, 0], [3, 12], [12, 3], [23, 23], [7, 19]]) {
      const e = evaluateCmap(tab, rad(-180 + 15 * i), rad(-180 + 15 * j)).e;
      expect(e).toBeCloseTo(G[i][j], 11);
    }
  });

  it('is periodic in phi and psi', () => {
    const a = evaluateCmap(tab, rad(-180), rad(37));
    const b = evaluateCmap(tab, rad(180), rad(37));
    expect(a.e).toBeCloseTo(b.e, 12);
    expect(a.dphi).toBeCloseTo(b.dphi, 9);
  });

  it('has derivatives consistent with the energy (finite differences)', () => {
    const h = 1e-6;
    for (const [p, q] of [[17.3, -62.9], [-141.2, 99.4], [5.5, 5.5], [170.2, -175.0]]) {
      const phi = rad(p), psi = rad(q);
      const r = evaluateCmap(tab, phi, psi);
      const dphi = (evaluateCmap(tab, phi + h, psi).e - evaluateCmap(tab, phi - h, psi).e) / (2 * h);
      const dpsi = (evaluateCmap(tab, phi, psi + h).e - evaluateCmap(tab, phi, psi - h).e) / (2 * h);
      expect(Math.abs(r.dphi - dphi)).toBeLessThan(1e-6);
      expect(Math.abs(r.dpsi - dpsi)).toBeLessThan(1e-6);
    }
  });
});

/** Runs an oracle input with fix cmap pushed into the session before read_data (the fix command is refused pre-box). */
const runWithFix = async (name: string, extra = '') => {
  const text = readFileSync(join(ORACLE, `${name}.in`), 'utf8');
  const lines = text.split('\n');
  const k = lines.findIndex((l) => /^\s*fix\s+\S+\s+all\s+cmap\s/.test(l));
  expect(k).toBeGreaterThan(-1);
  const cm = lines[k].trim().split(/\s+/);
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({ emit: (e) => events.push(e), writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  for (const f of ['w15cmap_grid.txt', 'w15cmap_chain.data']) session.addFile(f, readFileSync(join(ORACLE, f), 'utf8'));
  await session.execute(lines.slice(0, k).join('\n'));
  session.sys.fixes.push(new FixCmap(session.sys, cm[1], cm[2], [cm[4]]));
  await session.execute(`${extra}${lines.slice(k + 1).join('\n')}\n${FINAL_DUMP}\n`);
  return { events, files };
};

describe('oracle: fix cmap against native LAMMPS', () => {
  it('w15cmap_run: thermo (pe, f_cmap, press) and per-atom state match the native fixture', async () => {
    const fx = JSON.parse(readFileSync(join(FIX, 'w15cmap_run.json'), 'utf8')) as { thermo: ThermoRow[]; atoms: Record<string, number>[] };
    const { events, files } = await runWithFix('w15cmap_run');
    const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
    expect(rows.length).toBe(fx.thermo.length);
    const close = (a: number, b: number) => Math.abs(a - b) <= 1e-10 + 1e-8 * Math.max(Math.abs(a), Math.abs(b));
    for (let r = 0; r < fx.thermo.length; r++) {
      for (const k of ['pe', 'ebond', 'eangle', 'edihed', 'f_cmap', 'press']) {
        const got = rows[r][k] as number, want = fx.thermo[r][k] as number;
        expect(close(got, want), `row ${r} ${k}: engine ${got} vs native ${want}`).toBe(true);
      }
    }
    // per-atom final state (positions, velocities, forces) against the native dump
    const dump = files.get('oracle_final.dump') ?? '';
    const dl = dump.trim().split('\n');
    const k0 = dl.findIndex((l) => l.startsWith('ITEM: ATOMS'));
    const cols = dl[k0].split(/\s+/).slice(2);
    const mine = dl.slice(k0 + 1).map((l) => {
      const w = l.trim().split(/\s+/).map(Number);
      const a: Record<string, number> = {};
      cols.forEach((c, i) => { a[c] = w[i]; });
      return a;
    });
    expect(mine.length).toBe(fx.atoms.length);
    for (let i = 0; i < mine.length; i++) {
      for (const c of ['xu', 'yu', 'zu', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz']) {
        const want = fx.atoms[i][c];
        expect(close(mine[i][c], want), `atom ${fx.atoms[i].id} ${c}: engine ${mine[i][c]} vs native ${want}`).toBe(true);
      }
    }
    // the fix's energy is a nonzero part of pe (the case is not trivially zero)
    expect(Math.abs(fx.thermo[0].f_cmap as number)).toBeGreaterThan(0.1);
  });
});

/** A session whose written files are collected in `files`. */
const newSession = () => {
  const files = new Map<string, string>();
  const session = new Session({ emit: () => {}, writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  return { session, files };
};

const DATA_PA = '# t\n\n1 atoms\n1 atom types\n\n0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\nMasses\n\n1 1.0\n\nAtoms # atomic\n\n1 1 0 0 0\n';

describe('fix_modify energy (fix_cmap.rst: energy is on by default)', () => {
  it('fix_modify cmap energy no keeps f_cmap but leaves the CMAP term out of pe (measured with native)', async () => {
    const fx = JSON.parse(readFileSync(join(FIX, 'w15cmap_run.json'), 'utf8')) as { thermo: ThermoRow[] };
    const { events } = await runWithFix('w15cmap_run', 'fix_modify cmap energy no\n');
    const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
    // the native fixture is with energy yes: at step 0 pe = ebond + eangle + edihed + f_cmap
    const pe0 = rows[0].pe as number;
    expect(Math.abs(pe0 - (rows[0].ebond as number) - (rows[0].eangle as number) - (rows[0].edihed as number))).toBeLessThan(1e-9);
    expect(Math.abs((rows[0].f_cmap as number) - (fx.thermo[0].f_cmap as number))).toBeLessThan(1e-10);
  });
});

describe('per-atom CMAP energy and virial (fix_cmap.rst: fix_modify energy yes adds it per atom)', () => {
  it('the per-atom energies (compute pe/atom fix) sum to f_cmap at every row', async () => {
    const { events } = await runWithFix('w16cmap_peratom');
    const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
    expect(rows.length).toBeGreaterThan(1);
    for (const r of rows) expect(Math.abs((r.c_tpe as number) - (r.f_cmap as number))).toBeLessThan(1e-10);
  });

  it('fix_modify cmap energy no leaves the CMAP energy out of the per-atom energies, f_cmap unchanged', async () => {
    const { events } = await runWithFix('w16cmap_peratom', 'fix_modify cmap energy no\n');
    const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
    for (const r of rows) {
      expect(r.c_tpe as number).toBe(0);
      expect(Math.abs(r.f_cmap as number)).toBeGreaterThan(0);
    }
  });
});

describe('oracle: fix cmap in the minimizer setup (fix_modify energy yes)', () => {
  it('w15cmap_minimize: the setup thermo row and the atoms match the native fixture', async () => {
    const fx = JSON.parse(readFileSync(join(FIX, 'w15cmap_minimize.json'), 'utf8')) as { thermo: ThermoRow[]; atoms: Record<string, number>[] };
    const { events, files } = await runWithFix('w15cmap_minimize');
    const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
    // the setup row (native prints one; the engine's minimizer also prints the final row: compared as the first)
    expect(rows.length).toBeGreaterThanOrEqual(1);
    const close = (a: number, b: number) => Math.abs(a - b) <= 1e-10 + 1e-8 * Math.max(Math.abs(a), Math.abs(b));
    for (const k of ['pe', 'edihed', 'f_cmap']) {
      expect(close(rows[0][k] as number, fx.thermo[0][k] as number), `${k}: engine ${rows[0][k]} vs native ${fx.thermo[0][k]}`).toBe(true);
    }
    const dl = (files.get('oracle_final.dump') ?? '').trim().split('\n');
    const k0 = dl.findIndex((l) => l.startsWith('ITEM: ATOMS'));
    const cols = dl[k0].split(/\s+/).slice(2);
    const mine = dl.slice(k0 + 1).map((l) => {
      const w = l.trim().split(/\s+/).map(Number);
      const a: Record<string, number> = {};
      cols.forEach((c, i) => { a[c] = w[i]; });
      return a;
    });
    for (let i = 0; i < mine.length; i++) {
      for (const c of ['xu', 'yu', 'zu', 'fx', 'fy', 'fz']) {
        expect(close(mine[i][c], fx.atoms[i][c]), `atom ${fx.atoms[i].id} ${c}`).toBe(true);
      }
    }
  });
});

describe('read_data / write_data plumbing for fix cmap', () => {
  const setup = (session: Session, extra: string[]) =>
    session.execute(['units real', 'atom_style full', 'boundary f f f', ...extra].join('\n'));

  it('refuses a header-string other than NULL for a fix that reads no header lines (property/atom message kept)', async () => {
    const { session } = newSession();
    session.addFile('d.data', DATA_PA);
    await setup(session, ['fix pa all property/atom mol']);
    await expect(session.execute('read_data d.data fix pa crossterm CMAP')).rejects.toThrow(/header-string must be NULL \(fix property\/atom reads no header lines\)/);
  });

  it('a NULL header gives fix cmap no crossterm count, which is refused', async () => {
    const { session } = newSession();
    session.addFile('g.txt', gridText(syntheticGrid()) + '\n');
    session.addFile('d.data', '# t\n\n1 atoms\n1 atom types\n\n0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\nMasses\n\n1 1.0\n\nAtoms # full\n\n1 1 1 0 0 0 0\n\nCMAP\n\n1 1 1 1 1 1 1\n');
    await setup(session, []);
    session.sys.fixes.push(new FixCmap(session.sys, 'cmap', 'all', ['g.txt']));
    await expect(session.execute('read_data d.data fix cmap NULL CMAP')).rejects.toThrow(/header must select the "N crossterms" line/);
  });

  it('write_data writes "N crossterms" and the CMAP section, and nofix leaves both out', async () => {
    const { session, files } = newSession();
    session.addFile('w15cmap_grid.txt', readFileSync(join(ORACLE, 'w15cmap_grid.txt'), 'utf8'));
    session.addFile('w15cmap_chain.data', readFileSync(join(ORACLE, 'w15cmap_chain.data'), 'utf8'));
    await setup(session, []);
    session.sys.fixes.push(new FixCmap(session.sys, 'cmap', 'all', ['w15cmap_grid.txt']));
    await session.execute([
      'read_data w15cmap_chain.data fix cmap crossterm CMAP',
      'bond_style harmonic', 'angle_style harmonic', 'dihedral_style harmonic', 'pair_style zero 10.0', 'pair_coeff * *',
      'bond_coeff 1 1.0 1.5', 'angle_coeff 1 1.0 111.0', 'dihedral_coeff 1 1.0 1 1',
      'write_data wd_full.data', 'write_data wd_nofix.data nofix',
    ].join('\n'));
    const full = files.get('wd_full.data') ?? '';
    const bare = files.get('wd_nofix.data') ?? '';
    expect(full).toMatch(/\n4 crossterms\n/);
    expect(full).toMatch(/\nCMAP\n\n1 1 1 2 3 4 5\n/);
    expect(bare).not.toMatch(/crossterms/);
    expect(bare).not.toMatch(/\nCMAP\n/);
  });

  it('write_data sorts the CMAP rows by their first atom and renumbers them (measured: reversed input)', async () => {
    const { session, files } = newSession();
    session.addFile('w15cmap_grid.txt', readFileSync(join(ORACLE, 'w15cmap_grid.txt'), 'utf8'));
    const src = readFileSync(join(ORACLE, 'w15cmap_chain.data'), 'utf8').split('\n');
    const at = src.indexOf('CMAP');
    const cm = src.slice(at + 2).filter((l) => l.trim()).reverse().map((l, k) => `${k + 1} ${l.trim().split(/\s+/).slice(1).join(' ')}`);
    session.addFile('rev.data', [...src.slice(0, at), 'CMAP', '', ...cm, ''].join('\n'));
    await setup(session, []);
    session.sys.fixes.push(new FixCmap(session.sys, 'cmap', 'all', ['w15cmap_grid.txt']));
    await session.execute([
      'read_data rev.data fix cmap crossterm CMAP',
      'bond_style harmonic', 'angle_style harmonic', 'dihedral_style harmonic', 'pair_style zero 10.0', 'pair_coeff * *',
      'bond_coeff 1 1.0 1.5', 'angle_coeff 1 1.0 111.0', 'dihedral_coeff 1 1.0 1 1', 'write_data wd_rev.data',
    ].join('\n'));
    const out = files.get('wd_rev.data') ?? '';
    const block = out.slice(out.indexOf('\nCMAP\n') + 7).split('\n').filter((l) => l.trim());
    expect(block.slice(0, 4).map((l) => l.split(' ')[2])).toEqual(['1', '2', '3', '4']);
    expect(block[0].split(' ').length).toBe(7);
  });
});
