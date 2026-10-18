import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, copyFileSync, writeFileSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { ACCEL_ALIASES } from '../src/engine/accelAliases';
import { Session } from '../src/engine/interpreter';
import { accelerator } from '../src/engine/run/accel';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import { SharedThreadsBackend } from '../src/engine/cpu/pairThreads';
import { PairThreadWorker, type PairThreadMessage, type PairWorkerLike } from '../src/engine/cpu/pairThreadsCore';
import { isResident, type ResidentBackend } from '../src/engine/md';
import { halfKick, drift, wrapPositions } from '../src/engine/integrate';
import type { EngineEvent, ForceResult, PairTable, SimState, ThermoRow } from '../src/engine/types';
import { buildMap, readAliases } from '../scripts/gen-accel-aliases.mjs';

/*
 * Accelerator-suffix aliases (src/engine/accelAliases.ts, generated from the LAMMPS index pages by
 * scripts/gen-accel-aliases.mjs). An alias is a name only: the session replaces it by its base style
 * before anything is created, so an input written for native LAMMPS with lj/cut/omp, nve/omp, suffix omp
 * or package omp runs the same engine path as the plain input, and the fast paths (accel.ts, the threaded
 * pair term) see the base style.
 */

const DOC_SRC_CANDIDATES = [process.env.LAMMPS_DOC_SRC, '/store/shuvam/lammps_github/doc/src'].filter((p): p is string => !!p);
const DOC_SRC = DOC_SRC_CANDIDATES.find((d) => existsSync(join(d, 'Commands_pair.rst')));
const SCRIPT = resolve(__dirname, '..', 'scripts', 'gen-accel-aliases.mjs');

interface Run { events: EngineEvent[]; error: Error | null; session: Session }

/** Runs input text on a session (optionally with a force backend); keeps the session for inspection. */
const runScript = async (text: string, backend?: CpuForceBackend | ResidentBackend | SharedThreadsBackend): Promise<Run> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => undefined }, backend);
  let error: Error | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as Error;
  }
  return { events, error, session };
};

const thermo = (events: EngineEvent[]): ThermoRow[] =>
  events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);

const logs = (events: EngineEvent[]): string[] =>
  events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);

const MELT_HEAD = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.44 87287 loop geom
`;

const MELT_TAIL = `neighbor 0.3 bin
thermo_style custom step temp pe ke etotal press
thermo_modify format float %.17g
thermo 10
run 50
`;

/** Melt with the given style names (pair, fix) and optional prelude lines. */
const melt = (pair: string, fix: string, prelude = '') =>
  `${prelude}${MELT_HEAD}pair_style ${pair} 2.5\npair_coeff 1 1 1.0 1.0 2.5\nfix 1 all ${fix}\n${MELT_TAIL}`;

describe('the generated alias map (src/engine/accelAliases.ts)', () => {
  it('maps the accelerated names of the index pages to their base styles', () => {
    expect(ACCEL_ALIASES.get('lj/cut/omp')).toBe('lj/cut');
    expect(ACCEL_ALIASES.get('lj/cut/coul/long/opt')).toBe('lj/cut/coul/long');
    expect(ACCEL_ALIASES.get('nve/kk')).toBe('nve');
    expect(ACCEL_ALIASES.get('nve/omp')).toBe('nve');
    expect(ACCEL_ALIASES.get('pppm/gpu')).toBe('pppm');
    expect(ACCEL_ALIASES.get('pppm/omp')).toBe('pppm');
    expect(ACCEL_ALIASES.get('harmonic/omp')).toBe('harmonic');
    expect(ACCEL_ALIASES.get('table/omp')).toBe('table');
    expect(ACCEL_ALIASES.get('hybrid/kk')).toBe('hybrid');
    expect(ACCEL_ALIASES.get('eam/intel')).toBe('eam');
  });

  it('keeps the letters of the index entries apart: (gko) gives gpu, kk and omp, and no opt', () => {
    // Commands_pair.rst: "* :doc:`table (gko) <pair_table>`"
    expect(['table/gpu', 'table/kk', 'table/omp'].every((n) => ACCEL_ALIASES.get(n) === 'table')).toBe(true);
    expect(ACCEL_ALIASES.has('table/opt')).toBe(false);
    // Commands_pair.rst: "* :doc:`aip/water/2dm (t) <pair_aip_water_2dm>`" (only the opt package)
    expect([...ACCEL_ALIASES.keys()].filter((n) => n.startsWith('aip/water/2dm/'))).toEqual(['aip/water/2dm/opt']);
  });

  it('gives every alias a suffix that is an accelerator name and a base that carries none', () => {
    for (const [alias, base] of ACCEL_ALIASES) {
      expect(alias).toMatch(/\/(gpu|intel|kk|omp|opt)$/);
      expect(alias.slice(0, alias.lastIndexOf('/'))).toBe(base); // alias = base + "/" + suffix
      expect(base).not.toMatch(/\/(gpu|intel|kk|omp|opt)$/);
    }
    expect(ACCEL_ALIASES.size).toBeGreaterThan(600);
  });

  it('regenerating from the docs gives the same file', () => {
    if (!DOC_SRC) return; // the LAMMPS docs (doc/src of the native binary) are not in this checkout
    const dir = mkdtempSync(join(tmpdir(), 'accel-regen-'));
    try {
      const out = join(dir, 'accelAliases.ts');
      execFileSync(process.execPath, [SCRIPT, DOC_SRC, out], { stdio: 'pipe' });
      expect(readFileSync(out, 'utf8')).toBe(readFileSync(resolve(__dirname, '..', 'src', 'engine', 'accelAliases.ts'), 'utf8'));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('counts the aliases of every kind from the index pages (spot checks against the .rst entries)', () => {
    if (!DOC_SRC) return;
    const rows = readAliases(DOC_SRC);
    const kinds = new Set(rows.map((r) => r.kind));
    expect([...kinds].sort()).toEqual(['angle', 'bond', 'compute', 'dihedral', 'fix', 'improper', 'kspace', 'pair']);
    // Commands_bond.rst: "* :doc:`harmonic (iko) <bond_harmonic>`", and the same letters for angle, dihedral and improper
    expect(rows.filter((r) => r.alias === 'harmonic/omp').map((r) => r.kind).sort()).toEqual(['angle', 'bond', 'dihedral', 'improper']);
    // Commands_fix.rst entries with (...) letters become fix aliases
    expect(rows.some((r) => r.kind === 'fix' && r.alias === 'nve/omp')).toBe(true);
  });

  it('refuses a conflict between two bases of one alias, and an unknown accelerator letter', () => {
    if (!DOC_SRC) return;
    expect(() => buildMap([
      { alias: 'x/omp', base: 'x', kind: 'pair', letter: 'o', index: 'a' },
      { alias: 'x/omp', base: 'y', kind: 'fix', letter: 'o', index: 'b' },
    ])).toThrow(/x\/omp maps to two base styles/);
    // the same base from two kinds is one entry
    expect(buildMap([
      { alias: 'x/omp', base: 'x', kind: 'pair', letter: 'o', index: 'a' },
      { alias: 'x/omp', base: 'x', kind: 'fix', letter: 'o', index: 'b' },
    ]).size).toBe(1);
    const dir = mkdtempSync(join(tmpdir(), 'accel-bad-'));
    try {
      const docs = join(dir, 'doc');
      mkdirSync(docs);
      for (const f of ['Commands_pair.rst', 'Commands_fix.rst', 'Commands_compute.rst', 'Commands_bond.rst', 'Commands_kspace.rst']) {
        copyFileSync(join(DOC_SRC, f), join(docs, f));
      }
      writeFileSync(join(docs, 'Commands_pair.rst'), readFileSync(join(docs, 'Commands_pair.rst'), 'utf8').replace('* :doc:`table (gko) <pair_table>`', '* :doc:`table (gkqo) <pair_table>`'));
      expect(() => readAliases(docs)).toThrow(/unknown accelerator letter 'q'/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('an alias runs as its base style, bit for bit', () => {
  it('lj/cut/omp with nve/omp gives the same thermo as the plain input', async () => {
    const plain = await runScript(melt('lj/cut', 'nve'));
    const aliased = await runScript(melt('lj/cut/omp', 'nve/omp'));
    expect(plain.error).toBeNull();
    expect(aliased.error).toBeNull();
    expect(thermo(aliased.events).length).toBe(6);
    expect(thermo(aliased.events)).toEqual(thermo(plain.events));
  });

  it('suffix omp gives the same thermo as the plain input', async () => {
    const plain = await runScript(melt('lj/cut', 'nve'));
    const suffixed = await runScript(`suffix omp\npackage omp 4\n${melt('lj/cut', 'nve')}`);
    expect(suffixed.error).toBeNull();
    expect(thermo(suffixed.events)).toEqual(thermo(plain.events));
    expect(logs(suffixed.events).some((t) => /^suffix omp: accepted; styles named later run as their base styles/.test(t))).toBe(true);
  });

  it('package omp 4 with the explicit omp styles gives the same thermo as the plain input', async () => {
    const plain = await runScript(melt('lj/cut', 'nve'));
    const pkg = await runScript(melt('lj/cut/omp', 'nve/omp', 'package omp 4\n'));
    expect(pkg.error).toBeNull();
    expect(thermo(pkg.events)).toEqual(thermo(plain.events));
    expect(logs(pkg.events).some((t) => /^package omp 4: accepted/.test(t))).toBe(true);
  });

  it('logs each accelerated name once, saying what it runs as', async () => {
    const r = await runScript(`${MELT_HEAD}pair_style lj/cut/omp 2.5\npair_coeff 1 1 1.0 1.0 2.5\nfix 1 all nve/omp\nrun 0\npair_style lj/cut/omp 2.5\npair_coeff 1 1 1.0 1.0 2.5\nrun 0\n`);
    expect(r.error).toBeNull();
    const notes = logs(r.events).filter((t) => t.startsWith('lj/cut/omp:') || t.startsWith('nve/omp:'));
    expect(notes).toEqual([
      'lj/cut/omp: runs as lj/cut (the browser engine picks CPU threads or the GPU itself)',
      'nve/omp: runs as nve (the browser engine picks CPU threads or the GPU itself)',
    ]);
  });

  it('a hybrid pair style with accelerated sub-styles runs the base sub-styles', async () => {
    const hybridLines = (names: [string, string]) => `${MELT_HEAD}pair_style hybrid ${names[0]} 2.5 ${names[1]} 3.0
pair_coeff 1 1 ${names[0]} 1.0 1.0 2.5
pair_coeff 1 1 ${names[1]} 1.0 2.0 1.1 2.5
fix 1 all nve
${MELT_TAIL}`;
    const plain = await runScript(hybridLines(['lj/cut', 'morse']));
    const aliased = await runScript(hybridLines(['lj/cut/omp', 'morse/opt']));
    expect(plain.error).toBeNull();
    expect(aliased.error).toBeNull();
    expect(thermo(aliased.events)).toEqual(thermo(plain.events));
    expect(thermo(aliased.events).length).toBe(6);
  });

  it('accepts suffix hybrid with two suffixes', async () => {
    const r = await runScript(`suffix hybrid intel omp\n${melt('lj/cut', 'nve')}`);
    expect(r.error).toBeNull();
    expect(logs(r.events).some((t) => /^suffix hybrid intel omp: accepted/.test(t))).toBe(true);
  });
});

/** fp64 stand-in for the WebGPU resident stepper: the same step, in the same order (as engineResident.test.ts). */
class FakeGpu implements ResidentBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'fake WebGPU';
  readonly chunks: number[] = [];
  private cpu = new CpuForceBackend();
  compute(s: SimState, t: PairTable): ForceResult { return this.cpu.compute(s, t); }
  canAdvance(): boolean { return true; }
  async advance(s: SimState, t: PairTable, nsteps: number, o: { enforce2d: boolean }): Promise<ForceResult> {
    this.chunks.push(nsteps);
    let res: ForceResult = { pe: 0, virial: 0 };
    for (let k = 0; k < nsteps; k++) {
      halfKick(s);
      drift(s);
      wrapPositions(s);
      res = this.cpu.compute(s, t);
      if (o.enforce2d) for (let i = 0; i < s.n; i++) { s.v[3 * i + 2] = 0; s.f[3 * i + 2] = 0; }
      halfKick(s);
    }
    return res;
  }
  dispose(): void {}
}

describe('the fast paths see the base style', () => {
  it('accelerator() chooses the accelerated path for lj/cut/omp exactly as for lj/cut', async () => {
    const gpu = new FakeGpu();
    const plain = await runScript(melt('lj/cut', 'nve'), gpu);
    const aliased = await runScript(melt('lj/cut/omp', 'nve/omp'), gpu);
    expect(plain.error).toBeNull();
    expect(aliased.error).toBeNull();
    const a = accelerator(plain.session.sys, gpu);
    const b = accelerator(aliased.session.sys, gpu);
    expect(a.reason).toBeNull();
    expect(b.reason).toBeNull();
    expect(a.accel).not.toBeNull();
    expect(b.accel).not.toBeNull();
    expect(b.accel!.resident).toBe(a.accel!.resident);
    expect(b.accel!.resident).toBe(true); // nve only, on all atoms: whole steps stay on the GPU
    // the run itself took the accelerated path, with the same thermo as the plain input
    expect(aliased.events.some((e) => e.kind === 'log' && /general fp64 CPU engine/.test(e.text))).toBe(false);
    expect(aliased.events.find((e) => e.kind === 'done')).toMatchObject({ backend: 'fake WebGPU' });
    expect(gpu.chunks.length).toBeGreaterThan(0);
    expect(thermo(aliased.events)).toEqual(thermo(plain.events));
  });

  it('the threaded pair path accepts an aliased lj/cut/coul/long/omp', async () => {
    const workers: InProcessWorker[] = [];
    const threads = () => new SharedThreadsBackend(4, () => { const w = new InProcessWorker(); workers.push(w); return w; });
    const charged = (pair: string, kspace: string) => `${CHARGED_HEAD}pair_style ${pair} 2.5
pair_coeff * * 1.0 1.0
kspace_style ${kspace}
fix 1 all nve
thermo_style custom step pe evdwl ecoul elong press
thermo_modify format float %.17g
thermo 10
run 20
`;
    const plainBackend = threads();
    plainBackend.pairThreads.minAtoms = 1;
    const plain = await runScript(charged('lj/cut/coul/long', 'ewald 1e-4'), plainBackend);
    plainBackend.dispose();
    const plainComputes = workers.reduce((s, w) => s + w.computes, 0);
    workers.length = 0;
    const aliasBackend = threads();
    aliasBackend.pairThreads.minAtoms = 1;
    const aliased = await runScript(charged('lj/cut/coul/long/omp', 'ewald/omp 1e-4'), aliasBackend);
    aliasBackend.dispose();
    expect(plain.error).toBeNull();
    expect(aliased.error).toBeNull();
    const computes = workers.reduce((s, w) => s + w.computes, 0);
    expect(plainComputes).toBeGreaterThan(0);
    expect(computes).toBe(plainComputes);
    expect(thermo(aliased.events)).toEqual(thermo(plain.events));
  });
});

const CHARGED_HEAD = `units lj
atom_style charge
lattice fcc 0.8442
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
mass 1 1.0
set type 1 charge 0.3
velocity all create 1.4 87287 loop geom
`;

class InProcessWorker implements PairWorkerLike {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  readonly worker = new PairThreadWorker();
  computes = 0;
  postMessage(msg: unknown): void {
    const m = msg as PairThreadMessage;
    if (m.type === 'compute') this.computes++;
    this.worker.handle(m);
  }
  terminate(): void {}
}

describe('suffix, package and aliased names: errors', () => {
  it('refuses an unknown suffix, naming the allowed styles', async () => {
    const r = await runScript('suffix bogus\n');
    expect(r.error?.message).toMatch(/suffix style must be off, on, gpu, intel, kk, omp, opt or hybrid \(got 'bogus'\)/);
  });

  it('refuses a suffix hybrid without two suffixes, and arguments on a plain suffix', async () => {
    expect((await runScript('suffix hybrid omp\n')).error?.message).toMatch(/usage: suffix hybrid/);
    expect((await runScript('suffix hybrid omp bogus\n')).error?.message).toMatch(/usage: suffix hybrid/);
    expect((await runScript('suffix omp fast\n')).error?.message).toMatch(/suffix omp takes no arguments/);
  });

  it('refuses an unknown package style', async () => {
    const r = await runScript('package bogus 4\n');
    expect(r.error?.message).toMatch(/package style must be gpu, intel, kokkos or omp \(got 'bogus'\)/);
    expect((r.error as { command?: string } | null)?.command).toBe('package');
  });

  it('names the base style when the base is not supported', async () => {
    // pace (ML-PACE) needs the external PACE library, so its KOKKOS alias stays an example of an unsupported base
    const r = await runScript(`${MELT_HEAD}pair_style pace/kk\n`);
    expect(r.error).not.toBeNull();
    expect(r.error!.message).toMatch(/pace/);
    expect(r.error!.message).not.toMatch(/pace\/kk/);
  });
});

describe('a speed check on the general engine (4000-atom melt, 100 steps)', () => {
  const BIG = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.44 87287 loop geom
neighbor 0.3 bin
`;

  /** Steps per second of run 100 after a setup that is not timed. */
  const stepsPerSecond = async (pair: string, fix: string): Promise<number> => {
    const session = new Session({ emit: () => undefined, writeFile: () => undefined });
    await session.execute(`${BIG}pair_style ${pair} 2.5\npair_coeff 1 1 1.0 1.0 2.5\nfix 1 all ${fix}\nthermo 100\n`);
    const t0 = performance.now();
    await session.execute('run 100\n');
    return 100 / ((performance.now() - t0) / 1000);
  };

  // wall-clock timing is noisy on shared CI runners: opt-in (BENCH=1); the tests above prove the alias takes the same path
  it.skipIf(!process.env.BENCH)('lj/cut and lj/cut/omp agree within 10%', { timeout: 300_000 }, async () => {
    const plain: number[] = [];
    const aliased: number[] = [];
    // interleaved repetitions, best of three each, so a slow moment hits both
    for (let rep = 0; rep < 3; rep++) {
      plain.push(await stepsPerSecond('lj/cut', 'nve'));
      aliased.push(await stepsPerSecond('lj/cut/omp', 'nve/omp'));
    }
    const best = (xs: number[]) => Math.max(...xs);
    const p = best(plain), a = best(aliased);
    console.log(`speed (4000 atoms, 100 steps, best of 3): lj/cut ${p.toFixed(1)} steps/s, lj/cut/omp ${a.toFixed(1)} steps/s`);
    expect(Math.abs(a - p) / p).toBeLessThanOrEqual(0.1);
  });
});
