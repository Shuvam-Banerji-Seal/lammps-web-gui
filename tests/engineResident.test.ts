import { describe, expect, it } from 'vitest';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { latticePoints, makeLattice } from '../src/engine/lattice';
import { newPairTable, resolvePairs, setPairCoeff } from '../src/engine/pairs';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import { FixEnforce2d, FixNve, halfKick, drift, wrapPositions, type Fix } from '../src/engine/integrate';
import { FixLangevin } from '../src/engine/thermostats';
import { addAtoms, emptyState, run, type ResidentBackend } from '../src/engine/md';
import { createVelocities } from '../src/engine/velocity';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ForceResult, PairTable, SimState, ThermoRow } from '../src/engine/types';

/*
 * The GPU-resident run path of md.run (gpu/resident.ts does the stepping in
 * WGSL; its numerics are checked against the CPU in a real browser). Here a
 * fake resident backend steps velocity Verlet in fp64 on the CPU, so the
 * chunked loop must reproduce the per-step loop exactly, and the chunk
 * boundaries — the only steps that come back to the host — can be asserted.
 */

const lj = UNIT_SYSTEMS.lj;

const meltState = (cells: number, dimension: 2 | 3 = 3): SimState => {
  const lat = makeLattice(dimension === 3 ? 'fcc' : 'sq', 0.8442, lj, dimension);
  const hi: [number, number, number] = [cells * lat.spacing[0], cells * lat.spacing[1], dimension === 3 ? cells * lat.spacing[2] : 0.5];
  const lo: [number, number, number] = [0, 0, dimension === 3 ? 0 : -0.5];
  const s = emptyState(lj, dimension, { lo, hi, periodic: [true, true, true] }, 1);
  addAtoms(s, latticePoints(lat, lo, hi, dimension), 1);
  s.massByType[1] = 1.0;
  createVelocities(s, 3.0, 87287);
  return s;
};

const meltPairs = (): PairTable => {
  const t = newPairTable(1, 2.5);
  setPairCoeff(t, '1', '1', 1.0, 1.0, 2.5);
  resolvePairs(t);
  return t;
};

/** fp64 stand-in for the WebGPU resident stepper: same step, same order as fix nve (+ enforce2d). */
class FakeResident implements ResidentBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'fake resident';
  readonly chunks: number[] = [];
  private cpu = new CpuForceBackend();
  constructor(private ok = true) {}
  compute(s: SimState, t: PairTable): Promise<ForceResult> { return Promise.resolve(this.cpu.compute(s, t)); }
  canAdvance(): boolean { return this.ok; }
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

const KEYS = ['step', 'temp', 'epair', 'etotal', 'press'] as const;

const runWith = async (backend: CpuForceBackend | FakeResident, fixes: Fix[], s: SimState, nsteps: number, opts: { thermoEvery: number; hostStep?: (k: number) => boolean; stopAt?: number }) => {
  const rows: ThermoRow[] = [];
  const hostSteps: number[] = [];
  await run(s, meltPairs(), backend, fixes, nsteps, {
    thermoEvery: opts.thermoEvery,
    keywords: KEYS,
    onThermo: (r) => rows.push(r),
    onStep: (st) => { hostSteps.push(st.step); return opts.stopAt === undefined || st.step < opts.stopAt; },
    hostStep: opts.hostStep,
  });
  return { rows, hostSteps, s };
};

describe('GPU-resident run loop (md.run with a ResidentBackend)', () => {
  it('reproduces the per-step loop exactly (thermo and final state)', async () => {
    const a = await runWith(new CpuForceBackend(), [new FixNve('1')], meltState(4), 120, { thermoEvery: 50 });
    const fake = new FakeResident();
    const b = await runWith(fake, [new FixNve('1')], meltState(4), 120, { thermoEvery: 50 });
    expect(fake.chunks.length).toBeGreaterThan(0);
    expect(a.s.n).toBe(256);
    expect(a.rows[a.rows.length - 1].temp).not.toBeCloseTo(a.rows[0].temp as number, 1);
    expect(b.rows).toEqual(a.rows);
    expect(Array.from(b.s.x)).toEqual(Array.from(a.s.x));
    expect(Array.from(b.s.v)).toEqual(Array.from(a.s.v));
    expect(Array.from(b.s.image)).toEqual(Array.from(a.s.image));
    expect(b.s.step).toBe(120);
  });

  it('comes back to the host only at thermo steps, host steps and the end', async () => {
    const fake = new FakeResident();
    const r = await runWith(fake, [new FixNve('1')], meltState(4), 120, { thermoEvery: 50, hostStep: (k) => k % 30 === 0 });
    // due: 30 50 60 90 100 120
    expect(fake.chunks).toEqual([30, 20, 10, 30, 10, 20]);
    expect(r.hostSteps).toEqual([30, 50, 60, 90, 100, 120]);
    expect(r.rows.map((x) => x.step)).toEqual([0, 50, 100, 120]);
  });

  it('caps a chunk at 200 steps so Stop stays responsive', async () => {
    const fake = new FakeResident();
    await runWith(fake, [new FixNve('1')], meltState(4), 450, { thermoEvery: 0 });
    expect(fake.chunks).toEqual([200, 200, 50]);
  });

  it('stops after the chunk in which onStep returns false', async () => {
    const fake = new FakeResident();
    const r = await runWith(fake, [new FixNve('1')], meltState(4), 300, { thermoEvery: 100, stopAt: 100 });
    expect(fake.chunks).toEqual([100]);
    expect(r.s.step).toBe(100);
  });

  it('enforce2d runs resident and matches the per-step loop in 2d', async () => {
    const fixes = () => [new FixNve('1'), new FixEnforce2d('2')];
    const a = await runWith(new CpuForceBackend(), fixes(), meltState(8, 2), 60, { thermoEvery: 20 });
    const fake = new FakeResident();
    const b = await runWith(fake, fixes(), meltState(8, 2), 60, { thermoEvery: 20 });
    expect(fake.chunks).toEqual([20, 20, 20]);
    expect(a.s.n).toBe(64);
    expect(Array.from(b.s.x).filter((_, k) => k % 3 === 2).every((z) => z === 0)).toBe(true);
    expect(b.rows).toEqual(a.rows);
    expect(Array.from(b.s.x)).toEqual(Array.from(a.s.x));
  });

  it('thermostatted runs, and systems the backend declines, use the per-step path', async () => {
    const f1 = new FakeResident();
    await runWith(f1, [new FixNve('1'), new FixLangevin('2', 1.0, 1.0, 1.0, 48279)], meltState(4), 30, { thermoEvery: 10 });
    expect(f1.chunks).toEqual([]);
    const f2 = new FakeResident(false);
    await runWith(f2, [new FixNve('1')], meltState(4), 30, { thermoEvery: 10 });
    expect(f2.chunks).toEqual([]);
    const f3 = new FakeResident();
    await runWith(f3, [], meltState(4), 30, { thermoEvery: 10 });
    expect(f3.chunks).toEqual([]);
  });

  it('through the interpreter: a thermostatted run uses accelerated forces with every fix, matching the fp64 engine', async () => {
    const script = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 3.0 87287
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nvt temp 1.5 1.5 0.5
thermo_style custom step temp pe etotal press
thermo 10
run 60`;
    const go = async (backend?: FakeResident) => {
      const rows: ThermoRow[] = [];
      const logs: string[] = [];
      const session = new Session({
        emit: (ev: EngineEvent) => { if (ev.kind === 'thermo') rows.push(ev.row); if (ev.kind === 'log') logs.push(ev.text); },
        writeFile: () => {},
      }, backend);
      await session.execute(script);
      return { rows, logs };
    };
    const fake = new FakeResident();
    const acc = await go(fake);
    const ref = await go();
    // per-step accelerated forces (no resident chunks: fix nvt integrates on the host)
    expect(fake.chunks).toEqual([]);
    expect(acc.logs.some((l) => /general fp64/.test(l))).toBe(false);
    expect(acc.rows.length).toBe(ref.rows.length);
    for (let r = 0; r < ref.rows.length; r++) {
      for (const k of ['temp', 'pe', 'etotal', 'press']) expect(acc.rows[r][k]).toBeCloseTo(ref.rows[r][k], 9);
    }
  });

  it('through the interpreter: dumps and viewer frames land on their steps', async () => {
    const fake = new FakeResident();
    const frames: number[] = [];
    const files = new Map<string, string>();
    const session = new Session({
      emit: (ev: EngineEvent) => { if (ev.kind === 'frame') frames.push(ev.step); },
      writeFile: (name, body, append) => files.set(name, (append ? files.get(name) ?? '' : '') + body),
    }, fake, 40);
    await session.execute(`units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 3.0 87287
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
thermo 50
dump d all atom 30 m.dump
run 100`);
    const dumpSteps = (files.get('m.dump') ?? '').split('ITEM: TIMESTEP\n').slice(1).map((t) => Number(t.split('\n')[0]));
    expect(dumpSteps).toEqual([0, 30, 60, 90]);
    // due: 30 40 50 60 80 90 100
    expect(fake.chunks).toEqual([30, 10, 10, 10, 20, 10, 10]);
    expect(frames.filter((k) => k % 40 === 0 && k > 0)).toEqual([40, 80]);
    expect(frames[frames.length - 1]).toBe(100);
  });
});
