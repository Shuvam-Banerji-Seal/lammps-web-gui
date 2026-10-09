import { describe, expect, it } from 'vitest';
import { autoPlan, autoThreads, deviceHints, type DeviceProfile } from '../src/engine/device';
import { EngineHost } from '../src/engine/host';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import type { FromEngine } from '../src/engine/protocol';
import { simulatedPerDay } from '../src/lammps/thermoUnits';

const dev = (over: Partial<DeviceProfile> = {}): DeviceProfile => ({
  cores: 16, memoryGB: 8, workers: true, sharedMemory: true,
  gpu: { kind: 'none', name: '', compatibility: false, maxStorageBufferMB: 0 }, ...over,
});

describe('device profile and Auto', () => {
  it('auto threads: half the cores, at most 8, one without Web Workers', () => {
    expect(autoThreads(dev({ cores: 24 }))).toBe(8);
    expect(autoThreads(dev({ cores: 8 }))).toBe(4);
    expect(autoThreads(dev({ cores: 2 }))).toBe(1);
    expect(autoThreads(dev({ cores: 1 }))).toBe(1);
    expect(autoThreads(dev({ cores: 16, workers: false }))).toBe(1);
  });

  it('auto picks a hardware GPU (keeping CPU threads for the rest), else the CPU', () => {
    expect(autoPlan(dev({ gpu: { kind: 'hardware', name: 'nvidia ampere', compatibility: true, maxStorageBufferMB: 2048 } })))
      .toMatchObject({ backend: 'webgpu', threads: 8 });
    const sw = autoPlan(dev({ gpu: { kind: 'software', name: 'google swiftshader', compatibility: false, maxStorageBufferMB: 128 } }));
    expect(sw).toMatchObject({ backend: 'cpu', threads: 8 });
    expect(sw.why).toMatch(/software/);
    expect(autoPlan(dev({ cores: 4 })).why).toMatch(/2 CPU threads of 4 cores; no WebGPU adapter/);
  });

  it('hints name what limits the device', () => {
    expect(deviceHints(dev())).toEqual([]);
    expect(deviceHints(dev({ sharedMemory: false })).join(' ')).toMatch(/not cross-origin isolated/);
    expect(deviceHints(dev({ workers: false, sharedMemory: false })).join(' ')).toMatch(/Web Workers are unavailable/);
    expect(deviceHints(dev({ cores: 2, memoryGB: 2 })).length).toBe(2);
  });

  it('the host answers reset(auto) with the device profile and the plan it used', async () => {
    const out: FromEngine[] = [];
    const host = new EngineHost((m) => out.push(m));
    await host.handle({ type: 'reset', backend: 'auto', threads: 0, frameEvery: 0 });
    const ready = out.find((m) => m.type === 'ready') as Extract<FromEngine, { type: 'ready' }>;
    expect(ready.device.cores).toBeGreaterThanOrEqual(1);
    expect(ready.auto?.backend).toBe('cpu'); // no WebGPU in the test environment
    expect(ready.kind).toBe('cpu');
    expect(ready.threads).toBeGreaterThanOrEqual(1);
  });
});

describe('run speed for the resource monitor', () => {
  it('a run reports its timestep/units and its speed (perf) at the end', async () => {
    const ev: EngineEvent[] = [];
    await new Session({ emit: (e) => ev.push(e) }).execute(`units real
lattice fcc 5.3
region b block 0 3 0 3 0 3
create_box 1 b
create_atoms 1 box
mass 1 39.95
pair_style lj/cut 8.5
pair_coeff 1 1 0.238 3.405
fix 1 all nve
timestep 2.0
run 20
`);
    const run = ev.find((e) => e.kind === 'run');
    expect(run).toMatchObject({ kind: 'run', from: 0, to: 20, dt: 2, units: 'real' });
    const perf = ev.filter((e) => e.kind === 'perf');
    expect(perf.length).toBeGreaterThanOrEqual(1);
    const last = perf[perf.length - 1] as Extract<EngineEvent, { kind: 'perf' }>;
    expect(last.step).toBe(20);
    expect(last.atoms).toBe(108);
    expect(last.stepsPerSec).toBeGreaterThan(0);
  });

  it('simulated time per day follows the units style', () => {
    expect(simulatedPerDay('real', 1, 1000)).toBe('86.4 ns/day'); // 1 fs x 1000 steps/s x 86400 s
    expect(simulatedPerDay('metal', 0.001, 100)).toBe('8.6 ns/day'); // 1 fs (0.001 ps) x 100 steps/s
    expect(simulatedPerDay('lj', 0.005, 1000)).toBe('4.3e+5 τ/day');
    expect(simulatedPerDay('real', 1, 0)).toBeNull();
  });
});
