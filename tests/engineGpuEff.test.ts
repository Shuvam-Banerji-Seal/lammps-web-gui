import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { accelerator } from '../src/engine/run/accel';
import { ATOM_KERNEL_OUT, type GpuAtomKernel } from '../src/engine/gpu/atomKernel';
import { gplHook } from '../src/engine/gplHooks';
import { ELECTRON_RFORCE } from '../src/engine/atoms';
import type { EngineEvent, ForceBackend, ForceResult, SimState, ThermoRow } from '../src/engine/types';

/*
 * pair eff/cut on the GPU path (engine-gpl/eff/gpu_eff.ts, gpu/atomKernel.ts, webgpuForces.ts computeAtomKernel): the
 * kernel's arithmetic in fp64 (each atom's row of a full list, what the C++ half loop gives that atom) gives the CPU
 * style's forces, radial electron forces, energy, compute pair terms and virial, for every pair of particle kinds the
 * style has (nuclei, electrons of both spins, fixed cores, s- and p-type pseudo-cores, limit/eradius and
 * pressure/evirials), and a run through the accelerated loop with a backend that does what the kernel does gives the CPU
 * engine's thermo. The WGSL itself is checked in Chromium (SwiftShader).
 */

/**
 * A periodic box of 5^3 lattice cells holding the kinds of the given types (spin, charge, radius): nuclei, electrons
 * (spin up in the upper half, down below), fixed cores and pseudo-cores on offset copies of the lattice.
 */
const SETUP = (pairArgs: string, kinds: { spin: number; q: number; r: number }[], extra = '') => {
  const lines = [`units electron
atom_style electron
lattice sc 2.4
region box block 0 5 0 5 0 5
create_box ${kinds.length} box`];
  kinds.forEach((k, t) => {
    const o = [0.13, 0.27, 0.41, 0.07][t] ?? 0.5;
    lines.push(`lattice sc 2.4 origin ${o} ${(o * 1.7) % 1} ${(o * 2.3) % 1}`, `create_atoms ${t + 1} box`, `mass ${t + 1} ${k.spin === 0 || k.spin >= 2 ? 1.00794 : 1.0}`,
      `set type ${t + 1} charge ${k.q}`, `set type ${t + 1} spin/electron ${k.spin}`);
    if (k.spin !== 0) lines.push(`set type ${t + 1} radius/electron ${k.r}`);
  });
  const e = kinds.findIndex((k) => k.spin === 1);
  if (e >= 0) {
    lines.push('region lower block INF INF INF INF INF 6.0 units box', 'group lower region lower', `group electrons type ${e + 1}`,
      'group down intersect lower electrons', 'set group down spin/electron -1');
  }
  lines.push(`pair_style eff/cut ${pairArgs}`, 'pair_coeff * *', extra, 'comm_modify vel yes', 'neighbor 0.5 bin', 'fix 1 all nve/eff',
    'timestep 0.001', 'thermo_style custom step pe ke etotal press', 'thermo_modify format float %.16g', 'thermo 5');
  return lines.join('\n') + '\n';
};

const NUC = { spin: 0, q: 1.0, r: 0 }, ELEC = { spin: 1, q: -1.0, r: 1.1 };
const CASES: Record<string, { args: string; kinds: { spin: number; q: number; r: number }[]; extra?: string }> = {
  'nuclei and electrons of both spins': { args: '4.0', kinds: [NUC, ELEC] },
  'pressure/evirials': { args: '4.0 pressure/evirials', kinds: [NUC, ELEC] },
  // electron radii past half the box: the restraint of limit/eradius acts
  'limit/eradius with radii past half the box': { args: '4.0 limit/eradius pressure/evirials', kinds: [NUC, { spin: 1, q: -1.0, r: 6.5 }] },
  'fixed cores': { args: '4.0', kinds: [NUC, ELEC, { spin: 2, q: 2.0, r: 0.5 }] },
  's-type pseudo-cores (Si) with electrons, fixed cores and nuclei': { args: '4.0 ecp 4 Si', kinds: [NUC, ELEC, { spin: 2, q: 2.0, r: 0.5 }, { spin: 3, q: 4.0, r: 0.6 }] },
  'p-type pseudo-cores (C)': { args: '4.0 ecp 3 C', kinds: [NUC, ELEC, { spin: 3, q: 4.0, r: 0.6 }] },
};

const session = (backend?: ForceBackend) => {
  const rows: ThermoRow[] = [], logs: string[] = [];
  const s = new Session({
    emit: (ev: EngineEvent) => { if (ev.kind === 'thermo') rows.push(ev.row); else if (ev.kind === 'log') logs.push(ev.text); },
    writeFile: () => {},
  }, backend);
  return { s, rows, logs, sys: () => (s as unknown as { sys: System }).sys };
};

const kernelOf = (sys: System): GpuAtomKernel => {
  const k = gplHook<{ gpuKernel(pair: unknown, sys: System): GpuAtomKernel | string }>('eff/cut').gpuKernel(sys.ff.pair, sys);
  if (typeof k === 'string') throw new Error(k);
  return k;
};

const rel = (a: number, b: number, scale: number) => Math.abs(a - b) / Math.max(scale, 1e-300);

/** A backend that does what computeAtomKernel does, in fp64. */
class AtomKernelInFp64 implements ForceBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'WebGPU · fp64 stand-in';
  calls = 0;
  compute(): ForceResult { throw new Error('the stand-in runs computeAtomKernel only'); }
  dispose(): void {}
  async computeAtomKernel(st: SimState, k: GpuAtomKernel): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    return k.finish(st, k.fp64(st));
  }
}

describe('pair eff/cut on the GPU path', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(`the kernel's fp64 arithmetic gives the CPU style's results: ${name}`, async () => {
      const { s, sys } = session();
      await s.execute(`${SETUP(c.args, c.kinds, c.extra)}run 0`);
      const st = sys().state;
      const n = st.n;
      const f = st.f.slice(0, 3 * n);
      const erf = st.custom.get(ELECTRON_RFORCE)!.data.slice(0, n);
      const acc = sys().ff.acc;
      const pe = acc.ecoul + acc.evdwl;
      const vir = Array.from(acc.virial);
      const pv = Float64Array.from(sys().ff.pair!.extract!('pvector') as Float64Array);
      // the restraint acts in its case, and every case has pair terms in both of its compute pair sums
      if (name.startsWith('limit/eradius')) expect(pv[3]).toBeGreaterThan(0);
      expect(Math.abs(pv[1])).toBeGreaterThan(0);
      const k = kernelOf(sys());
      const out = k.fp64(st);
      let fscale = 0, rscale = 0;
      for (let i = 0; i < 3 * n; i++) fscale = Math.max(fscale, Math.abs(f[i]));
      for (let i = 0; i < n; i++) rscale = Math.max(rscale, Math.abs(erf[i]));
      let fworst = 0, rworst = 0;
      for (let i = 0; i < n; i++) {
        for (let d = 0; d < 3; d++) fworst = Math.max(fworst, rel(out[ATOM_KERNEL_OUT * i + d], f[3 * i + d], fscale));
        rworst = Math.max(rworst, rel(out[ATOM_KERNEL_OUT * i + 3], erf[i], rscale));
      }
      expect(fworst).toBeLessThan(1e-9);
      expect(rworst).toBeLessThan(1e-9);
      const r = k.finish(st, out);
      expect(rel(r.pe, pe, Math.abs(pe))).toBeLessThan(1e-11);
      expect(r.ecoul).toBe(r.pe);
      const pv2 = sys().ff.pair!.extract!('pvector') as Float64Array;
      for (let c2 = 0; c2 < 4; c2++) expect(Math.abs(pv2[c2] - pv[c2])).toBeLessThan(1e-11 * Math.max(1, Math.abs(pe)));
      const vscale = Math.max(...vir.map(Math.abs), 1);
      for (let c2 = 0; c2 < 6; c2++) expect(Math.abs(r.virialTensor![c2] - vir[c2])).toBeLessThan(1e-10 * vscale);
      expect(rel(r.virial, vir[0] + vir[1] + vir[2], vscale)).toBeLessThan(1e-10);
    });
  }

  it('a run through the accelerated loop with the kernel in fp64 gives the CPU thermo', async () => {
    const c = CASES['s-type pseudo-cores (Si) with electrons, fixed cores and nuclei'];
    const cpu = session();
    await cpu.s.execute(`${SETUP(c.args, c.kinds)}run 20`);
    const gpu = new AtomKernelInFp64();
    const acc = session(gpu);
    await acc.s.execute(`${SETUP(c.args, c.kinds)}run 20`);
    expect(gpu.calls).toBeGreaterThan(20);
    expect(acc.rows.length).toBe(cpu.rows.length);
    for (let r = 0; r < cpu.rows.length; r++) {
      for (const key of ['pe', 'ke', 'etotal', 'press'] as const) {
        const a = Number(acc.rows[r][key]), b = Number(cpu.rows[r][key]);
        expect(Math.abs(a - b)).toBeLessThan(1e-8 * Math.max(1, Math.abs(b)));
      }
    }
  });

  it('accelerator() takes eff/cut onto the GPU, and says why a short box stays on the CPU', async () => {
    const ok = session();
    await ok.s.execute(`${SETUP('4.0', [NUC, ELEC])}run 0`);
    const a = accelerator(ok.sys(), new AtomKernelInFp64());
    expect(a.reason).toBeNull();
    expect(a.accel?.atom?.name).toBe('eff/cut');
    const short = session();
    await short.s.execute(`${SETUP('7.0', [NUC, ELEC])}run 0`);
    const b = accelerator(short.sys(), new AtomKernelInFp64());
    expect(b.accel).toBeNull();
    expect(b.reason).toMatch(/shorter than 2 pair cutoffs/);
  });
});
