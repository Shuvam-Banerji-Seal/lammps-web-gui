import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { accelerator } from '../src/engine/run/accel';
import type { GpuAtomKernel } from '../src/engine/gpu/atomKernel';
import { gplHook } from '../src/engine/gplHooks';
import type { EngineEvent, ForceBackend, ForceResult, SimState, ThermoRow } from '../src/engine/types';

/*
 * pair gran/hooke, gran/hooke/history and gran/hertz/history on the GPU path (engine-gpl/granular/gpu_gran.ts,
 * gpu/atomKernel.ts): the kernel's arithmetic in fp64 (each atom evaluates its contacts in the frame of the lower atom
 * ID) gives the CPU style's forces, torques, virial and updated shear history; the history goes from the CPU pair to the
 * kernel and back; a contact table that overflows makes the evaluation run again with a larger one; and a run through
 * the accelerated loop with a backend that does what the kernel does gives the CPU engine's run. The WGSL itself is
 * checked in Chromium (SwiftShader).
 */

/**
 * A periodic box of 216 polydisperse spheres on a simple cubic lattice of spacing 1 (diameters 0.95 to 1.12, so most
 * neighbors overlap), with random velocities and spins; extra adds commands before the run.
 */
const SETUP = (style: string, extra = '') => `units lj
atom_style sphere
boundary p p p
lattice sc 1.0
region box block 0 6 0 6 0 6
create_box 1 box
create_atoms 1 box
variable d atom 0.95+0.17*random(0,1,4321)
set group all diameter v_d
set group all density 1.0
variable wx atom random(-1,1,77)
variable wy atom random(-1,1,78)
variable wz atom random(-1,1,79)
set group all omega v_wx v_wy v_wz
velocity all create 0.5 4928459 loop geom
pair_style ${style}
pair_coeff * *
comm_modify vel yes
neighbor 0.2 bin
neigh_modify every 1 delay 0
fix 1 all nve/sphere
timestep 0.0005
compute er all erotate/sphere
thermo_style custom step ke c_er press pxx pxy pyz
thermo_modify format float %.16g
thermo 10
${extra}
`;

const STYLES: Record<string, string> = {
  'gran/hooke': 'gran/hooke 2000.0 NULL 50.0 NULL 0.5 1',
  'gran/hooke/history': 'gran/hooke/history 2000.0 NULL 50.0 NULL 0.5 1',
  'gran/hertz/history': 'gran/hertz/history 2000.0 NULL 50.0 NULL 0.5 1',
  'gran/hertz/history limit_damping, a frozen group': 'gran/hertz/history 2000.0 NULL 50.0 NULL 0.5 1 limit_damping',
};
// a checkerboard of frozen spheres: no two touch (a contact of two frozen spheres moves nothing, and its m_eff is the
// mass of the lower ID's sphere on the GPU, native's follows the list orientation, gpu_gran.ts)
const FROZEN = 'variable cb atom (round(x)+round(y)+round(z))%2==0\ngroup low variable cb\nfix fz low freeze';

const session = (backend?: ForceBackend) => {
  const rows: ThermoRow[] = [];
  const s = new Session({
    emit: (ev: EngineEvent) => { if (ev.kind === 'thermo') rows.push(ev.row); },
    writeFile: () => {},
  }, backend);
  return { s, rows, sys: () => (s as unknown as { sys: System }).sys };
};

type GranPair = {
  historyContacts(): { lo: Int32Array; hi: Int32Array; shear: Float64Array; n: number };
  setHistoryContacts(lo: Int32Array, hi: Int32Array, shear: Float64Array): void;
};

const kernelOf = (sys: System): GpuAtomKernel => {
  const k = gplHook<{ gpuKernel(pair: unknown, sys: System): GpuAtomKernel | string }>(sys.ff.pair!.name).gpuKernel(sys.ff.pair, sys);
  if (typeof k === 'string') throw new Error(k);
  return k;
};

/** What computeAtomKernel does, in fp64: the kernel again while it asks for it, then finish(). */
const runKernel = (st: SimState, k: GpuAtomKernel): ForceResult & { evdwl: number; ecoul: number } => {
  for (let t = 0; t < 8; t++) {
    const out = k.fp64(st);
    if (!k.again?.(st, out)) return k.finish(st, out);
  }
  throw new Error('again 8 times');
};

class AtomKernelInFp64 implements ForceBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'WebGPU · fp64 stand-in';
  calls = 0;
  compute(): ForceResult { throw new Error('the stand-in runs computeAtomKernel only'); }
  dispose(): void {}
  async computeAtomKernel(st: SimState, k: GpuAtomKernel): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    return runKernel(st, k);
  }
}

const byPair = (c: { lo: Int32Array; hi: Int32Array; shear: Float64Array; n: number }) => {
  const m = new Map<string, number[]>();
  for (let k = 0; k < c.n; k++) m.set(`${c.lo[k]}:${c.hi[k]}`, [c.shear[3 * k], c.shear[3 * k + 1], c.shear[3 * k + 2]]);
  return m;
};

describe('pair gran on the GPU path', () => {
  for (const [name, style] of Object.entries(STYLES)) {
    it(`the kernel's fp64 arithmetic gives the CPU style's forces, torques, virial and history: ${name}`, async () => {
      const extra = name.includes('frozen') ? FROZEN : '';
      const { s, sys } = session();
      // a run builds the shear history; then the setup evaluation of run 0 (no shear update) is the reference
      await s.execute(`${SETUP(style, extra)}run 40`);
      const pair = sys().ff.pair as unknown as GranPair;
      const history = style.includes('history');
      const h0 = pair.historyContacts();
      if (history) expect(h0.n).toBeGreaterThan(200);
      await s.execute('run 0');
      const st = sys().state;
      const n = st.n;
      const f = st.f.slice(0, 3 * n), tq = st.torque!.slice(0, 3 * n);
      const vir = Array.from(sys().ff.acc.virial);
      const h1 = byPair(pair.historyContacts());
      // the kernel from the history before that evaluation
      pair.setHistoryContacts(h0.lo.slice(), h0.hi.slice(), h0.shear.slice());
      const k = kernelOf(sys());
      k.begin!(st, false);
      const r = runKernel(st, k);
      // fix freeze zeroes the force and torque of its atoms after the pair (the CPU reference has them zeroed)
      const fz = sys().fixes.find((x) => x.id === 'fz');
      const frozen = (i: number) => !!fz && (st.mask[i] & sys().groups.bit('low')) !== 0;
      if (fz) expect(Array.from({ length: n }, (_, i) => frozen(i)).filter(Boolean).length).toBeGreaterThan(10);
      let fs = 0, ts = 0;
      for (let i = 0; i < 3 * n; i++) { fs = Math.max(fs, Math.abs(f[i])); ts = Math.max(ts, Math.abs(tq[i])); }
      expect(fs).toBeGreaterThan(0);
      if (name !== 'gran/hooke') expect(ts).toBeGreaterThan(0);
      for (let i = 0; i < 3 * n; i++) {
        if (frozen((i / 3) | 0)) continue;
        expect(Math.abs(st.f[i] - f[i])).toBeLessThan(1e-10 * fs);
        expect(Math.abs(st.torque![i] - tq[i])).toBeLessThan(1e-10 * Math.max(ts, 1e-300));
      }
      const vs = Math.max(...vir.map(Math.abs));
      for (let c = 0; c < 6; c++) expect(Math.abs(r.virialTensor![c] - vir[c])).toBeLessThan(1e-10 * vs);
      if (history) {
        const h2 = byPair(pair.historyContacts());
        expect(h2.size).toBe(h1.size);
        let ss = 0;
        for (const v of h1.values()) for (const x of v) ss = Math.max(ss, Math.abs(x));
        for (const [key, v] of h1) {
          const w = h2.get(key);
          expect(w, key).toBeDefined();
          for (let c = 0; c < 3; c++) expect(Math.abs(w![c] - v[c])).toBeLessThan(1e-10 * ss);
        }
      }
    });
  }

  it('a table with too few entries makes the kernel run again with a larger one', async () => {
    // a sphere of diameter 2.2 among the unit spheres touches more than the first table length (12)
    const { s, sys } = session();
    await s.execute(`${SETUP(STYLES['gran/hooke/history'])}set atom 86 diameter 2.2\nrun 5`);
    const st = sys().state;
    const k = kernelOf(sys());
    // without a stored history the first table has 12 entries; the big sphere then has 18 contacts
    const pair = sys().ff.pair as unknown as GranPair;
    pair.setHistoryContacts(new Int32Array(0), new Int32Array(0), new Float64Array(0));
    k.begin!(st, true);
    const w0 = k.extWidth!;
    expect(w0).toBe(3 + 12);
    const out = k.fp64(st);
    expect(k.again!(st, out)).toBe(true);
    expect(k.extWidth!).toBeGreaterThan(w0);
    expect(k.again!(st, k.fp64(st))).toBe(false);
  });

  it('runs through the accelerated loop give the CPU run, with the history carried from a CPU run and back', async () => {
    for (const style of [STYLES['gran/hooke/history'], STYLES['gran/hertz/history']]) {
      const input = `${SETUP(style)}set atom 86 diameter 2.2\n`;
      const cpu = session();
      await cpu.s.execute(`${input}run 30\nrun 60\nrun 20`);
      const gpu = new AtomKernelInFp64();
      const acc = session(gpu);
      // the first and last runs on the CPU (a compute that needs per-atom virials keeps a run there), the middle one
      // through the accelerated loop
      await acc.s.execute(`${input}compute sa all stress/atom NULL\nrun 30\nuncompute sa\nrun 60\ncompute sa all stress/atom NULL\nrun 20`);
      expect(gpu.calls).toBeGreaterThan(60);
      expect(acc.rows.length).toBe(cpu.rows.length);
      for (let r = 0; r < cpu.rows.length; r++) {
        for (const key of ['ke', 'c_er', 'press', 'pxx', 'pxy', 'pyz'] as const) {
          const a = Number(acc.rows[r][key]), b = Number(cpu.rows[r][key]);
          expect(Math.abs(a - b), `${style} row ${r} ${key}`).toBeLessThan(1e-8 * Math.max(1, Math.abs(b)));
        }
      }
    }
  });

  it('accelerator() takes the gran pairs onto the GPU, and says why special bonds or a short box keep them on the CPU', async () => {
    for (const style of Object.values(STYLES)) {
      const ok = session();
      await ok.s.execute(`${SETUP(style)}run 0`);
      const a = accelerator(ok.sys(), new AtomKernelInFp64());
      expect(a.reason).toBeNull();
      expect(a.accel?.atom?.name).toBe(style.split(' ')[0]);
    }
    const short = session();
    await short.s.execute(`${SETUP(STYLES['gran/hooke/history'])}set atom 86 diameter 3.2\nrun 0`);
    const b = accelerator(short.sys(), new AtomKernelInFp64());
    expect(b.accel).toBeNull();
    expect(b.reason).toMatch(/shorter than 2 pair cutoffs/);
  });
});
