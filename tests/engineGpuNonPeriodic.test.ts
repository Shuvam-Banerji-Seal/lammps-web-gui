import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { pairSpecEval, type GpuPairSpec } from '../src/engine/gpu/pairSpec';
import { newCellScratch, NO_IMAGE, sortByCell } from '../src/engine/gpu/cellSort';
import { periodicCell } from '../src/engine/cpu/cells';
import { accelerator } from '../src/engine/run/accel';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import { newPairTable } from '../src/engine/pairs';
import type { EngineEvent, ForceBackend, ForceResult, PairTable, SimState, ThermoRow } from '../src/engine/types';

/*
 * Non-periodic boxes (boundary f, s, m) on the GPU path. The kernels take no image along a non-periodic dimension
 * (gpu/cellSort.ts: its box length there is NO_IMAGE, and the atoms past an edge are binned in the edge cell), and the
 * accelerated loop (run/accel.ts) reneighbors on the CPU path's steps: the periodic wrap, the reset of a shrink-wrapped
 * box and the lost atoms of a fixed boundary come there, and the atoms move on unwrapped in between.
 *
 * A backend that does what computeSpec does in fp64 (pairSpecEval, the minimum image along the periodic dimensions)
 * gives the CPU engine's thermo, atom count, build count and final positions through the accelerated loop. The cell
 * scan of the styles kernel, written out in TypeScript on the real cell sort, finds every pair within the cutoff with
 * atoms past the edges of the box.
 */

class SpecInFp64 implements ForceBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'WebGPU · fp64 stand-in';
  calls = 0;
  private cpu = new CpuForceBackend();
  /** Plain lj/cut in a fully periodic box (the lj/cut kernel): the fp64 cell list, binned as the GPU's. */
  compute(st: SimState, t: PairTable): ForceResult {
    this.calls++;
    return this.cpu.compute(st, t);
  }
  dispose(): void {}
  async computeSpec(st: SimState, spec: GpuPairSpec): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const n = st.n, L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
    const per = st.box.periodic;
    st.f.fill(0, 0, 3 * n);
    let evdwl = 0, ecoul = 0, virial = 0;
    const vt = [0, 0, 0, 0, 0, 0];
    const d = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        for (let k = 0; k < 3; k++) {
          const v = st.x[3 * i + k] - st.x[3 * j + k];
          d[k] = per[k] ? v - L[k] * Math.round(v / L[k]) : v;
        }
        if (st.dimension === 2) d[2] = 0;
        const rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        const r = pairSpecEval(spec, st.type[i], st.type[j], rsq, spec.charges ? st.q[i] : 0, spec.charges ? st.q[j] : 0);
        if (r.fpair === 0 && r.evdwl === 0 && r.ecoul === 0) continue;
        evdwl += r.evdwl; ecoul += r.ecoul; virial += r.fpair * rsq;
        vt[0] += r.fpair * d[0] * d[0]; vt[1] += r.fpair * d[1] * d[1]; vt[2] += r.fpair * d[2] * d[2];
        vt[3] += r.fpair * d[0] * d[1]; vt[4] += r.fpair * d[0] * d[2]; vt[5] += r.fpair * d[1] * d[2];
        for (let k = 0; k < 3; k++) { st.f[3 * i + k] += r.fpair * d[k]; st.f[3 * j + k] -= r.fpair * d[k]; }
      }
    }
    return { pe: evdwl + ecoul, virial, evdwl, ecoul, virialTensor: vt };
  }
}

const session = (backend?: ForceBackend) => {
  const rows: ThermoRow[] = [], logs: string[] = [];
  const s = new Session({
    emit: (ev: EngineEvent) => { if (ev.kind === 'thermo') rows.push(ev.row); else if (ev.kind === 'log') logs.push(ev.text); },
    writeFile: () => {},
  }, backend);
  return { s, rows, logs, sys: () => (s as unknown as { sys: System }).sys };
};

const LJ2 = 'pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\npair_coeff 1 2 0.8 1.1';

const CASES: Record<string, { input: string; keys: string[]; lost?: boolean }> = {
  // free surfaces in y that evaporate (the box grows at its resets), langevin's per-atom random numbers in the sorted
  // order of every 20 steps
  'shrink-wrapped y (p s p), fix langevin, atom_modify sort': {
    input: `units lj
atom_style atomic
boundary p s p
lattice fcc 0.8442
region box block 0 5 0 4 0 4
create_box 2 box
create_atoms 1 box
mass * 1.0
set group all type/fraction 2 0.4 4321
atom_modify sort 20 1.0
velocity all create 2.5 87287 loop geom
${LJ2}
fix 1 all nve
fix 2 all langevin 2.5 2.5 0.5 48279
timestep 0.004
thermo_style custom step temp pe press vol ly nbuild ndanger
thermo 10
run 100
`,
    keys: ['temp', 'pe', 'press', 'vol', 'ly', 'nbuild', 'ndanger'],
  },
  // a hot 2d gas leaving through fixed walls, the lost atoms dropped on the reneighboring steps
  'fixed x and y (f f p) in 2d, thermo_modify lost ignore': {
    input: `units lj
dimension 2
atom_style atomic
boundary f f p
lattice sq2 0.8
region box block 0 12 0 12 -0.5 0.5
create_box 1 box
region inner block 1 11 1 11 -0.5 0.5
create_atoms 1 region inner
mass 1 1.0
velocity all create 10.0 4928459 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
fix 1 all nve
fix 2 all enforce2d
timestep 0.005
thermo_style custom step temp pe press atoms nbuild
thermo_modify lost ignore
thermo 10
run 120
`,
    keys: ['temp', 'pe', 'press', 'atoms', 'nbuild'],
    lost: true,
  },
  // a charged slab between reflecting walls at the fixed z edges, Ewald with the slab correction on the CPU beside it
  'fixed z (p p f), lj/cut/coul/long with ewald and kspace_modify slab, fix wall/reflect': {
    input: `units lj
atom_style charge
boundary p p f
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 2 box
create_atoms 1 box
mass * 1.0
set group all type/fraction 2 0.5 4321
set type 1 charge 0.5
set type 2 charge -0.5
velocity all create 1.5 87287 loop geom
pair_style lj/cut/coul/long 2.5
pair_coeff * * 1.0 1.0
pair_coeff 1 2 0.8 1.1
pair_modify table 0
kspace_style ewald 1e-4
kspace_modify slab 3.0
fix 1 all nve
fix 2 all wall/reflect zlo EDGE zhi EDGE
timestep 0.004
thermo_style custom step temp pe evdwl ecoul elong press nbuild
thermo 10
run 60
`,
    keys: ['temp', 'pe', 'evdwl', 'ecoul', 'elong', 'press', 'nbuild'],
  },
  // a fully periodic box: the builds and the unwrapped motion between them as on the CPU path
  'fully periodic: the CPU path builds and positions': {
    input: `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 5 0 5 0 4
create_box 2 box
create_atoms 1 box
mass * 1.0
set group all type/fraction 2 0.4 4321
velocity all create 3.0 87287 loop geom
${LJ2}
neigh_modify every 2 delay 4 check yes
fix 1 all nve
timestep 0.005
thermo_style custom step temp pe press nbuild ndanger
thermo 10
run 100
`,
    keys: ['temp', 'pe', 'press', 'nbuild', 'ndanger'],
  },
};

describe('non-periodic boxes on the GPU path', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(`${name}: the CPU engine's thermo, atoms and positions`, async () => {
      const cpu = session(new CpuForceBackend());
      await cpu.s.execute(c.input);
      const stand = new SpecInFp64();
      const gpu = session(stand);
      await gpu.s.execute(c.input);
      expect(gpu.logs.filter((l) => l.includes('general fp64 CPU engine'))).toEqual([]);
      expect(stand.calls).toBeGreaterThan(50);
      expect(gpu.rows.length).toBe(cpu.rows.length);
      for (let r = 0; r < cpu.rows.length; r++) {
        for (const k of c.keys) {
          const a = cpu.rows[r][k], b = gpu.rows[r][k];
          expect(Math.abs(a - b), `row ${r} ${k}: ${a} vs ${b}`).toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(a)));
        }
      }
      // more than one build, and (where atoms leave) atoms lost on the way
      expect(cpu.rows[cpu.rows.length - 1].nbuild).toBeGreaterThan(2);
      if (c.lost) expect(cpu.rows[cpu.rows.length - 1].atoms).toBeLessThan(cpu.rows[0].atoms);
      const a = cpu.sys().state, b = gpu.sys().state;
      expect(b.n).toBe(a.n);
      for (let k = 0; k < 3 * a.n; k++) {
        expect(Math.abs(a.x[k] - b.x[k]), `x[${k}]`).toBeLessThan(1e-9);
        expect(b.image[k]).toBe(a.image[k]);
      }
      for (let d = 0; d < 3; d++) {
        expect(Math.abs(a.box.lo[d] - b.box.lo[d])).toBeLessThan(1e-12);
        expect(Math.abs(a.box.hi[d] - b.box.hi[d])).toBeLessThan(1e-12);
      }
    });
  }

  it('lost atoms through a fixed boundary stop both runs on the same step with the same error', async () => {
    const input = CASES['fixed x and y (f f p) in 2d, thermo_modify lost ignore'].input.replace('thermo_modify lost ignore\n', '');
    const cpu = session(new CpuForceBackend());
    const cpuErr = await cpu.s.execute(input).then(() => '', (e: Error) => e.message);
    const gpu = session(new SpecInFp64());
    const gpuErr = await gpu.s.execute(input).then(() => '', (e: Error) => e.message);
    expect(cpuErr).toMatch(/Lost atoms: original \d+ current \d+/);
    expect(gpuErr).toBe(cpuErr);
    expect(gpu.rows.length).toBe(cpu.rows.length);
    expect(gpu.sys().state.step).toBe(cpu.sys().state.step);
  });

  it('the accelerator: the styles kernel for lj/cut, not resident; the lj/cut threads and triclinic boxes stay on the CPU', async () => {
    const run = async (boundary: string, extra = '') => {
      const { s, sys } = session();
      await s.execute(`units lj\nboundary ${boundary}\nlattice fcc 0.8442\nregion box block 0 5 0 5 0 5\ncreate_box 1 box\n`
        + `create_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\n${extra}fix 1 all nve\nrun 0\n`);
      return sys();
    };
    const gpu = { kind: 'webgpu', label: 'WebGPU · test', compute: () => ({ pe: 0, virial: 0 }), dispose() {}, canAdvance: () => true, canAdvanceSpec: () => false } as unknown as ForceBackend;
    const threads = { kind: 'cpu', label: 'CPU · threads', compute: () => ({ pe: 0, virial: 0 }), dispose() {} } as unknown as ForceBackend;
    for (const b of ['p s p', 'f f p', 'p p m', 's s s']) {
      const sys = await run(b);
      const a = accelerator(sys, gpu);
      expect(a.reason, b).toBeNull();
      expect(a.accel?.spec?.style).toBe(10);
      expect(a.accel?.resident).toBe(false);
      expect(accelerator(sys, threads).reason).toBe('a non-periodic box');
    }
    // a periodic edge of fewer than 2 cutoffs is still refused; a non-periodic one of any length is not
    const thinPeriodic = await run('p p f', 'change_box all z final 0 50 units box\nchange_box all x final 0 4 remap units box\n');
    expect(accelerator(thinPeriodic, gpu).reason).toBe('a box shorter than 2 pair cutoffs along a periodic edge');
    const thinFixed = await run('p p f', 'change_box all z final 0 4 remap units box\n');
    expect(accelerator(thinFixed, gpu).reason).toBeNull();
    const tri = await run('p p p', 'change_box all triclinic\n');
    expect(accelerator(tri, gpu).reason).toBe('a triclinic box');
    // a fully periodic box keeps the plain lj/cut kernel
    const plain = accelerator(await run('p p p'), gpu);
    expect(plain.accel?.spec).toBeNull();
  });
});

describe('the GPU cell sort with atoms past the box edges', () => {
  /** The cell scan of KERNEL_STYLES (webgpuForces.ts) on the sort's output: every pair within cut, as atom indices. */
  const scan = (st: SimState, cut: number): Set<string> => {
    const buf = newCellScratch();
    const g = sortByCell(buf, st, cut, 0, 1);
    if (!g) throw new Error('no cells');
    const { nc, L, base, two } = g;
    const ints = buf.intsData, pos = buf.posData, sorted = buf.sorted;
    const wrap = (c: number, m: number) => ((c % m) + m) % m;
    const lo = nc.map((m) => (m < 3 ? 0 : -1)), hi = nc.map((m) => (m < 3 ? m - 1 : 1));
    const out = new Set<string>();
    for (let i = 0; i < st.n; i++) {
      const c = ints[base + i];
      const cx = c % nc[0], cy = Math.floor(c / nc[0]) % nc[1], cz = Math.floor(c / (nc[0] * nc[1]));
      for (let dz = lo[2]; dz <= hi[2]; dz++) {
        for (let dy = lo[1]; dy <= hi[1]; dy++) {
          for (let dx = lo[0]; dx <= hi[0]; dx++) {
            const c2 = (wrap(cz + dz, nc[2]) * nc[1] + wrap(cy + dy, nc[1])) * nc[0] + wrap(cx + dx, nc[0]);
            for (let j = ints[c2]; j < ints[c2 + 1]; j++) {
              if (j === i) continue;
              const d = [0, 1, 2].map((k) => { const v = pos[4 * i + k] - pos[4 * j + k]; return v - L[k] * Math.round(v / L[k]); });
              if (two) d[2] = 0;
              if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < cut * cut) out.add(`${sorted[i]},${sorted[j]}`);
            }
          }
        }
      }
    }
    return out;
  };

  /** Every pair within cut by the minimum image along the periodic dimensions; pairs within 1e-4 of cut in `edge`. */
  const brute = (st: SimState, cut: number): { pairs: Set<string>; edge: Set<string> } => {
    const L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
    const pairs = new Set<string>(), edge = new Set<string>();
    for (let i = 0; i < st.n; i++) {
      for (let j = 0; j < st.n; j++) {
        if (i === j) continue;
        const d = [0, 1, 2].map((k) => { const v = st.x[3 * i + k] - st.x[3 * j + k]; return st.box.periodic[k] ? v - L[k] * Math.round(v / L[k]) : v; });
        if (st.dimension === 2) d[2] = 0;
        const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (Math.abs(r - cut) < 1e-4) edge.add(`${i},${j}`);
        else if (r < cut) pairs.add(`${i},${j}`);
      }
    }
    return { pairs, edge };
  };

  /** n atoms in [lo - out, hi + out) along each dimension (a deterministic stream). */
  const atoms = (n: number, lo: number[], hi: number[], periodic: [boolean, boolean, boolean], out: number, dimension: 2 | 3): SimState => {
    let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const x = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 3; d++) x[3 * i + d] = d === 2 && dimension === 2 ? 0 : lo[d] - out + rnd() * (hi[d] - lo[d] + 2 * out);
    }
    return {
      n, x, type: new Int32Array(n).fill(1), dimension,
      box: { lo: [...lo], hi: [...hi], periodic, triclinic: false, tilt: [0, 0, 0] },
    } as unknown as SimState;
  };

  const GRIDS: Array<{ name: string; lo: number[]; hi: number[]; periodic: [boolean, boolean, boolean]; dim: 2 | 3 }> = [
    // periodic edges of 4 to 6 cells: an atom past one edge is scanned from its image's cell
    { name: 'p s p', lo: [0, -1, 0], hi: [15, 8, 12.5], periodic: [true, false, true], dim: 3 },
    { name: 'f f f with edges of 1 and 2 cells', lo: [0, 0, 0], hi: [2, 6, 9], periodic: [false, false, false], dim: 3 },
    { name: 'p p f with a thin z', lo: [-3, -3, 0], hi: [9, 10, 1.2], periodic: [true, true, false], dim: 3 },
    { name: 'f p in 2d', lo: [0, 0, -0.5], hi: [10, 13, 0.5], periodic: [false, true, true], dim: 2 },
  ];

  for (const gr of GRIDS) {
    it(`${gr.name}: the scan finds the pairs of the minimum image along the periodic edges`, () => {
      const cut = 2.5;
      const st = atoms(gr.dim === 2 ? 200 : 400, gr.lo, gr.hi, gr.periodic, 1.1, gr.dim);
      const got = scan(st, cut);
      const { pairs, edge } = brute(st, cut);
      expect(pairs.size).toBeGreaterThan(200);
      for (const p of pairs) expect(got.has(p), `missing ${p}`).toBe(true);
      for (const p of got) expect(pairs.has(p) || edge.has(p), `extra ${p}`).toBe(true);
      const g = sortByCell(newCellScratch(), st, cut, 0, 1)!;
      for (let d = 0; d < 3; d++) {
        if (!gr.periodic[d] && !(d === 2 && gr.dim === 2)) expect(g.L[d]).toBe(NO_IMAGE);
        else expect(g.L[d]).toBe(gr.hi[d] - gr.lo[d]);
      }
    });
  }

  it('periodicCell: inside the box a plain floor (hi in the last cell), outside it the cell of the image', () => {
    const len = 10, nc = 4, sc = nc / len;
    expect(periodicCell(0, len, sc, nc)).toBe(0);
    expect(periodicCell(9.99, len, sc, nc)).toBe(3);
    expect(periodicCell(9.999999999999998, len, sc, nc)).toBe(3);
    expect(periodicCell(10, len, sc, nc)).toBe(0);
    expect(periodicCell(10.1, len, sc, nc)).toBe(0);
    expect(periodicCell(13, len, sc, nc)).toBe(1);
    expect(periodicCell(-0.1, len, sc, nc)).toBe(3);
    expect(periodicCell(-2.6, len, sc, nc)).toBe(2);
    expect(periodicCell(-25, len, sc, nc)).toBe(2);
  });
});

describe('the CPU lj/cut cell list with atoms past the periodic edges', () => {
  it('gives the forces of the wrapped positions', () => {
    let seed = 99;
    const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const n = 300, Lb = 9;
    const x = new Float64Array(3 * n);
    for (let k = 0; k < 3 * n; k++) x[k] = rnd() * Lb;
    const mk = (xs: Float64Array) => ({
      n, x: xs, f: new Float64Array(3 * n), type: new Int32Array(n).fill(1), dimension: 3,
      box: { lo: [0, 0, 0], hi: [Lb, Lb, Lb], periodic: [true, true, true], triclinic: false, tilt: [0, 0, 0] },
    }) as unknown as SimState;
    const table = newPairTable(1, 2.5);
    table.pairs[3] = { epsilon: 1, sigma: 1, cutoff: 2.5 };
    table.explicit[3] = true;
    const wrapped = mk(x);
    const shifted = Float64Array.from(x);
    // every other atom within 0.6 of an edge moved to its image past the other edge (as between reneighborings): the
    // same physical system
    let out = 0;
    for (let i = 0; i < n; i += 2) {
      for (let d = 0; d < 3; d++) {
        const k = 3 * i + d;
        if (x[k] < 0.6) { shifted[k] += Lb; out++; } else if (x[k] > Lb - 0.6) { shifted[k] -= Lb; out++; }
      }
    }
    expect(out).toBeGreaterThan(20);
    const moved = mk(shifted);
    const be = new CpuForceBackend();
    const a = be.compute(wrapped, table);
    const b = be.compute(moved, table);
    expect(Math.abs(a.pe - b.pe)).toBeLessThan(1e-9 * Math.abs(a.pe));
    for (let k = 0; k < 3 * n; k++) expect(Math.abs(wrapped.f[k] - moved.f[k])).toBeLessThan(1e-8);
  });
});
