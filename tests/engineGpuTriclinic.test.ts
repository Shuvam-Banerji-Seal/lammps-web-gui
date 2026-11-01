import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { pairSpecEval, type GpuPairSpec } from '../src/engine/gpu/pairSpec';
import { boxWidths, newCellScratch, NO_IMAGE, sortByCell } from '../src/engine/gpu/cellSort';
import { accelerator } from '../src/engine/run/accel';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import type { EngineEvent, ForceBackend, ForceResult, SimState, ThermoRow } from '../src/engine/types';

/*
 * Triclinic boxes on the GPU path. The cell sort (gpu/cellSort.ts) bins the atoms' fractional coordinates, nc_d cells
 * along d with the box's width perpendicular to the other two edges at least nc_d cutoffs, and the kernels' minimum
 * image (WGSL_BOX_IMAGE) rounds the fractional coordinates of a separation. Written out in TypeScript on the real cell
 * sort, the scan of the styles kernel finds exactly the pairs within the cutoff of an all-images search. A backend
 * that does what computeSpec does in fp64 gives the CPU engine's thermo through the accelerated loop, fix npt with a
 * changing tilt included.
 */

/** WGSL_BOX_IMAGE in TypeScript (L and tilt as the cell sort hands them to the kernels). */
const boxImage = (d: number[], L: number[], t: number[]): number[] => {
  const fz = d[2] / L[2];
  const fy = (d[1] - t[2] * fz) / L[1];
  const fx = (d[0] - t[0] * fy - t[1] * fz) / L[0];
  const kx = Math.round(fx), ky = Math.round(fy), kz = Math.round(fz);
  return [d[0] - (kx * L[0] + ky * t[0] + kz * t[1]), d[1] - (ky * L[1] + kz * t[2]), d[2] - kz * L[2]];
};

class SpecInFp64 implements ForceBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'WebGPU · fp64 stand-in';
  calls = 0;
  compute(): ForceResult { throw new Error('the stand-in runs computeSpec only'); }
  dispose(): void {}
  async computeSpec(st: SimState, spec: GpuPairSpec): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const n = st.n, b = st.box;
    const L = [0, 1, 2].map((d) => (b.periodic[d] ? b.hi[d] - b.lo[d] : NO_IMAGE));
    const t = b.triclinic ? [...b.tilt] : [0, 0, 0];
    st.f.fill(0, 0, 3 * n);
    let evdwl = 0, ecoul = 0, virial = 0;
    const vt = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = boxImage([0, 1, 2].map((k) => st.x[3 * i + k] - st.x[3 * j + k]), L, t);
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

const CASES: Record<string, { input: string; keys: string[] }> = {
  // a tilted cell in all three tilts, a hot liquid that wraps through every face
  'lj/cut in a prism (xy, xz, yz), fix nvt': {
    input: `units lj
atom_style atomic
lattice fcc 0.8442
region box prism 0 5 0 5 0 4 1.5 -1.0 0.8
create_box 2 box
create_atoms 1 box
mass * 1.0
set group all type/fraction 2 0.4 4321
velocity all create 2.0 87287 loop geom
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
pair_coeff 1 2 0.8 1.1
fix 1 all nvt temp 2.0 2.0 0.5
timestep 0.004
thermo_style custom step temp pe press pxy pxz pyz nbuild
thermo 10
run 80
`,
    keys: ['temp', 'pe', 'press', 'pxy', 'pxz', 'pyz', 'nbuild'],
  },
  // the barostat changes the tilts (flip no: no box flip, which runs on the CPU path)
  'fix npt tri flip no: a changing tilt': {
    input: `units lj
atom_style atomic
lattice fcc 0.8442
region box prism 0 5 0 5 0 5 0.8 0.4 -0.6
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.5 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
fix 1 all npt temp 1.5 1.5 0.5 tri 1.0 1.0 2.0 flip no
timestep 0.004
thermo_style custom step temp pe press vol xy xz yz pxy
thermo 10
run 80
`,
    keys: ['temp', 'pe', 'press', 'vol', 'xy', 'xz', 'yz', 'pxy'],
  },
  // a charged triclinic box: the real-space term on the stand-in, PPPM on the CPU beside it
  'lj/cut/coul/long with pppm in a triclinic box': {
    input: `units lj
atom_style charge
lattice fcc 0.8442
region box prism 0 4 0 4 0 4 1.0 0.5 -0.5
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
kspace_style pppm 1e-4
fix 1 all nve
timestep 0.004
thermo_style custom step temp pe evdwl ecoul elong press
thermo 10
run 50
`,
    keys: ['temp', 'pe', 'evdwl', 'ecoul', 'elong', 'press'],
  },
  // 2d, a tilted periodic cell (sq2 at 0.7: at 0.8 the lattice has pairs at the cutoff, sqrt(5) a/sqrt(2) = 2.5, whose
  // count then depends on the rounding of each path)
  'lj/cut in a 2d triclinic box': {
    input: `units lj
dimension 2
atom_style atomic
lattice sq2 0.7
region box prism 0 12 0 10 -0.5 0.5 3.0 0 0
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.5 4928459 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
fix 1 all nve
fix 2 all enforce2d
timestep 0.005
thermo_style custom step temp pe press pxy nbuild
thermo 10
run 80
`,
    keys: ['temp', 'pe', 'press', 'pxy', 'nbuild'],
  },
};

describe('triclinic boxes on the GPU path', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(`${name}: the CPU engine's thermo and positions`, async () => {
      const cpu = session(new CpuForceBackend());
      await cpu.s.execute(c.input);
      const stand = new SpecInFp64();
      const gpu = session(stand);
      await gpu.s.execute(c.input);
      expect(gpu.logs.filter((l) => l.includes('general fp64 CPU engine'))).toEqual([]);
      expect(stand.calls).toBeGreaterThan(40);
      expect(gpu.rows.length).toBe(cpu.rows.length);
      for (let r = 0; r < cpu.rows.length; r++) {
        for (const k of c.keys) {
          const a = cpu.rows[r][k], b = gpu.rows[r][k];
          expect(Math.abs(a - b), `row ${r} ${k}: ${a} vs ${b}`).toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(a)));
        }
      }
      const a = cpu.sys().state, b = gpu.sys().state;
      expect(b.n).toBe(a.n);
      for (let k = 0; k < 3 * a.n; k++) {
        expect(Math.abs(a.x[k] - b.x[k]), `x[${k}]`).toBeLessThan(1e-9);
        expect(b.image[k]).toBe(a.image[k]);
      }
    });
  }

  it('the accelerator: the styles kernel, not resident; the lj/cut threads stay on the CPU; the widths decide a short box', async () => {
    const run = async (prism: string) => {
      const { s, sys } = session();
      await s.execute(`units lj\nlattice fcc 0.8442\nregion box prism ${prism}\ncreate_box 1 box\n`
        + 'create_atoms 1 box\nmass 1 1.0\npair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0\nfix 1 all nve\nrun 0\n');
      return sys();
    };
    const gpu = { kind: 'webgpu', label: 'WebGPU · test', compute: () => ({ pe: 0, virial: 0 }), dispose() {}, canAdvance: () => true, canAdvanceSpec: () => true } as unknown as ForceBackend;
    const threads = { kind: 'cpu', label: 'CPU · threads', compute: () => ({ pe: 0, virial: 0 }), dispose() {} } as unknown as ForceBackend;
    const sys = await run('0 5 0 5 0 5 1.0 0.5 -0.5');
    const a = accelerator(sys, gpu);
    expect(a.reason).toBeNull();
    expect(a.accel?.spec?.style).toBe(10);
    expect(a.accel?.resident).toBe(false);
    expect(accelerator(sys, threads).reason).toBe('a triclinic box');
    // 3 lattice cells in y (5.04 sigma) is 2 cutoffs along the edge, but a yz tilt of 2 cells narrows the width
    // perpendicular to the xz face below that
    const upright = await run('0 5 0 3 0 5 0 0 0');
    expect(accelerator(upright, gpu).reason).toBeNull();
    const leaning = await run('0 5 0 3 0 5 0 0 2');
    expect(boxWidths(leaning.state.box)[1]).toBeLessThan(5);
    expect(accelerator(leaning, gpu).reason).toBe('a box shorter than 2 pair cutoffs along a periodic edge');
  });
});

describe('the GPU cell sort in a triclinic box', () => {
  /** The scan of KERNEL_STYLES on the sort's output: every pair within cut, as atom indices. */
  const scan = (st: SimState, cut: number): Set<string> => {
    const buf = newCellScratch();
    const g = sortByCell(buf, st, cut, 0, 1);
    if (!g) throw new Error('no cells');
    const { nc, L, tilt, base, two } = g;
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
              const d = boxImage([0, 1, 2].map((k) => pos[4 * i + k] - pos[4 * j + k]), L, tilt);
              if (two) d[2] = 0;
              if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < cut * cut) out.add(`${sorted[i]},${sorted[j]}`);
            }
          }
        }
      }
    }
    return out;
  };

  /** Every pair within cut over the images -2..2 of each periodic edge vector; pairs within 1e-4 of cut in `edge`. */
  const brute = (st: SimState, cut: number): { pairs: Set<string>; edge: Set<string> } => {
    const b = st.box;
    const lx = b.hi[0] - b.lo[0], ly = b.hi[1] - b.lo[1], lz = b.hi[2] - b.lo[2];
    const [xy, xz, yz] = b.tilt;
    const r = (d: number) => (b.periodic[d] && !(d === 2 && st.dimension === 2) ? 2 : 0);
    const pairs = new Set<string>(), edge = new Set<string>();
    for (let i = 0; i < st.n; i++) {
      for (let j = 0; j < st.n; j++) {
        if (i === j) continue;
        let best = Infinity;
        for (let c = -r(2); c <= r(2); c++) {
          for (let bb = -r(1); bb <= r(1); bb++) {
            for (let a = -r(0); a <= r(0); a++) {
              const dx = st.x[3 * i] - st.x[3 * j] + a * lx + bb * xy + c * xz;
              const dy = st.x[3 * i + 1] - st.x[3 * j + 1] + bb * ly + c * yz;
              const dz = st.dimension === 2 ? 0 : st.x[3 * i + 2] - st.x[3 * j + 2] + c * lz;
              best = Math.min(best, Math.sqrt(dx * dx + dy * dy + dz * dz));
            }
          }
        }
        if (Math.abs(best - cut) < 1e-4) edge.add(`${i},${j}`);
        else if (best < cut) pairs.add(`${i},${j}`);
      }
    }
    return { pairs, edge };
  };

  /** n atoms at fractional coordinates in [-out, 1 + out) of a triclinic box (a deterministic stream). */
  const atoms = (n: number, lo: number[], hi: number[], tilt: number[], periodic: [boolean, boolean, boolean], out: number, dimension: 2 | 3): SimState => {
    let seed = 777;
    const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const L = [0, 1, 2].map((d) => hi[d] - lo[d]);
    const x = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
      const f = [0, 1, 2].map((d) => (d === 2 && dimension === 2 ? 0 : -out + rnd() * (1 + 2 * out)));
      x[3 * i] = lo[0] + f[0] * L[0] + f[1] * tilt[0] + f[2] * tilt[1];
      x[3 * i + 1] = lo[1] + f[1] * L[1] + f[2] * tilt[2];
      x[3 * i + 2] = dimension === 2 ? 0 : lo[2] + f[2] * L[2];
    }
    return {
      n, x, type: new Int32Array(n).fill(1), dimension,
      box: { lo: [...lo], hi: [...hi], periodic, triclinic: true, tilt: [...tilt] },
    } as unknown as SimState;
  };

  const GRIDS: Array<{ name: string; lo: number[]; hi: number[]; tilt: number[]; periodic: [boolean, boolean, boolean]; dim: 2 | 3; n: number }> = [
    { name: 'p p p, all three tilts', lo: [0, 0, 0], hi: [14, 13, 12], tilt: [5, -4, 3.5], periodic: [true, true, true], dim: 3, n: 450 },
    { name: 'p p p, tilts at half the edges', lo: [-2, 1, 0], hi: [12, 14, 13], tilt: [-7, 7, -6.5], periodic: [true, true, true], dim: 3, n: 450 },
    { name: 'p p f, xy only', lo: [0, 0, 0], hi: [13, 12, 6], tilt: [6, 0, 0], periodic: [true, true, false], dim: 3, n: 300 },
    { name: '2d, xy', lo: [0, 0, -0.5], hi: [14, 12, 0.5], tilt: [-5, 0, 0], periodic: [true, true, true], dim: 2, n: 220 },
  ];

  for (const gr of GRIDS) {
    it(`${gr.name}: the scan finds the pairs of an all-images search`, () => {
      const cut = 2.5;
      const st = atoms(gr.n, gr.lo, gr.hi, gr.tilt, gr.periodic, 0.06, gr.dim);
      const w = boxWidths(st.box);
      for (let d = 0; d < (gr.dim === 2 ? 2 : 3); d++) if (gr.periodic[d]) expect(w[d]).toBeGreaterThanOrEqual(2 * cut);
      const got = scan(st, cut);
      const { pairs, edge } = brute(st, cut);
      expect(pairs.size).toBeGreaterThan(300);
      for (const p of pairs) expect(got.has(p), `missing ${p}`).toBe(true);
      for (const p of got) expect(pairs.has(p) || edge.has(p), `extra ${p}`).toBe(true);
    });
  }

  it('boxWidths: volume over the opposite face', () => {
    const box = { lo: [0, 0, 0], hi: [10, 8, 6], tilt: [3, -2, 4], periodic: [true, true, true], triclinic: true } as unknown as SimState['box'];
    const a = [10, 0, 0], b = [3, 8, 0], c = [-2, 4, 6];
    const cross = (u: number[], v: number[]) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const norm = (u: number[]) => Math.hypot(u[0], u[1], u[2]);
    const V = 480;
    const w = boxWidths(box);
    expect(w[0]).toBeCloseTo(V / norm(cross(b, c)), 12);
    expect(w[1]).toBeCloseTo(V / norm(cross(c, a)), 12);
    expect(w[2]).toBeCloseTo(V / norm(cross(a, b)), 12);
    const ortho = { ...box, triclinic: false } as SimState['box'];
    expect(boxWidths(ortho)).toEqual([10, 8, 6]);
  });
});
