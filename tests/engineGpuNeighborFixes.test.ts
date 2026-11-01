import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { pairSpecEval, type GpuPairSpec } from '../src/engine/gpu/pairSpec';
import { NO_IMAGE } from '../src/engine/gpu/cellSort';
import { excludedPair, type GpuExclusions } from '../src/engine/gpu/exclusions';
import { accelerator } from '../src/engine/run/accel';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import type { EngineEvent, ForceBackend, ForceResult, SimState, ThermoRow } from '../src/engine/types';

/*
 * Fixes that act at neighbor-list time on the GPU path (run/accel.ts NEIGHBOR_TIME_FIXES): their pre_exchange and
 * pre_neighbor hooks run on the accelerated loop's reneighboring steps, before and after System.pbc and the sort, as
 * on the CPU path. A backend that does what computeSpec does in fp64 gives the CPU engine's thermo, atom count and final
 * positions through box flips (fix deform, fix npt flip yes), insertions (fix deposit) and the body remap of fix
 * rigid/nvt/small. Fixes that need the CPU lists or evaluate energies (balance, atom/swap) stay on the CPU.
 */

/** WGSL_BOX_IMAGE (gpu/cellSort.ts) in TypeScript. */
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
  async computeSpec(st: SimState, spec: GpuPairSpec, _special: unknown, ex: GpuExclusions | null = null): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const n = st.n, b = st.box;
    const L = [0, 1, 2].map((d) => (b.periodic[d] ? b.hi[d] - b.lo[d] : NO_IMAGE));
    const t = b.triclinic ? [...b.tilt] : [0, 0, 0];
    st.f.fill(0, 0, 3 * n);
    let evdwl = 0, virial = 0;
    const vt = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (ex && excludedPair(ex, st.mask, st.molecule, i, j)) continue;
        const d = boxImage([0, 1, 2].map((k) => st.x[3 * i + k] - st.x[3 * j + k]), L, t);
        if (st.dimension === 2) d[2] = 0;
        const rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        const r = pairSpecEval(spec, st.type[i], st.type[j], rsq, 0, 0);
        if (r.fpair === 0 && r.evdwl === 0) continue;
        evdwl += r.evdwl; virial += r.fpair * rsq;
        vt[0] += r.fpair * d[0] * d[0]; vt[1] += r.fpair * d[1] * d[1]; vt[2] += r.fpair * d[2] * d[2];
        vt[3] += r.fpair * d[0] * d[1]; vt[4] += r.fpair * d[0] * d[2]; vt[5] += r.fpair * d[1] * d[2];
        for (let k = 0; k < 3; k++) { st.f[3 * i + k] += r.fpair * d[k]; st.f[3 * j + k] -= r.fpair * d[k]; }
      }
    }
    return { pe: evdwl, virial, evdwl, ecoul: 0, virialTensor: vt };
  }
}

/**
 * Runs the input with the last line (its run command) after the fixes' pre_exchange and pre_neighbor are wrapped to
 * record the steps they run on: fix ID and hook -> steps.
 */
const runRecording = async (ses: ReturnType<typeof session>, input: string): Promise<Record<string, number[]>> => {
  const lines = input.trimEnd().split('\n');
  const runLine = lines.pop()!;
  await ses.s.execute(lines.join('\n') + '\n');
  const sys = ses.sys();
  const seen: Record<string, number[]> = {};
  for (const f of sys.fixes) {
    for (const hook of ['preExchange', 'preNeighbor'] as const) {
      const fn = f[hook];
      if (typeof fn !== 'function') continue;
      const key = `${f.id} ${hook}`;
      seen[key] = [];
      f[hook] = function (this: unknown) { seen[key].push(sys.state.step); return fn.call(f); } as never;
    }
  }
  await ses.s.execute(runLine + '\n');
  return seen;
};

const session = (backend?: ForceBackend) => {
  const rows: ThermoRow[] = [], logs: string[] = [];
  const s = new Session({
    emit: (ev: EngineEvent) => { if (ev.kind === 'thermo') rows.push(ev.row); else if (ev.kind === 'log') logs.push(ev.text); },
    writeFile: () => {},
  }, backend);
  return { s, rows, logs, sys: () => (s as unknown as { sys: System }).sys };
};

const CASES: Record<string, { input: string; keys: string[]; check?: (rows: ThermoRow[]) => void }> = {
  // a 2d shear flow whose tilt passes half the box length: fix deform flips the box in its pre_exchange
  'fix deform xy erate with remap v and nvt/sllod: a box flip': {
    input: `units lj
dimension 2
atom_style atomic
lattice sq2 0.7
region box prism 0 10 0 10 -0.5 0.5 4.5 0 0
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 4928459 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
fix 1 all nvt/sllod temp 1.0 1.0 0.5
fix 2 all deform 1 xy erate 0.1 remap v
fix 3 all enforce2d
timestep 0.005
thermo_style custom step temp pe press pxy xy nbuild
thermo 20
run 200
`,
    keys: ['temp', 'pe', 'press', 'pxy', 'xy', 'nbuild'],
    // the tilt grows and then flips to the other side
    check: (rows) => expect(rows.some((r, k) => k > 0 && r.xy < rows[k - 1].xy)).toBe(true),
  },
  // a tilt past 0.6 of its edge at the start: fix npt (flip yes) flips the box at its first reneighboring
  'fix npt tri, flip yes: a flip at the first rebuild': {
    input: `units lj
atom_style atomic
lattice fcc 0.8442
region box prism 0 6 0 6 0 6 3.7 0 0
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.5 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
fix 1 all npt temp 1.5 1.5 0.5 tri 1.0 1.0 2.0
timestep 0.004
thermo_style custom step temp pe press vol xy xz yz nbuild
thermo 5
run 60
`,
    keys: ['temp', 'pe', 'press', 'vol', 'xy', 'xz', 'yz', 'nbuild'],
    check: (rows) => expect(rows[rows.length - 1].xy).toBeLessThan(0),
  },
  // examples/deposit in.deposit.atom, shortened, with a 1-2 cutoff of 2.5 (its 5.0 needs a box of 2 cutoffs, 10 sigma,
  // and keeps the example on the CPU): atoms inserted every 100 steps above a substrate (boundary p p f)
  'fix deposit on a substrate (p p f)': {
    input: `units lj
atom_style atomic
boundary p p f
lattice fcc 1.0 origin 0.25 0.25 0.25
region box block 0 5 0 5 0 10
create_box 2 box
region substrate block INF INF INF INF INF 3
create_atoms 1 region substrate
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
pair_coeff 1 2 1.0 1.0 2.5
mass * 1.0
neigh_modify delay 0
group addatoms type 2
region mobile block 0 5 0 5 2 INF
group mobile region mobile
compute add addatoms temp
compute_modify add dynamic/dof yes extra/dof 0
fix 1 addatoms nve
fix 2 mobile langevin 1.0 1.0 0.1 587283
fix 3 mobile nve
region slab block 0 5 0 5 8 9
fix 4 addatoms deposit 100 2 100 12345 region slab near 1.0 vz -1.0 -1.0
fix 5 addatoms wall/reflect zhi EDGE
thermo_style custom step atoms temp epair etotal press nbuild
thermo 50
thermo_modify temp add
run 450
`,
    keys: ['atoms', 'temp', 'epair', 'etotal', 'press', 'nbuild'],
    // insertions on steps 1, 101, ..., 401
    check: (rows) => expect(rows[rows.length - 1].atoms).toBe(rows[0].atoms + 5),
  },
  // small rigid clusters (the 4 atoms of each fcc cell one molecule): fix rigid/nvt/small (engine-gpl/rigid/fix_rigid.ts)
  // remaps its bodies in pre_neighbor
  'fix rigid/nvt/small: the body remap of pre_neighbor': {
    input: `units lj
atom_style molecular
lattice fcc 0.8442
region box block 0 6 0 6 0 6
create_box 1 box
create_atoms 1 box
mass 1 1.0
variable m atom 1+floor((x+0.2)/xlat)+6*floor((y+0.2)/ylat)+36*floor((z+0.2)/zlat)
set group all mol v_m
velocity all create 3.0 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
neigh_modify exclude molecule/intra all
fix 1 all rigid/nvt/small molecule temp 3.0 3.0 1.0
timestep 0.004
thermo_style custom step temp pe press nbuild
thermo 10
run 100
`,
    keys: ['temp', 'pe', 'press', 'nbuild'],
  },
};

describe('fixes that act at neighbor-list time on the GPU path', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(`${name}: the CPU engine's thermo and positions`, async () => {
      const cpu = session(new CpuForceBackend());
      const cpuHooks = await runRecording(cpu, c.input);
      const stand = new SpecInFp64();
      const gpu = session(stand);
      const gpuHooks = await runRecording(gpu, c.input);
      // the neighbor-time hooks run on the same steps
      expect(Object.keys(cpuHooks).length).toBeGreaterThan(0);
      expect(gpuHooks).toEqual(cpuHooks);
      expect(gpu.logs.filter((l) => l.includes("general fp64 CPU engine")).join(" | ")).toBe("");
      expect(stand.calls).toBeGreaterThan(50);
      expect(gpu.rows.length).toBe(cpu.rows.length);
      for (let r = 0; r < cpu.rows.length; r++) {
        for (const k of c.keys) {
          const a = cpu.rows[r][k], b = gpu.rows[r][k];
          expect(Math.abs(a - b), `row ${r} ${k}: ${a} vs ${b}`).toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(a)));
        }
      }
      c.check?.(cpu.rows);
      expect(cpu.rows[cpu.rows.length - 1].nbuild).toBeGreaterThan(2);
      const a = cpu.sys().state, b = gpu.sys().state;
      expect(b.n).toBe(a.n);
      for (let k = 0; k < 3 * a.n; k++) {
        expect(Math.abs(a.x[k] - b.x[k]), `x[${k}]`).toBeLessThan(1e-9);
        expect(b.image[k]).toBe(a.image[k]);
      }
    });
  }

  it('the accelerator: list-free neighbor-time fixes on the GPU path; balance and atom/swap on the CPU', async () => {
    const run = async (lines: string) => {
      const { s, sys } = session();
      await s.execute(`units lj\nlattice fcc 0.8442\nregion box block 0 5 0 5 0 5\ncreate_box 2 box\ncreate_atoms 1 box\n`
        + `mass * 1.0\npair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\n${lines}\nrun 0\n`);
      return sys();
    };
    const gpu = { kind: 'webgpu', label: 'WebGPU · test', compute: () => ({ pe: 0, virial: 0 }), dispose() {}, canAdvance: () => true, canAdvanceSpec: () => true } as unknown as ForceBackend;
    const threads = { kind: 'cpu', label: 'CPU · threads', compute: () => ({ pe: 0, virial: 0 }), dispose() {} } as unknown as ForceBackend;
    const deform = await run('fix 1 all nve\nfix 2 all deform 1 x erate 0.01');
    expect(accelerator(deform, gpu).reason).toBeNull();
    expect(accelerator(deform, gpu).accel?.resident).toBe(false);
    const balance = await run('fix 1 all nve\nfix 2 all balance 10 1.1 shift x 5 1.1');
    expect(accelerator(balance, gpu).reason).toBe('fix balance (it acts when neighbor lists are rebuilt)');
    const swap = await run('fix 1 all nve\nfix 2 all atom/swap 10 1 29494 1.0 types 1 2');
    expect(accelerator(swap, gpu).reason).toBe('fix atom/swap (it acts when neighbor lists are rebuilt)');
    // the lj/cut threads: deform changes the box, which they take, but deposit inserts atoms in a fixed box
    expect(accelerator(deform, threads).reason).toBeNull();
  });
});
