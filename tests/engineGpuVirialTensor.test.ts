import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { buildAtomMap, type SpecialList } from '../src/engine/atoms';
import { pairSpecEval, type GpuPairSpec } from '../src/engine/gpu/pairSpec';
import { eamDensity, eamPair, type EamPair, type GpuEamSpec } from '../src/engine/gpu/eamSpec';
import { swGather, type GpuSwSpec } from '../src/engine/gpu/swSpec';
import { tersoffGather, type GpuTersoffSpec } from '../src/engine/gpu/tersoffSpec';
import { tip4pGather, type GpuTip4pSpec, type Tip4pSites } from '../src/engine/gpu/tip4pSpec';
import { accelerator } from '../src/engine/run/accel';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import type { EngineEvent, ForceBackend, ForceResult, PairTable, SimState, ThermoRow } from '../src/engine/types';

/*
 * The GPU force paths return the virial tensor with the forces (xx yy zz xy xz yz, summed in fp64 on the host), so a
 * barostat (fix npt, nph: their own pressure compute), a user pressure compute and the pressure-tensor thermo keywords
 * run on the GPU path. Each case runs fix npt aniso through the accelerated loop with a backend that does what the
 * kernels do in fp64 (the references of gpu/*Spec.ts), and compares the pressure tensor, the volume and the energies
 * with the CPU engine.
 */

const POT = (f: string) => readFileSync(join(__dirname, '..', 'third_party', 'lammps', 'potentials', f), 'utf8');

/** a (x) b added to vt with weight s (xx yy zz xy xz yz). */
const outer = (vt: number[], s: number, a: number[], b: number[]) => {
  vt[0] += s * a[0] * b[0]; vt[1] += s * a[1] * b[1]; vt[2] += s * a[2] * b[2];
  vt[3] += s * a[0] * b[1]; vt[4] += s * a[0] * b[2]; vt[5] += s * a[1] * b[2];
};

/** The GPU backend's methods, in fp64. */
class Fp64Gpu implements ForceBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'WebGPU · fp64 stand-in';
  calls = 0;
  private readonly cpu = new CpuForceBackend();
  constructor(private readonly sysRef: () => System) {}
  dispose(): void {}

  compute(st: SimState, table: PairTable): ForceResult {
    this.calls++;
    return this.cpu.compute(st, table);
  }

  private L(st: SimState) { return [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]); }

  async computeSpec(st: SimState, spec: GpuPairSpec): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const n = st.n, L = this.L(st);
    st.f.fill(0, 0, 3 * n);
    let evdwl = 0, ecoul = 0, virial = 0;
    const vt = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = [0, 1, 2].map((k) => { const v = st.x[3 * i + k] - st.x[3 * j + k]; return v - L[k] * Math.round(v / L[k]); });
        const rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        const r = pairSpecEval(spec, st.type[i], st.type[j], rsq, spec.charges ? st.q[i] : 0, spec.charges ? st.q[j] : 0);
        evdwl += r.evdwl; ecoul += r.ecoul; virial += r.fpair * rsq;
        outer(vt, r.fpair, d, d);
        for (let k = 0; k < 3; k++) { st.f[3 * i + k] += r.fpair * d[k]; st.f[3 * j + k] -= r.fpair * d[k]; }
      }
    }
    return { pe: evdwl + ecoul, virial, evdwl, ecoul, virialTensor: vt };
  }

  async computeEAM(st: SimState, spec: GpuEamSpec): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const n = st.n, L = this.L(st);
    const dvec = (i: number, j: number) => [0, 1, 2].map((k) => { const v = st.x[3 * i + k] - st.x[3 * j + k]; return v - L[k] * Math.round(v / L[k]); });
    const rho = new Float64Array(n), fp = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const d = dvec(i, j);
        rho[i] += eamDensity(spec, st.type[i], st.type[j], d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      }
    }
    const eEmbed = (this.sysRef().ff.pair as EamPair).embedOwned(st.type, rho, n, fp);
    st.f.fill(0, 0, 3 * n);
    let epair = 0, virial = 0;
    const vt = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = dvec(i, j), rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        const r = eamPair(spec, st.type[i], st.type[j], rsq, fp[i], fp[j]);
        epair += r.phi; virial += r.fpair * rsq;
        outer(vt, r.fpair, d, d);
        for (let k = 0; k < 3; k++) { st.f[3 * i + k] += r.fpair * d[k]; st.f[3 * j + k] -= r.fpair * d[k]; }
      }
    }
    return { pe: epair + eEmbed, virial, evdwl: epair + eEmbed, ecoul: 0, virialTensor: vt };
  }

  async computeSW(st: SimState, spec: GpuSwSpec): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const r = swGather(spec, st.n, st.x, st.type, this.L(st));
    st.f.set(r.f);
    return { pe: r.e, virial: r.virial, evdwl: r.e, ecoul: 0, virialTensor: r.virialTensor };
  }

  async computeTersoff(st: SimState, spec: GpuTersoffSpec): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const r = tersoffGather(spec, st.n, st.x, st.type, this.L(st));
    st.f.set(r.f);
    return { pe: r.e, virial: r.virial, evdwl: r.e, ecoul: 0, virialTensor: r.virialTensor };
  }

  async computeTip4p(st: SimState, spec: GpuTip4pSpec, special: SpecialList | null, sites: Tip4pSites): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const idx = buildAtomMap(st);
    const order = (i: number, j: number): number => {
      if (!spec.special || !special) return 0;
      for (let e = special.offset[i]; e < special.offset[i + 1]; e++) if (idx[special.partner[e]] === j) return special.order[e];
      return 0;
    };
    const r = tip4pGather(spec, st.n, st.x, st.type, st.q, sites, order, this.L(st));
    st.f.set(r.f.subarray(0, 3 * st.n));
    return { pe: r.evdwl + r.ecoul, virial: r.virial, evdwl: r.evdwl, ecoul: r.ecoul, virialTensor: r.virialTensor };
  }
}

/** n^3 TIP4P waters (as engineGpuTip4p.test.ts). */
const waterData = (n: number, a: number): string => {
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  const r0 = 0.9572, half = (104.52 / 2) * Math.PI / 180;
  const atoms: string[] = [], bonds: string[] = [], angles: string[] = [];
  let id = 0, mol = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
    mol++;
    const o = [(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a];
    const th = Math.PI * rnd(), ph = 2 * Math.PI * rnd(), ps = 2 * Math.PI * rnd();
    const u = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
    const t0 = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const v = [u[1] * t0[2] - u[2] * t0[1], u[2] * t0[0] - u[0] * t0[2], u[0] * t0[1] - u[1] * t0[0]];
    const vn = Math.hypot(v[0], v[1], v[2]);
    const w = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]].map((c) => c / vn);
    const e = v.map((c, m) => (c / vn) * Math.cos(ps) + w[m] * Math.sin(ps));
    const h1 = o.map((c, m) => c + r0 * (Math.cos(half) * u[m] + Math.sin(half) * e[m]));
    const h2 = o.map((c, m) => c + r0 * (Math.cos(half) * u[m] - Math.sin(half) * e[m]));
    const oi = ++id;
    atoms.push(`${oi} ${mol} 1 -1.0484 ${o.join(' ')}`);
    atoms.push(`${++id} ${mol} 2 0.5242 ${h1.join(' ')}`);
    atoms.push(`${++id} ${mol} 2 0.5242 ${h2.join(' ')}`);
    bonds.push(`${bonds.length + 1} 1 ${oi} ${oi + 1}`, `${bonds.length + 2} 1 ${oi} ${oi + 2}`);
    angles.push(`${angles.length + 1} 1 ${oi + 1} ${oi} ${oi + 2}`);
  }
  const L = n * a;
  return ['water box', '', `${id} atoms`, `${bonds.length} bonds`, `${angles.length} angles`, '2 atom types', '1 bond types', '1 angle types', '',
    `0 ${L} xlo xhi`, `0 ${L} ylo yhi`, `0 ${L} zlo zhi`, '', 'Masses', '', '1 15.9994', '2 1.008', '',
    'Atoms # full', '', ...atoms, '', 'Bonds', '', ...bonds, '', 'Angles', '', ...angles, ''].join('\n');
};

const LJ = (charged: boolean) => `units lj
atom_style ${charged ? 'charge' : 'atomic'}
lattice fcc 0.8442
region box block 0 5 0 5 0 5
create_box 2 box
create_atoms 1 box
mass * 1.0
set group all type/fraction 2 0.5 4321
${charged ? 'set type 1 charge 0.5\nset type 2 charge -0.5\n' : ''}displace_atoms all random 0.05 0.05 0.05 8191 units box
velocity all create 1.2 87287 loop geom
`;

const METAL = (lattice: string, mass: number, cells = 4) => `units metal
atom_style atomic
lattice ${lattice}
region box block 0 ${cells} 0 ${cells} 0 ${cells}
create_box 1 box
create_atoms 1 box
mass 1 ${mass}
displace_atoms all random 0.05 0.05 0.05 8191 units box
velocity all create 900.0 87287 loop geom
`;

const WATER = `units real
atom_style full
read_data water.data
bond_style harmonic
bond_coeff 1 450.0 0.9572
angle_style harmonic
angle_coeff 1 55.0 104.52
velocity all create 300.0 4928459 loop geom
`;

const CASES: Record<string, { box: string; pair: string; npt: string; dt: number; files?: Record<string, string>; tol?: number }> = {
  'lj/cut (the lj/cut kernel)': { box: LJ(false), pair: 'pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\npair_coeff 1 2 0.9 1.05', npt: 'fix 1 all npt temp 1.2 1.2 0.5 aniso 1.0 1.0 2.0', dt: 0.004 },
  'lj/cut/coul/cut (the styles kernel)': { box: LJ(true), pair: 'pair_style lj/cut/coul/cut 2.5\npair_coeff * * 1.0 1.0', npt: 'fix 1 all npt temp 1.2 1.2 0.5 aniso 1.0 1.0 2.0', dt: 0.004 },
  // 5 cells of 3.615 A: 18 A, more than 3 cutoffs of Cu_u3.eam (4.95 A), as the barostat requires
  // the GPU's EAM tables hold the spline coefficients in fp32 (engineGpuEam.test.ts: the same 2e-5)
  eam: { box: METAL('fcc 3.615', 63.55, 5), pair: 'pair_style eam\npair_coeff 1 1 Cu_u3.eam', npt: 'fix 1 all npt temp 900 900 0.1 aniso 0 0 1.0', dt: 0.002, files: { 'Cu_u3.eam': POT('Cu_u3.eam') }, tol: 2e-5 },
  sw: { box: METAL('diamond 5.431', 28.06), pair: 'pair_style sw\npair_coeff * * Si.sw Si', npt: 'fix 1 all npt temp 900 900 0.1 aniso 0 0 1.0', dt: 0.001, files: { 'Si.sw': POT('Si.sw') } },
  tersoff: { box: METAL('diamond 5.431', 28.06), pair: 'pair_style tersoff\npair_coeff * * Si.tersoff Si', npt: 'fix 1 all npt temp 900 900 0.1 aniso 0 0 1.0', dt: 0.001, files: { 'Si.tersoff': POT('Si.tersoff') } },
  'lj/cut/tip4p/long + pppm/tip4p': {
    box: WATER, dt: 0.5, files: { 'water.data': waterData(4, 3.1) },
    pair: 'pair_style lj/cut/tip4p/long 1 2 1 1 0.1546 3.5\npair_coeff 1 1 0.1852 3.1589\npair_coeff 1 2 0.0 1.0\npair_coeff 2 2 0.0 1.0\npair_modify table 0\nkspace_style pppm/tip4p 1.0e-4',
    npt: 'fix 1 all npt temp 300 300 100 aniso 1 1 1000',
  },
};

const RUN = (c: { npt: string; dt: number }) => `${c.npt}
timestep ${c.dt}
thermo_style custom step temp pe press vol pxx pyy pzz pxy pxz pyz
thermo 5
run 20
`;

const session = (backend: ForceBackend | undefined, files: Record<string, string> = {}) => {
  const rows: ThermoRow[] = [], logs: string[] = [];
  const s = new Session({
    emit: (ev: EngineEvent) => { if (ev.kind === 'thermo') rows.push(ev.row); else if (ev.kind === 'log') logs.push(ev.text); },
    writeFile: () => {},
  }, backend);
  for (const [k, v] of Object.entries(files)) s.addFile(k, v);
  return { s, rows, logs, sys: () => (s as unknown as { sys: System }).sys };
};

describe('the virial tensor on the GPU path: fix npt aniso', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(`${name}: the CPU engine's pressure tensor, volume and energies`, async () => {
      const cpu = session(new CpuForceBackend(), c.files);
      await cpu.s.execute(`${c.box}${c.pair}\n${RUN(c)}`);
      let gpu: ReturnType<typeof session> | null = null;
      const stand = new Fp64Gpu(() => gpu!.sys());
      gpu = session(stand, c.files);
      await gpu.s.execute(`${c.box}${c.pair}\n${RUN(c)}`);
      expect(gpu.logs.filter((l) => l.includes('general fp64 CPU engine'))).toEqual([]);
      expect(stand.calls).toBe(21);
      expect(cpu.rows.length).toBe(5);
      for (let r = 0; r < cpu.rows.length; r++) {
        for (const k of ['temp', 'pe', 'press', 'vol', 'pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz']) {
          const a = cpu.rows[r][k], b = gpu.rows[r][k];
          expect(Math.abs(a - b), `row ${r} ${k}: ${a} vs ${b}`).toBeLessThan((c.tol ?? 1e-8) * Math.max(1, Math.abs(a)));
        }
      }
      // the barostat moved the box, and the off-diagonal components are not zero
      expect(cpu.rows[4].vol).not.toBe(cpu.rows[0].vol);
      expect(Math.abs(cpu.rows[4].pxy) + Math.abs(cpu.rows[4].pyz)).toBeGreaterThan(0);
    });
  }

  it('the accelerator: npt on the GPU path, not resident; the lj/cut threads keep it on the CPU; a barostat needs 3 cutoffs per edge', async () => {
    const run = async (box: string, extra: string) => {
      const { s, sys } = session(undefined);
      await s.execute(`${box}pair_style lj/cut 2.5\npair_coeff * * 1.0 1.0\n${extra}\nrun 0\n`);
      return sys();
    };
    const gpu = { kind: 'webgpu', label: 'WebGPU · test', compute: () => ({ pe: 0, virial: 0 }), dispose() {}, canAdvance: () => true, canAdvanceSpec: () => true } as unknown as ForceBackend;
    const threads = { kind: 'cpu', label: 'CPU · threads', compute: () => ({ pe: 0, virial: 0 }), dispose() {} } as unknown as ForceBackend;
    const npt = await run(LJ(false), 'fix 1 all npt temp 1 1 0.5 iso 1 1 2');
    const a = accelerator(npt, gpu);
    expect(a.reason).toBeNull();
    expect(a.accel?.resident).toBe(false);
    expect(accelerator(npt, threads).reason).toBe('a user pressure compute');
    // nve with pxx in the thermo output: the GPU path, not the whole-step kernel (which has the scalar virial only)
    const pxx = await run(LJ(false), 'fix 1 all nve\nthermo_style custom step pxx');
    expect(accelerator(pxx, gpu).accel?.resident).toBe(false);
    expect(accelerator(pxx, threads).reason).toBe('pressure-tensor thermo keywords');
    // 5 lattice cells of 1.68 sigma: 8.4 sigma, more than 3 cutoffs of 2.5; 3 cells: 5.0 sigma, between 2 and 3 cutoffs
    const smallNpt = await run(LJ(true).replace('block 0 5 0 5 0 5', 'block 0 3 0 3 0 3'), 'fix 1 all npt temp 1 1 0.5 iso 1 1 2\npair_style lj/cut/coul/cut 2.5\npair_coeff * * 1.0 1.0');
    expect(accelerator(smallNpt, gpu).reason).toBe('a box shorter than 3 pair cutoffs along a periodic edge (a fix that changes the box)');
  });
});
