import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { eamDensity, eamPair, gpuEamSpec, isGpuEam, type EamPair, type GpuEamSpec } from '../src/engine/gpu/eamSpec';
import { accelerator } from '../src/engine/run/accel';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import type { EngineEvent, ForceBackend, ForceResult, SimState, ThermoRow } from '../src/engine/types';

/*
 * pair eam, eam/alloy and eam/fs on the GPU path (gpu/eamSpec.ts, gpu/eamKernels.ts, webgpuForces.ts computeEAM). The
 * kernel's arithmetic in fp64 over the packed f32 tables (eamDensity, eamPair) gives the CPU style's forces and
 * energy, and a run through the accelerated loop with a backend that does what the two kernels do (densities, the
 * style's embedding step, forces) gives the CPU engine's thermo. The WGSL itself is checked in Chromium (SwiftShader).
 */

const POT = (f: string) => readFileSync(join(__dirname, '..', 'third_party', 'lammps', 'potentials', f), 'utf8');

const BOX = `units metal
atom_style atomic
lattice fcc 3.6
region box block 0 5 0 5 0 5
create_box 2 box
create_atoms 1 box
mass * 60.0
set group all type/fraction 2 0.5 4321
displace_atoms all random 0.1 0.1 0.1 8191
velocity all create 600.0 87287 loop geom
`;

const STYLES: Record<string, { lines: string; files: string[] }> = {
  eam: { lines: 'pair_style eam\npair_coeff 1 1 Cu_u3.eam\npair_coeff 2 2 Cu_u3.eam', files: ['Cu_u3.eam'] },
  'eam/alloy': { lines: 'pair_style eam/alloy\npair_coeff * * CuNi.eam.alloy Cu Ni', files: ['CuNi.eam.alloy'] },
  'eam/fs': { lines: 'pair_style eam/fs\npair_coeff * * AlFe_mm.eam.fs Al Fe', files: ['AlFe_mm.eam.fs'] },
};

/** What the two EAM kernels and the CPU embedding step compute, in fp64 over every pair (minimum image). */
const kernelForces = (st: SimState, spec: GpuEamSpec, pair: EamPair) => {
  const n = st.n;
  const L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
  const dvec = (i: number, j: number) => [0, 1, 2].map((k) => { const v = st.x[3 * i + k] - st.x[3 * j + k]; return v - L[k] * Math.round(v / L[k]); });
  const rho = new Float64Array(n), fp = new Float64Array(n), f = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const d = dvec(i, j);
      rho[i] += eamDensity(spec, st.type[i], st.type[j], d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    }
  }
  const eEmbed = pair.embedOwned(st.type, rho, n, fp);
  let epair = 0, virial = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = dvec(i, j), rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      const r = eamPair(spec, st.type[i], st.type[j], rsq, fp[i], fp[j]);
      epair += r.phi;
      virial += r.fpair * rsq;
      for (let k = 0; k < 3; k++) { f[3 * i + k] += r.fpair * d[k]; f[3 * j + k] -= r.fpair * d[k]; }
    }
  }
  return { f, evdwl: epair + eEmbed, virial };
};

/** A backend that does what computeEAM does, in fp64. */
class EamInFp64 implements ForceBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'WebGPU · fp64 stand-in';
  calls = 0;
  constructor(private readonly sysRef: () => System) {}
  compute(): ForceResult { throw new Error('the stand-in runs computeEAM only'); }
  dispose(): void {}
  async computeEAM(st: SimState, spec: GpuEamSpec): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const r = kernelForces(st, spec, this.sysRef().ff.pair as EamPair);
    st.f.set(r.f);
    return { pe: r.evdwl, virial: r.virial, evdwl: r.evdwl, ecoul: 0 };
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

const RUN = `fix 1 all nve
timestep 0.002
thermo_style custom step temp pe ke etotal press
thermo 10
run 30
`;

describe('pair eam on the GPU path', () => {
  for (const [name, c] of Object.entries(STYLES)) {
    it(`${name}: the kernels' law gives the CPU style's forces and energy`, async () => {
      const { s, sys } = session();
      for (const f of c.files) s.addFile(f, POT(f));
      await s.execute(`${BOX}${c.lines}\nrun 0\n`);
      const st = sys().state;
      const pair = sys().ff.pair!;
      expect(isGpuEam(pair)).toBe(true);
      const spec = gpuEamSpec(pair as EamPair);
      if (typeof spec === 'string') throw new Error(spec);
      const got = kernelForces(st, spec, pair as EamPair);
      // the tables are f32 on the GPU: agreement to f32 precision
      expect(Math.abs(got.evdwl - sys().ff.acc.evdwl)).toBeLessThan(2e-6 * Math.abs(sys().ff.acc.evdwl));
      let worst = 0, scale = 0;
      for (let k = 0; k < 3 * st.n; k++) { worst = Math.max(worst, Math.abs(got.f[k] - st.f[k])); scale = Math.max(scale, Math.abs(st.f[k])); }
      expect(worst / scale).toBeLessThan(1e-5);
    });

    it(`${name}: a run through the accelerated loop gives the CPU engine's thermo`, async () => {
      const cpu = session(new CpuForceBackend());
      for (const f of c.files) cpu.s.addFile(f, POT(f));
      await cpu.s.execute(`${BOX}${c.lines}\n${RUN}`);
      let gsys: System | null = null;
      const stand = new EamInFp64(() => gsys!);
      const gpu = session(stand);
      gsys = gpu.sys();
      for (const f of c.files) gpu.s.addFile(f, POT(f));
      await gpu.s.execute(`${BOX}${c.lines}\n${RUN}`);
      expect(gpu.logs.filter((l) => l.includes('general fp64 CPU engine'))).toEqual([]);
      expect(stand.calls).toBe(31);
      expect(gpu.rows.length).toBe(cpu.rows.length);
      for (let r = 0; r < cpu.rows.length; r++) {
        for (const k of ['temp', 'pe', 'ke', 'etotal', 'press']) {
          const a = cpu.rows[r][k], b = gpu.rows[r][k];
          expect(Math.abs(a - b), `row ${r} ${k}: ${a} vs ${b}`).toBeLessThan(2e-5 * Math.max(1, Math.abs(a)));
        }
      }
    });
  }

  it('the accelerator: an EAM spec, not resident; hybrid and short boxes stay on the CPU', async () => {
    const gpu = { kind: 'webgpu', label: 'WebGPU · test', compute: () => ({ pe: 0, virial: 0 }), dispose() {} } as unknown as ForceBackend;
    const run = async (box: string, lines: string, files: string[]) => {
      const { s, sys } = session();
      for (const f of files) s.addFile(f, POT(f));
      await s.execute(`${box}${lines}\nfix 1 all nve\nrun 0\n`);
      return accelerator(sys(), gpu);
    };
    const a = await run(BOX, STYLES['eam/alloy'].lines, ['CuNi.eam.alloy']);
    expect(a.reason).toBeNull();
    expect(a.accel?.eam).not.toBeNull();
    expect(a.accel?.resident).toBe(false);
    const hybrid = await run(BOX, 'pair_style hybrid eam/alloy lj/cut 3.0\npair_coeff * * eam/alloy CuNi.eam.alloy Cu NULL\npair_coeff 2 2 lj/cut 0.1 2.5\npair_coeff 1 2 lj/cut 0.1 2.5', ['CuNi.eam.alloy']);
    expect(hybrid.reason).toMatch(/pair style hybrid .*eam, eam\/alloy, eam\/fs, sw, sw\/mod, tersoff, lj\/cut\/tip4p\/long, lj\/cut\/tip4p\/cut, tip4p\/long, tip4p\/cut only\)/);
    const small = await run(BOX.replace('block 0 5 0 5 0 5', 'block 0 2 0 2 0 2'), STYLES['eam/alloy'].lines, ['CuNi.eam.alloy']);
    expect(small.reason).toBe('a box shorter than 2 pair cutoffs along a periodic edge');
  });
});
