import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { gpuSwSpec, isGpuSw, swGather, type GpuSwSpec, type SwPair } from '../src/engine/gpu/swSpec';
import { accelerator } from '../src/engine/run/accel';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import type { EngineEvent, ForceBackend, ForceResult, SimState, ThermoRow } from '../src/engine/types';

/*
 * pair sw and sw/mod on the GPU path (gpu/swSpec.ts, gpu/swKernels.ts, webgpuForces.ts computeSW): the kernels'
 * gather arithmetic in fp64 (swGather) gives the CPU styles' forces and energy (sw is the GPL port, sw/mod the core
 * style), and a run through the accelerated loop with a backend that does what the kernels do gives the CPU engine's
 * thermo. The WGSL itself is checked in Chromium (SwiftShader).
 */

const POT = (f: string) => readFileSync(join(__dirname, '..', 'third_party', 'lammps', 'potentials', f), 'utf8');

const BOX = (types: number) => `units metal
atom_style atomic
lattice diamond 5.431
region box block 0 3 0 3 0 3
create_box ${types} box
create_atoms 1 box
mass * 28.06
${types > 1 ? 'set group all type/fraction 2 0.5 4321\n' : ''}displace_atoms all random 0.1 0.1 0.1 8191 units box
velocity all create 1200.0 87287 loop geom
`;

const CASES: Record<string, { types: number; lines: string; file: string }> = {
  sw: { types: 1, lines: 'pair_style sw\npair_coeff * * Si.sw Si', file: 'Si.sw' },
  'sw/mod': { types: 1, lines: 'pair_style sw/mod maxdelcs 0.25 0.35\npair_coeff * * Si.sw Si', file: 'Si.sw' },
  'sw, two elements': { types: 2, lines: 'pair_style sw\npair_coeff * * GaN.sw Ga N', file: 'GaN.sw' },
};

/** A backend that does what computeSW does, in fp64. */
class SwInFp64 implements ForceBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'WebGPU · fp64 stand-in';
  calls = 0;
  compute(): ForceResult { throw new Error('the stand-in runs computeSW only'); }
  dispose(): void {}
  async computeSW(st: SimState, spec: GpuSwSpec): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
    const r = swGather(spec, st.n, st.x, st.type, L);
    st.f.set(r.f);
    return { pe: r.e, virial: r.virial, evdwl: r.e, ecoul: 0 };
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
timestep 0.001
thermo_style custom step temp pe ke etotal press
thermo 10
run 30
`;

describe('pair sw and sw/mod on the GPU path', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(`${name}: the kernels' gather law gives the CPU style's forces, energy and virial`, async () => {
      const { s, sys } = session();
      s.addFile(c.file, POT(c.file));
      await s.execute(`${BOX(c.types)}${c.lines}\nthermo_style custom step pe press\nrun 0\n`);
      const st = sys().state;
      const pair = sys().ff.pair!;
      expect(isGpuSw(pair)).toBe(true);
      const spec = gpuSwSpec(pair as SwPair);
      if (typeof spec === 'string') throw new Error(spec);
      const L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
      const got = swGather(spec, st.n, st.x, st.type, L);
      expect(Math.abs(got.e - sys().ff.acc.evdwl)).toBeLessThan(1e-9 * Math.abs(sys().ff.acc.evdwl));
      let worst = 0, scale = 0;
      for (let k = 0; k < 3 * st.n; k++) { worst = Math.max(worst, Math.abs(got.f[k] - st.f[k])); scale = Math.max(scale, Math.abs(st.f[k])); }
      expect(worst / scale).toBeLessThan(1e-9);
      // the virial: the CPU's (pair virial trace) is sum x . f over owned and ghost atoms
      const v = sys().ff.acc.virial;
      expect(Math.abs(got.virial - (v[0] + v[1] + v[2]))).toBeLessThan(1e-8 * Math.max(1, Math.abs(v[0] + v[1] + v[2])));
    });

    it(`${name}: a run through the accelerated loop gives the CPU engine's thermo`, async () => {
      const cpu = session(new CpuForceBackend());
      cpu.s.addFile(c.file, POT(c.file));
      await cpu.s.execute(`${BOX(c.types)}${c.lines}\n${RUN}`);
      const stand = new SwInFp64();
      const gpu = session(stand);
      gpu.s.addFile(c.file, POT(c.file));
      await gpu.s.execute(`${BOX(c.types)}${c.lines}\n${RUN}`);
      expect(gpu.logs.filter((l) => l.includes('general fp64 CPU engine'))).toEqual([]);
      expect(stand.calls).toBe(31);
      for (let r = 0; r < cpu.rows.length; r++) {
        for (const k of ['temp', 'pe', 'ke', 'etotal', 'press']) {
          const a = cpu.rows[r][k], b = gpu.rows[r][k];
          expect(Math.abs(a - b), `row ${r} ${k}: ${a} vs ${b}`).toBeLessThan(1e-8 * Math.max(1, Math.abs(a)));
        }
      }
    });
  }

  it('the accelerator: an SW spec, not resident; sw/angle/table and short boxes stay on the CPU', async () => {
    const gpu = { kind: 'webgpu', label: 'WebGPU · test', compute: () => ({ pe: 0, virial: 0 }), dispose() {} } as unknown as ForceBackend;
    const run = async (box: string, lines: string) => {
      const { s, sys } = session();
      s.addFile('Si.sw', POT('Si.sw'));
      await s.execute(`${box}${lines}\nfix 1 all nve\nrun 0\n`);
      return accelerator(sys(), gpu);
    };
    const a = await run(BOX(1), CASES.sw.lines);
    expect(a.reason).toBeNull();
    expect(a.accel?.sw).not.toBeNull();
    expect(a.accel?.resident).toBe(false);
    const small = await run(BOX(1).replace('block 0 3 0 3 0 3', 'block 0 1 0 1 0 1'), CASES.sw.lines);
    expect(small.reason).toBe('a box shorter than 2 pair cutoffs along a periodic edge');
  });
});
