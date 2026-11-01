import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { gpuTersoffSpec, isGpuTersoff, tersoffGather, type GpuTersoffSpec, type TersoffPair } from '../src/engine/gpu/tersoffSpec';
import { accelerator } from '../src/engine/run/accel';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import type { EngineEvent, ForceBackend, ForceResult, SimState, ThermoRow } from '../src/engine/types';

/*
 * pair tersoff on the GPU path (gpu/tersoffSpec.ts, gpu/tersoffKernels.ts, webgpuForces.ts computeTersoff): the
 * kernels' gather arithmetic in fp64 (tersoffGather) gives the CPU style's forces, energy and virial, and a run through
 * the accelerated loop with a backend that does what the kernels do gives the CPU engine's thermo. The WGSL itself is
 * checked in Chromium (SwiftShader).
 */

const POT = (f: string) => readFileSync(join(__dirname, '..', 'third_party', 'lammps', 'potentials', f), 'utf8');

const BOX = (types: number, a = 5.431) => `units metal
atom_style atomic
lattice diamond ${a}
region box block 0 3 0 3 0 3
create_box ${types} box
create_atoms 1 box
mass * 28.06
${types > 1 ? 'set group all type/fraction 2 0.5 4321\n' : ''}${types > 2 ? 'set group all type/fraction 3 0.3 9876\n' : ''}displace_atoms all random 0.1 0.1 0.1 8191 units box
velocity all create 1200.0 87287 loop geom
`;

const CASES: Record<string, { types: number; lines: string; file: string; a?: number }> = {
  tersoff: { types: 1, lines: 'pair_style tersoff\npair_coeff * * Si.tersoff Si', file: 'Si.tersoff' },
  'tersoff shift': { types: 1, lines: 'pair_style tersoff shift 0.05\npair_coeff * * Si.tersoff Si', file: 'Si.tersoff' },
  'tersoff, two elements': { types: 2, lines: 'pair_style tersoff\npair_coeff * * SiC.tersoff Si C', file: 'SiC.tersoff' },
  'tersoff, three elements': { types: 3, lines: 'pair_style tersoff\npair_coeff * * BNC.tersoff B N C', file: 'BNC.tersoff', a: 3.57 },
};

/** A backend that does what computeTersoff does, in fp64. */
class TersoffInFp64 implements ForceBackend {
  readonly kind = 'webgpu' as const;
  readonly label = 'WebGPU · fp64 stand-in';
  calls = 0;
  compute(): ForceResult { throw new Error('the stand-in runs computeTersoff only'); }
  dispose(): void {}
  async computeTersoff(st: SimState, spec: GpuTersoffSpec): Promise<ForceResult & { evdwl: number; ecoul: number }> {
    this.calls++;
    const L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
    const r = tersoffGather(spec, st.n, st.x, st.type, L);
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

describe('pair tersoff on the GPU path', () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(`${name}: the kernels' gather law gives the CPU style's forces, energy and virial`, async () => {
      const { s, sys } = session();
      s.addFile(c.file, POT(c.file));
      await s.execute(`${BOX(c.types, c.a)}${c.lines}\nthermo_style custom step pe press\nrun 0\n`);
      const st = sys().state;
      const pair = sys().ff.pair!;
      expect(isGpuTersoff(pair)).toBe(true);
      const spec = gpuTersoffSpec(pair as TersoffPair);
      if (typeof spec === 'string') throw new Error(spec);
      const L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
      const got = tersoffGather(spec, st.n, st.x, st.type, L);
      const ev = sys().ff.acc.evdwl;
      expect(ev).not.toBe(0);
      expect(Math.abs(got.e - ev)).toBeLessThan(1e-9 * Math.abs(ev));
      let worst = 0, scale = 0;
      for (let k = 0; k < 3 * st.n; k++) { worst = Math.max(worst, Math.abs(got.f[k] - st.f[k])); scale = Math.max(scale, Math.abs(st.f[k])); }
      expect(scale).toBeGreaterThan(0);
      expect(worst / scale).toBeLessThan(1e-9);
      // the virial: the CPU's (pair virial trace) is sum x . f over owned and ghost atoms
      const v = sys().ff.acc.virial;
      expect(Math.abs(got.virial - (v[0] + v[1] + v[2]))).toBeLessThan(1e-8 * Math.max(1, Math.abs(v[0] + v[1] + v[2])));
    });

    it(`${name}: a run through the accelerated loop gives the CPU engine's thermo`, async () => {
      const cpu = session(new CpuForceBackend());
      cpu.s.addFile(c.file, POT(c.file));
      await cpu.s.execute(`${BOX(c.types, c.a)}${c.lines}\n${RUN}`);
      const stand = new TersoffInFp64();
      const gpu = session(stand);
      gpu.s.addFile(c.file, POT(c.file));
      await gpu.s.execute(`${BOX(c.types, c.a)}${c.lines}\n${RUN}`);
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

  it('the accelerator: a Tersoff spec, not resident; tersoff/mod and short boxes stay on the CPU', async () => {
    const gpu = { kind: 'webgpu', label: 'WebGPU · test', compute: () => ({ pe: 0, virial: 0 }), dispose() {} } as unknown as ForceBackend;
    const run = async (box: string, lines: string, file: string) => {
      const { s, sys } = session();
      s.addFile(file, POT(file));
      await s.execute(`${box}${lines}\nfix 1 all nve\nrun 0\n`);
      return accelerator(sys(), gpu);
    };
    const a = await run(BOX(1), CASES.tersoff.lines, 'Si.tersoff');
    expect(a.reason).toBeNull();
    expect(a.accel?.tersoff).not.toBeNull();
    expect(a.accel?.resident).toBe(false);
    const small = await run(BOX(1).replace('block 0 3 0 3 0 3', 'block 0 1 0 1 0 1'), CASES.tersoff.lines, 'Si.tersoff');
    expect(small.reason).toBe('a box shorter than 2 pair cutoffs along a periodic edge');
    const mod = await run(BOX(1), 'pair_style tersoff/mod\npair_coeff * * Si.tersoff.mod Si', 'Si.tersoff.mod');
    expect(mod.reason).toMatch(/^pair style tersoff\/mod \(the GPU path implements/);
  });
});
