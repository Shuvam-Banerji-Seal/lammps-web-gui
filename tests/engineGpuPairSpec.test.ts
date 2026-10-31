import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { System } from '../src/engine/system';
import { gpuPairSpec, pairSpecEval, type GpuPairSpec } from '../src/engine/gpu/pairSpec';
import { accelerator } from '../src/engine/run/accel';
import type { ForceBackend } from '../src/engine/types';

/*
 * The pair styles the WebGPU backend runs besides lj/cut (gpu/pairSpec.ts): the coefficients taken from the engine's
 * initialized style and the kernel's force law (pairSpecEval, the same arithmetic in fp64) give the CPU style's forces
 * and energies over every pair of a charged system. The WGSL kernel itself is checked in Chromium (SwiftShader).
 */

const BOX = `units real
atom_style charge
lattice sc 3.4
region box block 0 6 0 6 0 6
create_box 2 box
create_atoms 1 box
mass * 20.0
set group all type/fraction 2 0.5 4321
set type 1 charge 0.6
set type 2 charge -0.6
displace_atoms all random 0.4 0.4 0.4 8191
`;

const STYLES: Record<string, string> = {
  'lj/cut/coul/cut': 'pair_style lj/cut/coul/cut 6.0 7.5\npair_coeff 1 1 0.2 3.0\npair_coeff 2 2 0.1 3.4 5.5 7.0\npair_modify shift yes',
  'coul/cut': 'pair_style coul/cut 7.0\npair_coeff * *\npair_coeff 1 2 6.5',
  buck: 'pair_style buck 6.5\npair_coeff * * 20000.0 0.3 120.0\npair_coeff 1 2 15000.0 0.28 90.0 5.8\npair_modify shift yes',
  'buck/coul/cut': 'pair_style buck/coul/cut 6.0 7.5\npair_coeff * * 20000.0 0.3 120.0\npair_coeff 1 2 15000.0 0.28 90.0 5.5 7.0',
  morse: 'pair_style morse 6.5\npair_coeff * * 0.3 1.2 3.3\npair_coeff 1 2 0.5 1.5 3.1 6.0\npair_modify shift yes',
};

const systemFor = async (lines: string): Promise<System> => {
  const s = new Session({ emit: () => {}, writeFile: () => {} });
  await s.execute(`${BOX}${lines}\nrun 0\n`);
  return (s as unknown as { sys: System }).sys;
};

/** Forces and energies of every pair with the spec's law (minimum image in the periodic box). */
const specForces = (sys: System, spec: GpuPairSpec) => {
  const st = sys.state;
  const L = [0, 1, 2].map((d) => st.box.hi[d] - st.box.lo[d]);
  const f = new Float64Array(3 * st.n);
  let evdwl = 0, ecoul = 0;
  for (let i = 0; i < st.n; i++) {
    for (let j = i + 1; j < st.n; j++) {
      const d = [0, 1, 2].map((k) => { const v = st.x[3 * i + k] - st.x[3 * j + k]; return v - L[k] * Math.round(v / L[k]); });
      const rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      const r = pairSpecEval(spec, st.type[i], st.type[j], rsq, st.q[i], st.q[j]);
      evdwl += r.evdwl; ecoul += r.ecoul;
      for (let k = 0; k < 3; k++) { f[3 * i + k] += r.fpair * d[k]; f[3 * j + k] -= r.fpair * d[k]; }
    }
  }
  return { f, evdwl, ecoul };
};

describe('GPU pair-style specs against the CPU styles', () => {
  for (const [name, lines] of Object.entries(STYLES)) {
    it(name, async () => {
      const sys = await systemFor(lines);
      const st = sys.state;
      const p = sys.ff.pair!;
      const spec = gpuPairSpec(p, st.ntypes, st.units.qqr2e * (p.coulConstScale ?? 1) / sys.ff.dielectric);
      if (typeof spec === 'string') throw new Error(spec);
      expect(spec.maxCutoff).toBeLessThan((st.box.hi[0] - st.box.lo[0]) / 2);
      const got = specForces(sys, spec);
      const acc = sys.ff.acc;
      expect(got.evdwl).toBeCloseTo(acc.evdwl, 8);
      expect(got.ecoul).toBeCloseTo(acc.ecoul, 8);
      let worst = 0, scale = 0;
      for (let k = 0; k < 3 * st.n; k++) { worst = Math.max(worst, Math.abs(got.f[k] - st.f[k])); scale = Math.max(scale, Math.abs(st.f[k])); }
      expect(worst / scale).toBeLessThan(1e-12);
    });
  }

  it('the GPU accelerator takes these styles, and names the ones it does not run', async () => {
    const gpu = { kind: 'webgpu', label: 'WebGPU · test', compute: () => ({ pe: 0, virial: 0 }), dispose() {} } as unknown as ForceBackend;
    const withFix = async (lines: string) => {
      const sys = await systemFor(`${lines}\nfix 1 all nve`);
      return accelerator(sys, gpu);
    };
    const ok = await withFix(STYLES.buck);
    expect(ok.reason).toBeNull();
    expect(ok.accel?.spec?.style).toBe(3);
    const soft = await withFix('pair_style soft 3.0\npair_coeff * * 10.0');
    expect(soft.reason).toMatch(/pair style soft \(the GPU path implements lj\/cut, lj\/cut\/coul\/cut, coul\/cut, buck, buck\/coul\/cut, morse only\)/);
  });
});
